"""Run an architecture x seed matrix with N parallel worker processes.

Why a separate launcher: the sandbox forbids pipe-creating child processes, so
each worker redirects stdout to a file instead of a pipe. Each task gets its own
output directory and log, then the script aggregates mean +/- std per config.

Usage:
  python test2/run_matrix.py --preset arch3 --epochs 100 --seeds 42 43 44 --jobs 3
  python test2/run_matrix.py --preset arch3 --epochs 1 --seeds 42 --jobs 3 --dry-run
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_CONFIGS = {
    "arch3": ["G_deep12", "G_wide80", "G_inv96"],
}


def run_one(preset, cfg, seed, epochs, runs_root, log_root, jobs, threads, dry_run=False):
    out_dir = runs_root / f"{preset}_{cfg}_s{seed}"
    log_file = log_root / f"{preset}_{cfg}_s{seed}.log"
    cmd = [sys.executable, str(HERE / "tune.py"),
           "--preset", preset, "--only", cfg,
           "--seed", str(seed), "--epochs", str(epochs),
           "--threads", str(threads),
           "--out-dir", str(out_dir)]
    if dry_run:
        print(" ".join(cmd), flush=True)
        return dict(cfg=cfg, seed=seed, rc=0, secs=0.0, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # without this every worker spawns 12 OMP threads -> N x 12 on 12 cores
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        rc = p.wait()
    return dict(cfg=cfg, seed=seed, rc=rc, secs=time.time() - t0, out_dir=out_dir)


def read_summary(out_dir):
    p = out_dir / "summary.csv"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return next(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="arch3")
    ap.add_argument("--configs", default="", help="comma list; default from preset")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3, help="intra-op threads per worker")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = (args.configs.split(",") if args.configs
               else DEFAULT_CONFIGS.get(args.preset, []))
    if not configs:
        raise SystemExit("no configs for preset, pass --configs")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs_root = HERE / "matrix_runs" / f"{args.preset}_{stamp}"
    log_root = runs_root / "logs"
    tasks = [(args.preset, c, s, args.epochs, runs_root, log_root, args.jobs, args.threads,
              args.dry_run) for c in configs for s in args.seeds]

    print(f"{len(tasks)} runs, {args.jobs} parallel, {args.epochs} epochs each", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(lambda t: run_one(*t), tasks):
            results.append(r)
            print(f"  done {r['cfg']} seed={r['seed']} rc={r['rc']} {r['secs']:.0f}s", flush=True)
    if args.dry_run:
        return

    print(f"\ntotal wall {time.time() - t0:.0f}s\n")
    rows = []
    for c in configs:
        vals, tests = [], []
        for r in results:
            if r["cfg"] != c or r["rc"] != 0:
                continue
            s = read_summary(r["out_dir"])
            if s:
                vals.append(float(s["val_acc"]))
                tests.append(float(s["test_acc"]))
        if vals:
            import statistics as st
            rows.append((c, len(vals), st.mean(vals) * 100, st.pstdev(vals) * 100,
                         st.mean(tests) * 100, st.pstdev(tests) * 100))
    print(f"{'config':12s} {'n':>2s} {'val mean':>9s} {'val std':>8s} {'test mean':>10s} {'test std':>9s}")
    for c, n, v, vs, t, ts in rows:
        print(f"{c:12s} {n:>2d} {v:>8.2f}% {vs:>7.2f}% {t:>9.2f}% {ts:>8.2f}%")
    print(f"\noutputs -> {runs_root}")


if __name__ == "__main__":
    main()
