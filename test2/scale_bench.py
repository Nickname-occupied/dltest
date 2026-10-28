"""Measure aggregate throughput vs number of parallel workers.

Each data point launches `jobs` workers, every worker trains the same config for
`--epochs` epochs, and we report the wall time plus aggregate epochs/minute.
With OMP threads pinned per worker this answers "how many processes actually
saturate this GPU" instead of guessing.

Usage:
  python test2/scale_bench.py --epochs 3 --jobs 1 2 3 4 --threads 2
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run_group(jobs, epochs, threads, cfg, preset):
    procs, logs = [], []
    t0 = time.time()
    for i in range(jobs):
        out = HERE / "scale_bench" / f"{cfg}_j{jobs}_w{i}"
        out.mkdir(parents=True, exist_ok=True)
        log = out / "run.log"
        cmd = [sys.executable, str(HERE / "tune.py"), "--preset", preset,
               "--only", cfg, "--seed", str(42 + i), "--epochs", str(epochs),
               "--threads", str(threads), "--out-dir", str(out)]
        env = os.environ.copy()
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(threads)
        logs.append(log.open("w", encoding="utf-8"))
        procs.append(subprocess.Popen(cmd, stdout=logs[-1], stderr=subprocess.STDOUT,
                                      cwd=str(ROOT), env=env))
    for p in procs:
        p.wait()
    for l in logs:
        l.close()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="arch3")
    ap.add_argument("--cfg", default="G_deep12")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    print(f"cfg={args.cfg} preset={args.preset} epochs={args.epochs} threads={args.threads}")
    print(f"{'jobs':>4s} {'wall(s)':>8s} {'per-worker s/ep':>16s} {'aggregate ep/min':>18s}")
    baseline = None
    for j in args.jobs:
        dt = run_group(j, args.epochs, args.threads, args.cfg, args.preset)
        per = dt / args.epochs
        agg = j * args.epochs / dt * 60
        if baseline is None:
            baseline = agg
        print(f"{j:>4d} {dt:>8.1f} {per:>16.1f} {agg:>18.2f}  (x{agg / baseline:.2f})", flush=True)


if __name__ == "__main__":
    main()
