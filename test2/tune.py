"""Standalone hyper-parameter search for the test2 3-block CNN (FashionMNIST).

Why standalone instead of importing test2/test2.py:
  * test2.py runs dataset setup / opens its CSV at import time (side effects)
  * its Mynet -> Myblock constructor currently disagrees on kernel_size, so it
    cannot be instantiated as written
This file re-implements the *same* design — conv(k)-BN-act-conv(k)-BN-(+res)-act-
(pool)-dropout, GAP head — and adds the training-side pieces the current script
lacks: per-trial augmentation config, cosine annealing with warm-up, label
smoothing, AMP, val-based selection.

Data path: the whole 60k set is kept as a uint8 tensor in RAM (47 MB) and every
batch is normalised + augmented on the GPU. That avoids the per-sample PIL
decode, which is the bottleneck on Windows (no worker processes under the
sandbox, so a DataLoader would be ~6x slower).

Usage:
  python test2/tune.py --trials 6 --epochs 30
  python test2/tune.py --baseline-only --epochs 2
"""
import argparse
import copy
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MEAN, STD = (0.286,), (0.353,)
NUM_CLASSES, SEED = 10, 42


# ---------------------------------------------------------------- model (same design as test2.py)
class Block(nn.Module):
    def __init__(self, cin, cout, dropout, use_bn, act_fn, pool, res, k, drop_mode="stream"):
        super().__init__()
        pad = k // 2
        self.drop_mode = drop_mode
        self.conv1 = nn.Conv2d(cin, cout, k, padding=pad, bias=not use_bn)
        self.conv2 = nn.Conv2d(cout, cout, k, padding=pad, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
        self.res = res
        if res:
            self.short = nn.Identity() if cin == cout else nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout) if use_bn else nn.Identity())
            if use_bn:
                nn.init.zeros_(self.bn2.weight)   # block starts as identity
        self.act = act_fn()                        # one instance per block
        self.pool = pool
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.res:
            if self.drop_mode == "branch":
                y = self.drop(y)          # dropout inside the residual branch
            y = y + self.short(x)
            y = self.act(y)
            if self.pool is not None:
                y = self.pool(y)
            if self.drop_mode != "branch":
                y = self.drop(y)          # legacy: dropout on the residual stream
            return y
        y = self.act(y)
        if self.pool is not None:
            y = self.pool(y)
        return self.drop(y)


class Net(nn.Module):
    def __init__(self, stages, act_fn, dropout, use_bn, res, pool, k, pool_last=False,
                 drop_mode="stream", head="gap"):
        """stages: [(width, n_blocks), ...]; pooling only at stage boundaries."""
        super().__init__()
        layers, cin, size = [], 1, 28
        for si, (w, nb) in enumerate(stages):
            for b in range(nb):
                at_stage_end = b == nb - 1
                at_last_stage = si == len(stages) - 1
                p = pool if (at_stage_end and (pool_last or not at_last_stage)) else None
                layers.append(Block(cin, w, dropout, use_bn, act_fn, p, res, k,
                                    drop_mode=drop_mode))
                cin = w
                if p is not None:
                    size //= 2
        flat = cin * size * size
        if head == "flat":
            layers += [nn.Flatten(1), nn.Linear(flat, NUM_CLASSES)]
        elif head == "mlp":
            layers += [nn.Flatten(1), nn.Linear(flat, 256), nn.ReLU(),
                       nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES)]
        else:
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(cin, NUM_CLASSES)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- data + augmentation
MILD = v2.Compose([v2.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1))])
STRONG = v2.Compose([
    v2.RandomAffine(15, translate=(0.10, 0.10), scale=(0.8, 1.2), shear=8),
    v2.RandomHorizontalFlip(),
    v2.RandomErasing(p=0.30, scale=(0.02, 0.20)),
])
HEAVY = v2.Compose([
    v2.RandomAffine(20, translate=(0.12, 0.12), scale=(0.75, 1.25), shear=10),
    v2.RandomHorizontalFlip(),
    v2.RandomErasing(p=0.50, scale=(0.02, 0.25)),
])
# larger rotation + flip + *small* scale: keeps content, changes pose
ROT = v2.Compose([
    v2.RandomAffine(22, translate=(0.10, 0.10), scale=(0.9, 1.1), shear=8),
    v2.RandomHorizontalFlip(),
    v2.RandomErasing(p=0.30, scale=(0.02, 0.20)),
])
# larger translation: the 12-block model showed a 2-point drop under a 1px shift TTA
STRONG2 = v2.Compose([
    v2.RandomAffine(15, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=8),
    v2.RandomHorizontalFlip(),
    v2.RandomErasing(p=0.30, scale=(0.02, 0.20)),
])
AUGS = {"mild": MILD, "strong": STRONG, "heavy": HEAVY, "rot": ROT, "strong2": STRONG2}


def build_aug(s):
    """Augmentation strength s in [0,1]: 1 = aggressive, 0 = near-clean."""
    return v2.Compose([
        v2.RandomAffine(10 + 10 * s,
                        translate=(0.05 + 0.05 * s, 0.05 + 0.05 * s),
                        scale=(0.9 - 0.2 * s, 1.1 + 0.2 * s),
                        shear=4 + 6 * s),
        v2.RandomHorizontalFlip(),
        v2.RandomErasing(p=0.1 + 0.3 * s, scale=(0.02, 0.20)),
    ])


def build_data():
    """FashionMNIST as uint8 tensors; same 90/10 split as test2.py (seed 42)."""
    ds = datasets.FashionMNIST(DATA, train=True, download=True)
    te = datasets.FashionMNIST(DATA, train=False, download=True)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(ds), generator=g)
    n_val = len(ds) // 10
    tr, va = perm[:-n_val], perm[-n_val:]
    return (ds.data.unsqueeze(1)[tr], ds.targets[tr],
            ds.data.unsqueeze(1)[va], ds.targets[va],
            te.data.unsqueeze(1), te.targets)


def batches(x, y, bs, shuffle, generator=None):
    n = x.size(0)
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    stop = n - (n % bs) if shuffle else n
    for i in range(0, stop, bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


_M = torch.tensor(MEAN).view(1, 1, 1, 1).cuda()
_S = torch.tensor(STD).view(1, 1, 1, 1).cuda()


def to_gpu(xb, yb):
    return xb.cuda(non_blocking=True).float().div_(255.0).sub_(_M).div_(_S), yb.cuda(non_blocking=True)


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, x, y, loss_fn, bs=1000, cl=False):
    model.eval()
    ls = cor = tot = 0
    for xb, yb in batches(x, y, bs, False):
        xb, yb = to_gpu(xb, yb)
        if cl:
            xb = xb.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb)
        out = out.float()
        ls += loss_fn(out, yb).item() * yb.size(0)
        cor += (out.argmax(1) == yb).sum().item()
        tot += yb.size(0)
    return ls / tot, cor / tot


def run_trial(cfg, epochs, warmup_epochs, trial_id, out_dir, log, data, seed=SEED):
    torch.manual_seed(seed)
    xtr, ytr, xva, yva, xte, yte = data
    aug = AUGS[cfg["aug"]]
    aug_sched = cfg.get("aug_sched")
    pool = {"max": nn.MaxPool2d(2), "avg": nn.AvgPool2d(2), "none": None}[cfg["pool"]]

    stages = cfg.get("stages") or [(w, 1) for w in cfg["widths"]]
    act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[cfg.get("act", "relu")]
    model = Net(stages, act_fn, cfg["dropout"], True, cfg["res"], pool, cfg["k"],
                pool_last=cfg.get("pool_last", True),
                drop_mode=cfg.get("drop_mode", "stream"),
                head=cfg.get("head", "gap")).cuda()
    cl = cfg.get("cl", False)
    if cl:
        model = model.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters())
    use_ema = cfg.get("ema", False)
    ema_params, shadow = None, None
    if use_ema:
        ema_params = [p.detach().clone() for p in model.parameters()]
        shadow = copy.deepcopy(model)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"],
                                fused=cfg.get("fused", False))
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["ls"])

    steps = len(xtr) // cfg["bs"]
    total, warm = steps * epochs, steps * warmup_epochs
    sched_mode = cfg.get("sched", "cosine")

    def lam(step):
        if step < warm:
            return (step + 1) / warm
        if sched_mode == "const":
            return 1.0
        p = (step - warm) / max(1, total - warm)
        p = min(max(p, 0.0), 1.0)
        if sched_mode == "cosine":
            return 0.5 * (1 + math.cos(math.pi * p))
        if sched_mode == "step":
            return 1.0 if p < 0.60 else (0.1 if p < 0.85 else 0.01)
        if sched_mode == "onecycle":
            return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p))
        if sched_mode == "exp":
            return 0.02 ** p
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    gen = torch.Generator().manual_seed(seed)
    log(f"[{trial_id}] {cfg}  params={n_par:,}")
    best = {"val": -1.0, "ep": 0, "state": None}
    best_ema = {"val": -1.0, "ep": 0, "state": None}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if aug_sched:
            p = (ep - 1) / max(1, epochs - 1)
            s = (1.0 - p) if aug_sched == "linear" else 0.5 * (1 + math.cos(math.pi * p))
            aug = build_aug(s)
        model.train()
        ls_t = torch.zeros((), device="cuda")
        cor_t = torch.zeros((), device="cuda", dtype=torch.long)
        tot = 0
        for xb, yb in batches(xtr, ytr, cfg["bs"], True, gen):
            xb, yb = to_gpu(xb, yb)
            if cl:
                xb = xb.to(memory_format=torch.channels_last)
            xb = aug(xb)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
            loss = loss_fn(out.float(), yb)
            loss.backward()
            opt.step()
            sched.step()
            if ema_params is not None:
                with torch.no_grad():
                    for e_p, p_ in zip(ema_params, model.parameters()):
                        e_p.mul_(0.999).add_(p_, alpha=0.001)
            ls_t += loss.detach() * yb.size(0)          # keep on GPU: no per-step sync
            cor_t += (out.argmax(1) == yb).sum()
            tot += yb.size(0)
        tr_loss, tr_acc = ls_t.item() / tot, cor_t.item() / tot
        ema_acc = None
        if xva is not None:
            va_loss, va_acc = evaluate(model, xva, yva, loss_fn, cl=cl)
            if va_acc > best["val"]:
                best = {"val": va_acc, "ep": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if ema_params is not None:
                with torch.no_grad():
                    for s_p, e_p in zip(shadow.parameters(), ema_params):
                        s_p.copy_(e_p)
                    for s_b, m_b in zip(shadow.buffers(), model.buffers()):
                        s_b.copy_(m_b)
                ema_loss, ema_acc = evaluate(shadow, xva, yva, loss_fn, cl=cl)
                if ema_acc > best_ema["val"]:
                    best_ema = {"val": ema_acc, "ep": ep,
                                "state": {k: v.detach().cpu().clone()
                                          for k, v in shadow.state_dict().items()}}
        else:
            va_acc = float("nan")
            best = {"val": float("nan"), "ep": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ep % 10 == 0 or ep == 1:
            extra = f" ema={ema_acc:.4f}" if ema_acc is not None else ""
            log(f"  [{trial_id}] ep{ep:>2d} train={tr_acc:.4f} val={va_acc:.4f} "
                f"best={best['val']:.4f}{extra} lr={opt.param_groups[0]['lr']:.2e}")
        if cfg.get("save_every") and ep % cfg["save_every"] == 0:
            torch.save({"state_dict": {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()},
                        "config": cfg, "epoch": ep},
                       out_dir / f"ckpt_ep{ep}.pt")

    chosen, tag = best, "raw"
    if use_ema and best_ema["val"] > best["val"]:
        chosen, tag = best_ema, "ema"
    model.load_state_dict(chosen["state"])
    te_loss, te_acc = evaluate(model, xte, yte, loss_fn, cl=cl)
    secs = time.time() - t0
    log(f"[{trial_id}] DONE best_ep={chosen['ep']} val={chosen['val']:.4f} test={te_acc:.4f} "
        f"weights={tag}" + (f" (raw val={best['val']:.4f}, ema val={best_ema['val']:.4f})"
                            if use_ema else "") + f" ({secs:.0f}s)")
    torch.save({"state_dict": chosen["state"], "config": cfg, "val_acc": chosen["val"],
                "test_acc": te_acc, "best_epoch": chosen["ep"], "weights": tag},
               out_dir / f"trial_{trial_id}.pt")
    return dict(trial=trial_id, params=n_par, best_epoch=chosen["ep"], val_acc=chosen["val"],
                ema_val=best_ema["val"] if use_ema else 0.0,
                test_acc=te_acc, weights=tag, secs=round(secs, 1),
                widths=str([w for w, _ in stages]), dropout=cfg["dropout"], lr=cfg["lr"],
                wd=cfg["wd"], ls=cfg["ls"], aug=cfg["aug"], bs=cfg["bs"])


def sample_config(rng, wide=False):
    return {
        "widths": [64, 128, 256] if wide else [32, 64, 128],
        "dropout": float(rng.choice([0.0, 0.1, 0.2])),
        "lr": float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))),
        "wd": float(rng.choice([1e-4, 1e-3, 1e-2])),
        "ls": float(rng.choice([0.0, 0.1])),
        "aug": str(rng.choice(["mild", "strong"])),
        "bs": int(rng.choice([64, 128])),
        "pool": "max", "res": False, "k": 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--only", default="", help="run only the config with this name")
    ap.add_argument("--full", action="store_true", help="train on all 60k (no val split)")
    ap.add_argument("--preset", choices=["screen", "confirm", "deep", "next", "b_opt", "iso", "iso_lr2", "const", "fast", "sched", "ema", "final", "arch3", "deepfix", "branchdrop", "deep150", "fatstem", "combo", "d16reg", "bigfc", "combo2", "augsched", "pyramid", "pyramid64", "actpool"], default="screen")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = default)")
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N epochs (0 = off)")
    ap.add_argument("--out-dir", default="", help="explicit output directory (for matrix runs)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = HERE / "tune_logs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "tune.log"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    torch.backends.cudnn.benchmark = True
    data = build_data()
    if args.full:
        xtr, ytr, xva, yva, xte, yte = data
        data = (torch.cat([xtr, xva], 0), torch.cat([ytr, yva], 0), None, None, xte, yte)
    rng = np.random.default_rng(args.seed)

    baseline = {"widths": [32, 64, 128], "dropout": 0.1, "lr": 1e-4, "wd": 0.0,
                "ls": 0.0, "aug": "mild", "bs": 64, "pool": "max", "res": False, "k": 3}
    # informed candidates around "strong aug + cosine"
    informed = [
        {"widths": [32, 64, 128], "dropout": 0.1, "lr": 1e-3, "wd": 1e-3,
         "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max", "res": False, "k": 3},
        {"widths": [32, 64, 128], "dropout": 0.2, "lr": 1e-3, "wd": 1e-3,
         "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max", "res": False, "k": 3},
        {"widths": [32, 64, 128], "dropout": 0.1, "lr": 3e-4, "wd": 1e-3,
         "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max", "res": False, "k": 3},
        {"widths": [32, 64, 128], "dropout": 0.1, "lr": 1e-3, "wd": 1e-3,
         "ls": 0.1, "aug": "mild", "bs": 128, "pool": "max", "res": False, "k": 3},
        {"widths": [64, 128, 256], "dropout": 0.2, "lr": 1e-3, "wd": 1e-3,
         "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max", "res": False, "k": 3},
    ]
    configs = [("baseline", baseline)]
    if args.preset == "confirm":
        # 100-epoch confirmation: does depth/width close the gap, or is it training budget?
        configs = [
            ("A_3block", {"widths": [32, 64, 128], "dropout": 0.1, "lr": 1e-3, "wd": 1e-3,
                          "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max",
                          "res": False, "k": 3, "pool_last": True}),
            ("B_4stage", {"widths": [32, 64, 128, 256], "dropout": 0.1, "lr": 1e-3, "wd": 1e-3,
                          "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max",
                          "res": False, "k": 3, "pool_last": False}),
        ]
    elif args.preset == "deep":
        # stage-based depth: pool only between stages, identity residual inside a stage
        configs = [
            ("C_deep12", {"stages": [(32, 1), (64, 2), (128, 2), (256, 1)],
                          "dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                          "aug": "strong", "bs": 128, "pool": "max",
                          "res": True, "k": 3, "pool_last": False}),
        ]
    elif args.preset == "next":
        # B was not overfitting (train 94.9 / val 94.7), so spend the headroom on
        # capacity (wider) and on stronger regularisation (heavier aug).
        configs = [
            ("D_wide", {"stages": [(64, 1), (128, 1), (256, 1), (512, 1)],
                        "dropout": 0.2, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                        "aug": "strong", "bs": 128, "pool": "max",
                        "res": False, "k": 3, "pool_last": False}),
            ("E_heavy", {"stages": [(32, 1), (64, 1), (128, 1), (256, 1)],
                         "dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                         "aug": "heavy", "bs": 128, "pool": "max",
                         "res": False, "k": 3, "pool_last": False}),
        ]
    elif args.preset == "b_opt":
        # B is the best model and shows almost no overfitting (train 94.9 / val 94.7).
        # Two isolated changes on B: smaller batch (more updates) and rotation-heavy aug.
        base_b = {"stages": [(32, 1), (64, 1), (128, 1), (256, 1)],
                  "dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                  "bs": 128, "pool": "max", "res": False, "k": 3, "pool_last": False}
        configs = [
            ("F1_bs64", {**base_b, "bs": 64, "aug": "strong"}),
            ("F2_rot", {**base_b, "aug": "rot"}),
        ]
    elif args.preset == "iso":
        # constant width everywhere -> every residual shortcut is a free identity,
        # and the capacity that B put in stage4 is moved into the shallow stages
        configs = [
            ("G_iso64", {"stages": [(64, 2), (64, 2), (64, 2), (64, 2)],
                         "dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                         "aug": "strong", "bs": 128, "pool": "max",
                         "res": True, "k": 3, "pool_last": False}),
        ]
    elif args.preset == "iso_lr2":
        configs = [
            ("G_iso64_lr2e3", {"stages": [(64, 2), (64, 2), (64, 2), (64, 2)],
                               "dropout": 0.1, "lr": 2e-3, "wd": 1e-3, "ls": 0.1,
                               "aug": "strong", "bs": 128, "pool": "max",
                               "res": True, "k": 3, "pool_last": False}),
        ]
    elif args.preset == "const":
        # same isotropic net, but no cosine decay: warmup then a flat LR
        configs = [
            ("H_const1e3", {"stages": [(64, 2), (64, 2), (64, 2), (64, 2)],
                            "dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                            "aug": "strong", "bs": 128, "pool": "max",
                            "res": True, "k": 3, "pool_last": False, "sched": "const"}),
        ]
    elif args.preset == "fast":
        # same isotropic net, throughput-optimised: bs256 + fused AdamW + channels_last
        configs = [
            ("G_fast", {"stages": [(64, 2), (64, 2), (64, 2), (64, 2)],
                        "dropout": 0.1, "lr": 2e-3, "wd": 1e-3, "ls": 0.1,
                        "aug": "strong", "bs": 256, "pool": "max",
                        "res": True, "k": 3, "pool_last": False,
                        "fused": True, "cl": True}),
        ]
    elif args.preset == "sched":
        # four LR schedules, same isotropic net; run them as separate processes
        base = {"stages": [(64, 2), (64, 2), (64, 2), (64, 2)],
                "dropout": 0.1, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 256, "pool": "max", "res": True, "k": 3,
                "pool_last": False, "fused": True, "cl": True}
        configs = [
            ("S1_cosine", {**base, "lr": 2e-3, "sched": "cosine"}),
            ("S2_step", {**base, "lr": 2e-3, "sched": "step"}),
            ("S3_onecycle", {**base, "lr": 2e-3, "sched": "onecycle"}),
            ("S4_exp", {**base, "lr": 2e-3, "sched": "exp"}),
        ]
    elif args.preset == "ema":
        configs = [
            ("G_ema", {"stages": [(64, 2), (64, 2), (64, 2), (64, 2)],
                       "dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                       "aug": "strong", "bs": 128, "pool": "max",
                       "res": True, "k": 3, "pool_last": False,
                       "fused": True, "cl": True, "ema": True}),
        ]
    elif args.preset == "final":
        configs = [
            ("G_150", {"stages": [(64, 2), (64, 2), (64, 2), (64, 2)],
                       "dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1,
                       "aug": "strong", "bs": 128, "pool": "max",
                       "res": True, "k": 3, "pool_last": False,
                       "fused": True, "cl": True}),
        ]
    elif args.preset == "arch3":
        # three capacity layouts on top of the isotropic winner (G), all identity shortcuts
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True}
        configs = [
            ("G_deep12", {**base, "stages": [(64, 3), (64, 3), (64, 3), (64, 3)]}),
            ("G_wide80", {**base, "stages": [(80, 2), (80, 2), (80, 2), (80, 2)]}),
            ("G_inv96", {**base, "stages": [(96, 2), (96, 2), (64, 2), (64, 2)]}),
        ]
    elif args.preset == "deepfix":
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True}
        configs = [
            ("G_deep12_branchdrop", {**base, "stages": [(64, 3), (64, 3), (64, 3), (64, 3)],
                                     "drop_mode": "branch"}),
        ]
    elif args.preset == "branchdrop":
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch"}
        configs = [
            ("G_bd", {**base, "stages": [(64, 2), (64, 2), (64, 2), (64, 2)]}),
            ("G_inv96_bd", {**base, "stages": [(96, 2), (96, 2), (64, 2), (64, 2)]}),
        ]
    elif args.preset == "deep150":
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong2",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch"}
        configs = [
            ("D12_t15", {**base, "stages": [(64, 3), (64, 3), (64, 3), (64, 3)]}),
            ("D16_t15", {**base, "stages": [(64, 4), (64, 4), (64, 4), (64, 4)]}),
        ]
    elif args.preset == "fatstem":
        # widen ONLY the first stage, keep the rest identical to G
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "stream"}
        configs = [
            ("G_fatstem96", {**base, "stages": [(96, 2), (64, 2), (64, 2), (64, 2)]}),
        ]
    elif args.preset == "combo":
        # deep (12 blocks) + wider shallow stages + branch dropout, meant for --full (60k)
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch"}
        configs = [
            ("C_deepwide", {**base, "stages": [(80, 3), (80, 3), (64, 3), (64, 3)]}),
        ]
    elif args.preset == "d16reg":
        # 16-block model that overfit (train 96.9 / test 94.3): does stronger
        # regularisation close the train-test gap at a fixed 100 epochs?
        base = {"lr": 1e-3, "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max",
                "res": True, "k": 3, "pool_last": False, "fused": True, "cl": True,
                "drop_mode": "branch", "save_every": 10,
                "stages": [(64, 4), (64, 4), (64, 4), (64, 4)]}
        configs = [
            ("D16_base", {**base, "dropout": 0.1, "wd": 1e-3}),
            ("D16_reg", {**base, "dropout": 0.2, "wd": 5e-3}),
        ]
    elif args.preset == "bigfc":
        # does GAP throw away useful spatial layout? compare flat-576 and 576-256-10 heads
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch",
                "stages": [(64, 3), (64, 3), (64, 3), (64, 3)]}
        configs = [
            ("deep12_flat", {**base, "head": "flat"}),
            ("deep12_mlp", {**base, "head": "mlp"}),
        ]
    elif args.preset == "combo2":
        # clean control for "wide shallow + 12 blocks": 54k split, val-based selection,
        # plus a stronger-regularisation twin at the same capacity.
        base = {"lr": 1e-3, "ls": 0.1, "aug": "strong", "bs": 128, "pool": "max",
                "res": True, "k": 3, "pool_last": False, "fused": True, "cl": True,
                "drop_mode": "branch", "stages": [(80, 3), (80, 3), (64, 3), (64, 3)]}
        configs = [
            ("C_clean", {**base, "dropout": 0.1, "wd": 1e-3}),
            ("C_reg", {**base, "dropout": 0.2, "wd": 5e-3}),
        ]
    elif args.preset == "augsched":
        # augmentation warm-down: strong early, near-clean late
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch",
                "stages": [(64, 3), (64, 3), (64, 3), (64, 3)]}
        configs = [
            ("A1_linear", {**base, "aug_sched": "linear"}),
            ("A2_cosine", {**base, "aug_sched": "cosine"}),
        ]
    elif args.preset == "pyramid":
        # revive the classic pyramid, but with the two fixes that landed later:
        # branch dropout + augmentation warm-down
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch",
                "aug_sched": "cosine"}
        configs = [
            ("P1_1blk", {**base, "stages": [(32, 1), (64, 1), (128, 1), (256, 1)]}),
            ("P2_2blk", {**base, "stages": [(32, 2), (64, 2), (128, 2), (256, 2)]}),
        ]
    elif args.preset == "pyramid64":
        # pyramid that KEEPS the first stage at 64 (same as the best model) and only
        # widens deeper; branch dropout + aug warm-down, the two fixes that landed later
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "pool": "max", "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch",
                "aug_sched": "cosine"}
        configs = [
            ("Q1_64_128_256_256", {**base, "stages": [(64, 1), (128, 1), (256, 1), (256, 1)]}),
            ("Q2_64_128_256_512", {**base, "stages": [(64, 1), (128, 1), (256, 1), (512, 1)]}),
        ]
    elif args.preset == "actpool":
        # single-variable changes on the A2 recipe: activation and pooling type
        base = {"dropout": 0.1, "lr": 1e-3, "wd": 1e-3, "ls": 0.1, "aug": "strong",
                "bs": 128, "res": True, "k": 3, "pool_last": False,
                "fused": True, "cl": True, "drop_mode": "branch",
                "aug_sched": "cosine",
                "stages": [(64, 3), (64, 3), (64, 3), (64, 3)]}
        configs = [
            ("X1_gelu", {**base, "act": "gelu", "pool": "max"}),
            ("X2_avg", {**base, "act": "relu", "pool": "avg"}),
            ("X3_gelu_avg", {**base, "act": "gelu", "pool": "avg"}),
        ]
    elif not args.baseline_only:
        for i, cfg in enumerate(informed[:max(0, args.trials - 1)], start=1):
            configs.append((f"t{i}", cfg))
        extra = args.trials - len(configs)
        for i in range(extra):
            configs.append((f"r{i}", sample_config(rng)))

    summary = []
    if args.only:
        configs = [c for c in configs if c[0] == args.only]
        if not configs:
            raise SystemExit(f"no config named {args.only}")
    for tid, cfg in configs:
        if args.save_every:
            cfg["save_every"] = args.save_every
        summary.append(run_trial(cfg, args.epochs, args.warmup_epochs, tid, out_dir, log,
                                 data, seed=args.seed))

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    ranked = sorted(summary, key=lambda r: r["val_acc"], reverse=True)
    log("\n=== ranked by val acc ===")
    for r in ranked:
        log(f"  {r['trial']:<9} val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
            f"ep={r['best_epoch']:>2d} lr={r['lr']:.4g} wd={r['wd']:.0e} dr={r['dropout']} "
            f"ls={r['ls']} aug={r['aug']:<6} bs={r['bs']:<3} widths={r['widths']} ({r['secs']}s)")
    log(f"\nbest: {ranked[0]['trial']}  val={ranked[0]['val_acc']:.4f}  test={ranked[0]['test_acc']:.4f}")
    log(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
