"""test2_ai.py — FashionMNIST CNN 消融实验（test1 风格）。

BASE 是调优后的 12-block 网络（4 个 stage，每个 3 块，等宽 64）。
一次运行跑完全部 8 组对照；已完成的 run 会按 checkpoint 自动跳过。
每个 epoch 写一行 results.csv，每个 run 结束后评估一次 test 并写 summary.csv。

运行：
    python test2_ai.py

想只跑某几组，改文件顶部的 GROUPS 列表。
想重跑某个 run，删掉 logs_ai/weights/<name>.pt。
"""
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data"
LOG_DIR = HERE / "logs_ai"
W_DIR = LOG_DIR / "weights"
LOG_DIR.mkdir(parents=True, exist_ok=True)
W_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 100
BATCH = 128
LR = 1e-3
BLOCKS = 3                      # 每个 stage 的 block 数（4 stage × 3 = 12 块）
WIDTHS = (64, 64, 64, 64)       # 每个 stage 的通道数
GROUPS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

MEAN, STD = 0.286, 0.353
NUM_CLASSES = 10
ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}
POOLS = {"max": nn.MaxPool2d(2), "avg": nn.AvgPool2d(2), "none": None}


# ---------------------------------------------------------------- 数据
def build_data():
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=True)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=True)
    perm = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(SEED))
    tr, va = perm[:-6000], perm[-6000:]
    x = train_set.data.unsqueeze(1)
    return (x[tr], train_set.targets[tr], x[va], train_set.targets[va],
            test_set.data.unsqueeze(1), test_set.targets)


XT, YT, XV, YV, XE, YE = build_data()

MEAN_T = torch.tensor([MEAN]).view(1, 1, 1, 1).cuda()
STD_T = torch.tensor([STD]).view(1, 1, 1, 1).cuda()


def to_gpu(xb, yb):
    xb = xb.cuda(non_blocking=True).float().div_(255.0).sub_(MEAN_T).div_(STD_T)
    xb = xb.to(memory_format=torch.channels_last)
    return xb, yb.cuda(non_blocking=True)


def batches(x, y, bs, shuffle, generator=None, drop_last=False):
    n = x.size(0)
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    stop = n - (n % bs) if drop_last else n
    for i in range(0, stop, bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


# ---------------------------------------------------------------- 数据增强
def build_aug(strength):
    """strength=1.0 强增强，0.0 接近干净数据（用于训练后期）。"""
    s = float(strength)
    return v2.Compose([
        v2.RandomAffine(10 + 10 * s,
                        translate=(0.05 + 0.05 * s, 0.05 + 0.05 * s),
                        scale=(0.9 - 0.2 * s, 1.1 + 0.2 * s),
                        shear=4 + 6 * s),
        v2.RandomHorizontalFlip(),
        v2.RandomErasing(p=0.1 + 0.3 * s),
    ])


# ---------------------------------------------------------------- 网络
class Block(nn.Module):
    """conv3x3 - BN - ReLU - conv3x3 - BN - dropout - (+shortcut) - ReLU - pool。"""

    def __init__(self, cin, cout, k, bn, act, pool, res, dropout):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv2d(cin, cout, k, padding=pad, bias=not bn)
        self.bn1 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(cout, cout, k, padding=pad, bias=not bn)
        self.bn2 = nn.BatchNorm2d(cout) if bn else nn.Identity()
        if bn and res:
            nn.init.zeros_(self.bn2.weight)      # 残差块从恒等映射开始
        self.act = act
        self.pool = pool
        self.drop = nn.Dropout(dropout)
        self.res = res
        if res:
            self.short = (nn.Identity() if cin == cout else
                          nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                        nn.BatchNorm2d(cout) if bn else nn.Identity()))

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.res:
            y = self.drop(y) + self.short(x)     # dropout 在残差分支内，相加之前
        else:
            y = self.drop(y)
        y = self.act(y)
        if self.pool is not None:
            y = self.pool(y)
        return y


class Net(nn.Module):
    def __init__(self, widths, blocks, activation, pool_name, res, bn, dropout, k, head):
        super().__init__()
        act = ACTS[activation]()
        pool = POOLS[pool_name]
        layers = []
        cin, size = 1, 28
        for si, w in enumerate(widths):
            last_stage = si == len(widths) - 1
            for b in range(blocks):
                at_stage_end = b == blocks - 1
                layers.append(Block(cin, w, k, bn, act,
                                    pool if (at_stage_end and not last_stage) else None,
                                    res, dropout))
                cin = w
            if not last_stage and pool is not None:
                size //= 2
        if head == "flat":
            layers += [nn.Flatten(1), nn.Linear(cin * size * size, NUM_CLASSES)]
        elif head == "mlp":
            layers += [nn.Flatten(1), nn.Linear(cin * size * size, 256), act(),
                       nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES)]
        else:
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(cin, NUM_CLASSES)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- 损失 / 评估 / 单轮训练
def loss_function(name):
    def ce(out, y):
        return F.cross_entropy(out, y, label_smoothing=0.1)

    def mse(out, y):
        return F.mse_loss(torch.softmax(out, 1), F.one_hot(y, NUM_CLASSES).float())

    return mse if name == "mse" else ce


@torch.no_grad()
def evaluate(model, x, y, loss_fn, bs=1000):
    model.eval()
    total_loss = correct = total = 0
    for xb, yb in batches(x, y, bs, shuffle=False):
        xb, yb = to_gpu(xb, yb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb)
        out = out.float()
        total_loss += loss_fn(out, yb).item() * yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


def train_epoch(model, optimizer, scheduler, loss_fn, bs, generator, aug):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in batches(XT, YT, bs, shuffle=True, generator=generator, drop_last=True):
        xb, yb = to_gpu(xb, yb)
        xb = aug(xb)
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb)
        out = out.float()
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------- 一个完整实验
def run(name, activation="relu", pool="max", res=True, bn=True, dropout=0.1,
        k=3, head="gap", widths=WIDTHS, blocks=BLOCKS, lr=LR, bs=BATCH,
        epochs=EPOCHS, loss_name="ce", warmdown=True):
    ckpt_path = W_DIR / f"{name}.pt"
    if ckpt_path.exists():                  # 跑过就跳过
        print(f"skip {name}")
        return

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()

    model = Net(widths, blocks, activation, pool, res, bn, dropout, k, head)
    model = model.to(memory_format=torch.channels_last).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, fused=True)
    loss_fn = loss_function(loss_name)

    steps = XT.size(0) // bs
    total = steps * epochs
    warm = steps * 3

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    generator = torch.Generator().manual_seed(SEED)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}: blocks={blocks} widths={widths} act={activation} pool={pool} "
          f"res={res} bn={bn} dropout={dropout} k={k} head={head} lr={lr} bs={bs} "
          f"loss={loss_name} params={n_par:,} ===")

    results_file = open(LOG_DIR / "results.csv", "a", newline="", encoding="utf-8")
    results_writer = csv.DictWriter(results_file, fieldnames=[
        "run", "epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
    if (LOG_DIR / "results.csv").stat().st_size == 0:
        results_writer.writeheader()

    best_val, best_ep, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        aug = build_aug(0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, epochs - 1)))) \
            if warmdown else build_aug(1.0)

        train_loss, train_acc = train_epoch(model, optimizer, scheduler, loss_fn,
                                            bs, generator, aug)
        val_loss, val_acc = evaluate(model, XV, YV, loss_fn)

        results_writer.writerow({"run": name, "epoch": epoch,
                                 "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                                 "train_loss": f"{train_loss:.6f}",
                                 "train_acc": f"{train_acc:.6f}",
                                 "val_loss": f"{val_loss:.6f}",
                                 "val_acc": f"{val_acc:.6f}"})
        results_file.flush()
        print(f"  ep{epoch:>3d} train={train_acc:.4f} val={val_acc:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val:
            best_val, best_ep = val_acc, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    results_file.close()

    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, XE, YE, loss_fn)
    torch.save({"state_dict": best_state, "name": name, "best_epoch": best_ep,
                "val_acc": best_val, "test_acc": test_acc, "config": {
                    "widths": widths, "blocks": blocks, "activation": activation,
                    "pool": pool, "res": res, "bn": bn, "dropout": dropout,
                    "k": k, "head": head, "lr": lr, "bs": bs,
                    "epochs": epochs, "loss": loss_name, "seed": SEED}}, ckpt_path)

    with open(LOG_DIR / "summary.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "best_epoch", "val_acc", "test_acc", "secs", "params",
            "blocks", "widths", "activation", "pool", "res", "bn", "dropout",
            "k", "head", "lr", "bs"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({"run": name, "best_epoch": best_ep,
                         "val_acc": f"{best_val:.6f}", "test_acc": f"{test_acc:.6f}",
                         "secs": f"{time.time() - t0:.1f}", "params": n_par,
                         "blocks": blocks, "widths": widths, "activation": activation,
                         "pool": pool, "res": res, "bn": bn, "dropout": dropout,
                         "k": k, "head": head, "lr": lr, "bs": bs})

    print(f"  -> {name}: val={best_val:.4f} test={test_acc:.4f} "
          f"(epoch {best_ep}, {time.time() - t0:.0f}s)")


def main():
    if 0 in GROUPS:
        run("1_base")
    if 1 in GROUPS:
        run("1_no_pool", pool="none")
        run("1_flat_head", head="flat")
        run("1_blocks8", blocks=2)
        run("1_blocks16", blocks=4)
        run("1_stages3", widths=(64, 64, 64))
        run("1_stages5", widths=(64, 64, 64, 64, 64))
        run("1_pyramid", widths=(64, 96, 128, 160))

    if 2 in GROUPS:
        run("2_sigmoid", activation="sigmoid")
        run("2_gelu", activation="gelu")

    if 3 in GROUPS:
        run("3_mse", loss_name="mse")

    if 4 in GROUPS:
        run("4_k5", k=5)
        run("4_w32", widths=(32, 32, 32, 32))
        run("4_w96", widths=(96, 96, 96, 96))

    if 5 in GROUPS:
        run("5_avg", pool="avg")

    if 6 in GROUPS:
        run("6_lr5e-4_bs128", lr=5e-4)
        run("6_lr2e-3_bs128", lr=2e-3)
        run("6_lr1e-3_bs64", bs=64)
        run("6_lr1e-3_bs256", bs=256)

    if 7 in GROUPS:
        run("7_dropout0.2", dropout=0.2)
        run("7_dropout0", dropout=0.0)
        run("7_no_bn", bn=False)

    if 8 in GROUPS:
        run("8_plain", res=False)


if __name__ == "__main__":
    main()
