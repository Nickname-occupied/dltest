"""Fine-tune an ImageNet-pretrained backbone on FashionMNIST.

Protocol (kept deliberately close to test1/test2):
  * 90/10 train/val split, seed 42, same split for every run
  * checkpoint selection on validation accuracy only
  * the 10k test set is touched exactly once, at the end, for the val winner
  * FashionMNIST 28x28 -> resize to --size -> replicate to 3 channels -> ImageNet stats

Two phases:
  1. head-only warm-up for --freeze-epochs (backbone frozen)
  2. full fine-tune with cosine LR

Usage:
  python test2/test2_pretrained.py --arch resnet18 --size 224 --epochs 8
  python test2/test2_pretrained.py --arch convnext_tiny --size 224 --epochs 10
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# pretrained weights must land inside the workspace (the sandbox cannot write ~/.cache)
os.environ.setdefault("TORCH_HOME", str(HERE.parent / ".cache" / "torch"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18", help="torchvision model name")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4, help="full fine-tune LR")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=0, help="debug: cap training samples")
    p.add_argument("--out-dir", default=str(HERE / "logs_pretrained"))
    return p.parse_args()


def replace_head(model, num_classes=NUM_CLASSES):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        c = model.classifier
        if isinstance(c, nn.Linear):
            model.classifier = nn.Linear(c.in_features, num_classes)
        else:  # e.g. ConvNeXt: LayerNorm -> Flatten -> Linear
            model.classifier[-1] = nn.Linear(c[-1].in_features, num_classes)
    else:
        raise ValueError(f"don't know how to replace the head of {type(model).__name__}")


def head_parameters(model):
    if hasattr(model, "fc"):
        return list(model.fc.parameters())
    return list(model.classifier.parameters())


def set_backbone_frozen(model, frozen):
    head_ids = {id(p) for p in head_parameters(model)}
    for p in model.parameters():
        p.requires_grad = (not frozen) or (id(p) in head_ids)


def build_loaders(args):
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(args.size, antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(args.size, antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_root = HERE.parent / "data"
    full_train = datasets.FashionMNIST(data_root, train=True, download=True, transform=train_tf)
    # the eval view needs its own copy: same underlying split, different transform
    full_train_eval = datasets.FashionMNIST(data_root, train=True, download=False, transform=eval_tf)
    test_ds = datasets.FashionMNIST(data_root, train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(args.seed)
    train_sub, val_sub = random_split(full_train, [0.9, 0.1], generator=g)
    g = torch.Generator().manual_seed(args.seed)
    _, val_sub_eval = random_split(full_train_eval, [0.9, 0.1], generator=g)
    if args.limit_train:
        train_sub = torch.utils.data.Subset(train_sub, range(min(args.limit_train, len(train_sub))))

    loader_kw = dict(num_workers=args.workers, pin_memory=True,
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), drop_last=True,
                              **loader_kw)
    val_loader = DataLoader(val_sub_eval, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader, len(val_sub_eval), len(test_ds)


@torch.no_grad()
def evaluate(model, loader, crit, device, max_batches=0):
    model.eval()
    loss_sum = correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = crit(out, y)
        loss_sum += loss.float().item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arch}_{args.size}_bs{args.batch_size}_lr{args.lr}_fz{args.freeze_epochs}"
    log_path = out_dir / f"{tag}.log"
    csv_path = out_dir / f"{tag}.csv"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, n_val, n_test = build_loaders(args)

    model = models.get_model(args.arch, weights="DEFAULT").to(device)
    replace_head(model)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"\n===== {tag}  device={device}  params={n_par:,}  train={len(train_loader.dataset)} val={n_val} =====")
    log(f"FashionMNIST -> {args.size}px RGB, ImageNet weights, epochs={args.epochs} "
        f"(freeze {args.freeze_epochs}), bs={args.batch_size}, lr={args.lr}, wd={args.weight_decay}")

    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    rows, best = [], {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0, "state": None}
    t0 = time.time()

    set_backbone_frozen(model, frozen=args.freeze_epochs > 0)
    head_ids = {id(p) for p in head_parameters(model)}

    def make_optimizer(lr, head_mult):
        back = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
        groups = [g for g in ({"params": back, "lr": lr}, {"params": head, "lr": lr * head_mult}) if g["params"]]
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    opt = make_optimizer(args.head_lr, 1.0)
    sched = None

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_frozen(model, frozen=False)
            log(f"  [epoch {epoch}] unfreezing backbone")
            opt = make_optimizer(args.lr, 3.0)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.epochs - args.freeze_epochs))

        model.train()
        ls = tot = cor = 0
        te = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = crit(out, y)
            loss.backward()
            opt.step()
            ls += loss.float().item() * y.size(0)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        train_loss, train_acc = ls / tot, cor / tot
        val_loss, val_acc = evaluate(model, val_loader, crit, device)
        secs = time.time() - te
        log(f"  ep{epoch:>2d} train={train_acc:.4f}/{train_loss:.4f}  val={val_acc:.4f}/{val_loss:.4f}  {secs:.0f}s")

        rows.append(dict(arch=args.arch, size=args.size, epoch=epoch, secs=round(secs, 1),
                         train_loss=f"{train_loss:.6f}", train_acc=f"{train_acc:.6f}",
                         val_loss=f"{val_loss:.6f}", val_acc=f"{val_acc:.6f}"))
        improved = val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"])
        if improved:
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if sched is not None:
            sched.step()

    model.load_state_dict(best["state"])
    test_loss, test_acc = evaluate(model, test_loader, crit, device)
    total_secs = time.time() - t0
    log(f"BEST ep={best['epoch']} val={best['val_acc']:.4f}  TEST={test_acc:.4f}  ({total_secs:.0f}s total)")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ckpt = {"state_dict": best["state"], "arch": args.arch, "val_acc": best["val_acc"],
            "test_acc": test_acc, "epoch": best["epoch"],
            "config": {k: v for k, v in vars(args).items()}}
    torch.save(ckpt, out_dir / f"{tag}.pt")
    (out_dir / f"{tag}.json").write_text(json.dumps(
        {"arch": args.arch, "size": args.size, "epochs": args.epochs,
         "params": n_par, "best_epoch": best["epoch"], "val_acc": best["val_acc"],
         "test_acc": test_acc, "test_loss": test_loss, "secs": round(total_secs, 1)},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
