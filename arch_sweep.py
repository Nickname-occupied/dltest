"""Architecture sweep with the tuned hyper-parameters held fixed.

Fixed: GELU, Adam lr=1e-4 (constant), batch 64, dropout 0.1, 50 epochs,
v2.RandomAffine(10deg, translate=0.05, scale=0.9-1.1) on GPU (train only), seed 42.
Varied: hidden_dims only.

Reports best-val and the test accuracy of the best-val checkpoint for each
architecture. Selection between architectures is by validation only.
"""
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'test1' / 'logs' / 'weights_arch'
OUT.mkdir(parents=True, exist_ok=True)
LOG = HERE / 'test1' / 'logs' / 'arch_sweep.log'
CURVE = HERE / 'test1' / 'logs' / 'arch_sweep.csv'

ARCHS = [[2048, 2048], [2048, 2048, 2048], [1024, 1024], [4096, 4096]]
EPOCHS, SEED, LR, BS, DR = 50, 42, 1e-4, 64, 0.1


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(msg + '\n')


class mynet(nn.Module):
    def __init__(self, hidden_dims, activation, dropout_rate):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.flatten = nn.Flatten()
        self.activation = activation
        layers, inp = [], 28 * 28
        for i in hidden_dims:
            layers.append(nn.Linear(inp, i))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            inp = i
        layers.append(nn.Linear(inp, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.flatten(x))


tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(DATA, train=True, download=True, transform=tf)
test_ds = datasets.MNIST(DATA, train=False, download=True, transform=tf)
g_split = torch.Generator().manual_seed(SEED)
train_sub, val_sub = random_split(train_ds, [0.9, 0.1], generator=g_split)
aug = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))

test_loader = DataLoader(test_ds, batch_size=1000)
curve, summary = [], []

for hidden in ARCHS:
    rid = str(hidden)
    log(f'\n===== {rid}  GELU dr={DR} lr={LR} bs={BS} {EPOCHS}ep =====')
    t0 = time.time()
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_sub, batch_size=BS, shuffle=True, generator=gen)
    valid_loader = DataLoader(val_sub, batch_size=BS, shuffle=False)

    model = mynet(hidden, nn.GELU(), DR).cuda()
    n_par = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_ep, best_state = -1.0, 0, None
    for e in range(1, EPOCHS + 1):
        model.train()
        ls = tot = cor = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            x = aug(x)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            ls += loss.item() * y.size(0)
            tot += y.size(0)
            cor += (out.argmax(1) == y).sum().item()
        tr_loss, tr_acc = ls / tot, cor / tot

        model.eval()
        ls = tot = cor = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                ls += loss_fn(out, y).item() * y.size(0)
                cor += (out.argmax(1) == y).sum().item()
                tot += y.size(0)
        va_loss, va_acc = ls / tot, cor / tot
        curve.append(dict(arch=rid, epoch=e, train_loss=f'{tr_loss:.6f}', train_acc=f'{tr_acc:.6f}',
                          val_loss=f'{va_loss:.6f}', val_acc=f'{va_acc:.6f}'))
        if va_acc > best_val:
            best_val, best_ep = va_acc, e
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if e % 10 == 0 or e == 1:
            log(f'  ep{e:>2d} train={tr_acc:.4f} val={va_acc:.4f}')

    model.load_state_dict(best_state)
    model.eval()
    tc = tt = 0
    tls = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            tls += loss_fn(out, y).item() * y.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tt += y.size(0)
    test_acc, test_loss = tc / tt, tls / tt

    ckpt = {'state_dict': best_state, 'run_id': rid,
            'config': {'hidden_dims': hidden, 'activation': 'GELU', 'dropout': DR,
                       'lr': LR, 'batch_size': BS, 'loss': 'CrossEntropyLoss',
                       'epochs': EPOCHS, 'seed': SEED,
                       'augment': 'v2.RandomAffine(10deg,translate=0.05,scale=0.9-1.1),batch-level,GPU'},
            'best_epoch': best_ep, 'best_val_acc': best_val,
            'test_acc': test_acc, 'test_loss': test_loss}
    tag = '_'.join(map(str, hidden))
    torch.save(ckpt, OUT / f'{tag}.pt')
    secs = round(time.time() - t0, 1)
    summary.append(dict(arch=rid, params=n_par, best_ep=best_ep, val=best_val,
                        test=test_acc, delta=test_acc - best_val, secs=secs))
    log(f'BEST {rid}: ep={best_ep} val={best_val:.4f} TEST={test_acc:.4f} ({secs}s)')

with CURVE.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(curve[0]))
    w.writeheader()
    w.writerows(curve)

log('\n=== summary ===')
for r in summary:
    log(f"  {r['arch']:22s} params={r['params']:>11,} ep={r['best_ep']:>2d} "
        f"val={r['val']:.4f} test={r['test']:.4f} d={r['delta']:+.4f}")
log('done')
