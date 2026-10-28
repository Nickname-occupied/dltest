"""Dropout sweep at [2048,2048] + GELU, replicating test1.py's protocol exactly.

Protocol: v2.RandomAffine(10deg, translate=0.05, scale=0.9-1.1) on GPU (train only),
Adam lr=1e-4 constant, batch 64, 50 epochs, best-on-val checkpoint, seed 42.

Model selection is by val only. The test set is evaluated exactly once, at the
end, for the val-winning configuration.
"""
import csv
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'test1' / 'logs' / 'weights_final'
OUT.mkdir(parents=True, exist_ok=True)
LOG = HERE / 'test1' / 'logs' / 'final_sweep.log'
CURVE = HERE / 'test1' / 'logs' / 'final_sweep.csv'

HIDDEN = [2048, 2048]
DROPOUTS = [0.0, 0.1, 0.3, 0.5]
EPOCHS = 50
SEED = 42
LR = 1e-4
BS = 64


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
log(f'\n===== dropout sweep  hidden={HIDDEN} act=GELU  epochs={EPOCHS} lr={LR} bs={BS} =====')

curve_rows = []
results = {}

for dr in DROPOUTS:
    rid = f'[2048, 2048]_GELU_dropout_{dr}'
    t0 = time.time()
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_sub, batch_size=BS, shuffle=True, generator=gen)
    valid_loader = DataLoader(val_sub, batch_size=BS, shuffle=False)

    model = mynet(HIDDEN, nn.GELU(), dr).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_ep, best_state = -1.0, 0, None
    for e in range(1, EPOCHS + 1):
        model.train()
        loss_sum = total = correct = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            x = aug(x)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
        train_loss, train_acc = loss_sum / total, correct / total

        model.eval()
        loss_sum = total = correct = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                loss_sum += loss_fn(out, y).item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_loss, val_acc = loss_sum / total, correct / total

        curve_rows.append(dict(run_id=rid, dropout=dr, epoch=e, train_loss=f'{train_loss:.6f}',
                               train_acc=f'{train_acc:.6f}', val_loss=f'{val_loss:.6f}',
                               val_acc=f'{val_acc:.6f}'))
        if val_acc > best_val:
            best_val, best_ep = val_acc, e
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if e % 10 == 0 or e == 1:
            log(f'  dr={dr} ep{e:>2d} train={train_acc:.4f} val={val_acc:.4f} best={best_val:.4f}')

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = torch.cat([model(x.cuda()).argmax(1).cpu() for x, _ in valid_loader])
    ckpt = {'state_dict': best_state, 'run_id': rid,
            'config': {'hidden_dims': HIDDEN, 'activation': 'GELU', 'dropout': dr,
                       'lr': LR, 'batch_size': BS, 'loss': 'CrossEntropyLoss',
                       'epochs': EPOCHS, 'seed': SEED,
                       'augment': 'v2.RandomAffine(10deg,translate=0.05,scale=0.9-1.1),batch-level,GPU'},
            'best_epoch': best_ep, 'best_val_acc': best_val}
    torch.save(ckpt, OUT / f'dropout_{dr}.pt')
    results[dr] = dict(state=best_state, val_pred=val_pred, val=best_val, ep=best_ep,
                       secs=round(time.time() - t0, 1), cfg=ckpt['config'])
    log(f'BEST dr={dr}: ep={best_ep} val={best_val:.4f}  ({results[dr]["secs"]}s)')

with CURVE.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
    w.writeheader()
    w.writerows(curve_rows)

labels = torch.cat([y for _, y in DataLoader(val_sub, batch_size=1000)])


def mcnemar(pa, pb):
    ra, rb = pa == labels, pb == labels
    win, lose = (ra & ~rb).sum().item(), (~ra & rb).sum().item()
    n = win + lose
    if n == 0:
        return win, lose, 1.0
    k = min(win, lose)
    return win, lose, min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)


log('\n=== val results (model selection) ===')
for dr in DROPOUTS:
    r = results[dr]
    log(f'  dropout {dr:<4} val={r["val"]:.4f} @ep{r["ep"]:>2d}')

base = 0.5
log(f'\n=== exact McNemar on val vs dropout {base} ===')
for dr in DROPOUTS:
    if dr == base:
        continue
    w_, l_, p = mcnemar(results[dr]['val_pred'], results[base]['val_pred'])
    log(f'  {dr:<4} vs {base}: delta={results[dr]["val"] - results[base]["val"]:+.4f}  '
        f'win={w_:3d} lose={l_:3d}  p={p:.4f}  {"SIGNIFICANT" if p < 0.05 else "not significant"}')

# single test-set evaluation for the val winner
winner = max(DROPOUTS, key=lambda d: results[d]['val'])
log(f'\nval winner: dropout={winner}. Evaluating the test set once.')
m = mynet(HIDDEN, nn.GELU(), winner).cuda()
m.load_state_dict(results[winner]['state'])
m.eval()
test_loader = DataLoader(test_ds, batch_size=1000)
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        correct += (m(x).argmax(1) == y).sum().item()
        total += y.size(0)
test_acc = correct / total
log(f'TEST dropout={winner}: {correct}/{total} = {test_acc:.4f}')
log('done')
