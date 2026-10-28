"""Evaluate every saved checkpoint on the MNIST test set.

Scans test1/logs/weights (the 24 comparison runs) and test1/logs/weights_final
(the 4 final-config candidates). Writes test1/logs/test_eval_all.csv.

NOTE: this touches the test set once per checkpoint. Use it to describe the
results, not to pick a configuration - selection stays on validation.
"""
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
DIRS = [HERE / 'test1' / 'logs' / 'weights', HERE / 'test1' / 'logs' / 'weights_final']
OUT = HERE / 'test1' / 'logs' / 'test_eval_all.csv'


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


ACTS = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "Sigmoid": nn.Sigmoid(), "GELU": nn.GELU()}
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_ds = datasets.MNIST(DATA, train=False, download=True, transform=tf)
tl = DataLoader(test_ds, batch_size=1000, shuffle=False)


def group_of(ck, src):
    rid = ck['run_id']
    cfg = ck['config']
    if src == 'weights_final':
        return 'final config'
    if '_lr_' in rid:
        return 'hyperparam'
    if '_dropout_' in rid:
        return 'dropout'
    if cfg['loss'] == 'MSELoss' or '_CrossEntropyLoss' in rid:
        return 'loss'
    if cfg['activation'] != 'ReLU':
        return 'activation'
    return 'structure'


rows = []
for d in DIRS:
    for p in sorted(d.glob('*.pt')):
        if p.name == 'final_model.pt':
            continue
        try:
            ck = torch.load(p, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f'  !! {p.name}: {e}')
            continue
        cfg = ck['config']
        m = mynet(cfg['hidden_dims'], ACTS[cfg['activation']], cfg['dropout']).cuda()
        m.load_state_dict(ck['state_dict'])
        m.eval()
        loss_fn = nn.CrossEntropyLoss() if cfg['loss'] == 'CrossEntropyLoss' else nn.MSELoss()
        correct = total = 0
        lsum = 0.0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.cuda(), y.cuda()
                out = m(x)
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    l = loss_fn(out, y)
                else:
                    l = loss_fn(nn.functional.one_hot(y, 10).float(),
                                nn.functional.softmax(out, dim=1))
                lsum += l.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        rid = ck['run_id']
        if d.name == 'weights_final':
            rid = f"[2048,2048]+GELU+dr{cfg['dropout']}"
        rows.append(dict(group=group_of(ck, d.name), run_id=rid,
                         params=sum(v.numel() for v in m.parameters()),
                         dropout=cfg['dropout'], activation=cfg['activation'],
                         best_ep=ck['best_epoch'], val_acc=ck['best_val_acc'],
                         test_acc=correct / total, test_loss=lsum / total,
                         delta=correct / total - ck['best_val_acc'], src=d.name))

order = {'structure': 0, 'activation': 1, 'loss': 2, 'hyperparam': 3, 'dropout': 4, 'final config': 5}
rows.sort(key=lambda r: (order.get(r['group'], 9), -r['test_acc']))

hdr = f"{'group':13s} {'run_id':30s} {'params':>10s} {'ep':>3s} {'val':>7s} {'test':>7s} {'d':>7s}"
print(hdr)
print('-' * len(hdr))
for r in rows:
    print(f"{r['group']:13s} {r['run_id']:30s} {r['params']:>10,} {r['best_ep']:>3d} "
          f"{r['val_acc']:>7.4f} {r['test_acc']:>7.4f} {r['delta']:>+7.4f}")

with OUT.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()})
print(f'\n{len(rows)} checkpoints -> {OUT}')

