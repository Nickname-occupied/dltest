"""报告用图脚本（2026-10-25 版）

输出目录: test1/figs/

  1. 最终模型
       final_loss.png   训练/验证损失
       final_acc.png    训练/验证准确率
  2. 汇总表
       table_val_summary.csv / .md   全部模型的验证集准确率（仅基线附测试准确率）
  3. 每组实验各两张（损失用验证损失，准确率用验证准确率，一模型一条曲线）
       exp1_structure_{loss,acc}.png
       exp2_activation_{loss,acc}.png
       exp3_criterion_{loss,acc}.png
       exp4_hyper_{loss,acc}.png
       exp5_dropout_{loss,acc}.png
  4. 第四组额外四张
       exp4_bs_{loss,acc}.png   固定 lr=1e-4，变 batch size
       exp4_lr_{loss,acc}.png   固定 bs=64，变 lr

图内不含任何测试集指标（只有汇总表里基线那一行有测试准确率）。

运行: python test1/make_figs.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOG = HERE / 'logs'
OUT = HERE / 'figs'
OUT.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 160
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

CYCLE = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2',
         '#72B7B2', '#EECA3B', '#9D755D', '#BAB0AC']

BASELINE = '[2048, 2048]'
BASELINE_LR = '0.0001'
BASELINE_BS = 64

# ------------------------------------------------------------------ load
rows = list(csv.DictReader(open(LOG / 'results.csv', encoding='utf-8')))
curves = {}
for r in rows:
    curves.setdefault(r['run_id'], []).append(dict(
        epoch=int(r['epoch']), train_loss=float(r['train_loss']),
        train_acc=float(r['train_acc']), val_loss=float(r['val_loss']),
        val_acc=float(r['val_acc'])))
for k in curves:
    curves[k].sort(key=lambda d: d['epoch'])


def xy(rid, key):
    return [d['epoch'] for d in curves[rid]], [d[key] for d in curves[rid]]


def best_val(rid):
    v = [d['val_acc'] for d in curves[rid]]
    i = v.index(max(v))
    return max(v), curves[rid][i]['epoch']


def params_of(rid):
    """参数总量，从已有 checkpoint 里读到的配置推算。"""
    if rid == '[2048]':
        dims = [2048]
    elif '_lr_' in rid:
        dims = [2048, 2048]
    elif '_dropout_' in rid or '_ReLU' in rid or '_Tanh' in rid or '_Sigmoid' in rid \
            or '_GELU' in rid or 'Loss' in rid:
        dims = [2048, 2048]
    else:
        dims = [int(t) for t in rid.strip('[]').split(',')]
    d = [784] + dims + [10]
    return sum(d[i] * d[i + 1] + d[i + 1] for i in range(len(d) - 1))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print('  ', name)


def plot_lines(ax, items, key, ylabel, title, logy=False):
    """items: [(run_id, label)] -> 每个模型一条曲线；纵轴按实际范围自动收紧。"""
    allvals = []
    for k, (rid, lab) in enumerate(items):
        ep, y = xy(rid, key)
        allvals += y
        ax.plot(ep, y, label=lab, linewidth=1.6,
                color=CYCLE[k % len(CYCLE)], linestyle='-' if len(items) <= 6 or k % 3 == 0
                else ('--' if k % 3 == 1 else ':'))
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    else:
        lo, hi = min(allvals), max(allvals)
        pad = (hi - lo) * 0.08 or 0.002
        ax.set_ylim(lo - pad, hi + pad)
    ax.legend(fontsize=7.5, loc='best', ncol=2 if len(items) > 6 else 1)


print('writing figures to', OUT)

# ------------------------------------------------------------------ 1. final model
ep, tr = xy(BASELINE, 'train_loss')
_, va = xy(BASELINE, 'val_loss')
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(ep, tr, label='训练集损失', color='#4C78A8', linewidth=1.8)
ax.plot(ep, va, label='验证集损失', color='#F58518', linewidth=1.8)
ax.set_xlabel('epoch')
ax.set_ylabel('损失')
ax.set_title('最终模型：损失曲线（[2048,2048] + GELU + dropout 0.1）')
ax.legend()
save(fig, 'final_loss.png')

ep, tr = xy(BASELINE, 'train_acc')
_, va = xy(BASELINE, 'val_acc')
bv, bep = best_val(BASELINE)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(ep, tr, label='训练集准确率', color='#4C78A8', linewidth=1.8)
ax.plot(ep, va, label='验证集准确率', color='#F58518', linewidth=1.8)
ax.scatter([bep], [bv], color='#E45756', zorder=5, s=28)
ax.annotate(f'最优验证 {bv:.4f} (ep{bep})', (bep, bv), textcoords='offset points',
            xytext=(-8, -16), fontsize=8, color='#E45756', ha='right')
ax.set_xlabel('epoch')
ax.set_ylabel('准确率')
ax.set_title('最终模型：准确率曲线（[2048,2048] + GELU + dropout 0.1）')
ax.legend(loc='lower right')
save(fig, 'final_acc.png')

# ------------------------------------------------------------------ 2. summary table
GROUPS = [
    ('1 网络结构', ['[256, 256]', '[1024, 1024]', '[2048]', '[2048, 2048]',
                    '[2048, 2048, 2048]', '[4096, 4096]']),
    ('2 激活函数', ['[2048, 2048]_ReLU', '[2048, 2048]_Tanh',
                    '[2048, 2048]_Sigmoid', '[2048, 2048]_GELU']),
    ('3 损失函数', ['[2048, 2048]_CrossEntropyLoss', '[2048, 2048]_MSELoss']),
    ('4 超参数', [f'[2048, 2048]_lr_{lr}_batch_{bs}'
                  for lr in ('0.01', '0.0001', '1e-06') for bs in (16, 64, 256)]),
    ('5 Dropout', [f'[2048, 2048]_dropout_{d}' for d in ('0', '0.1', '0.3', '0.5')]),
]
TEST = {BASELINE: 0.9914}          # 只有基线报告测试准确率

table = []
for gname, ids in GROUPS:
    for rid in ids:
        if rid not in curves:
            continue
        v, e = best_val(rid)
        table.append(dict(group=gname, run=rid, params=params_of(rid), best_ep=e,
                          val_acc=v, test_acc=TEST.get(rid)))

with (OUT / 'table_val_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['group', 'run', 'params', 'best_ep', 'val_acc', 'test_acc'])
    w.writeheader()
    for r in table:
        w.writerow(dict(group=r['group'], run=r['run'], params=r['params'], best_ep=r['best_ep'],
                        val_acc=f"{r['val_acc'] * 100:.2f}",
                        test_acc='' if r['test_acc'] is None else f"{r['test_acc'] * 100:.2f}"))

lines = ['| 实验组 | 模型配置 | 参数量 | 最优轮 | 验证准确率(%) | 测试准确率(%) |',
         '|---|---|---|---|---|---|']
for r in table:
    t = '' if r['test_acc'] is None else f"{r['test_acc'] * 100:.2f}"
    lines.append(f"| {r['group']} | `{r['run']}` | {r['params']:,} | {r['best_ep']} | "
                 f"{r['val_acc'] * 100:.2f} | {t} |")
(OUT / 'table_val_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
print('   table_val_summary.csv / .md')

# ------------------------------------------------------------------ 3. per-group figures
GROUP_FILES = {
    '1 网络结构': ('exp1_structure', ['[256, 256]', '[1024, 1024]', '[2048]',
                                      '[2048, 2048]', '[2048, 2048, 2048]', '[4096, 4096]']),
    '2 激活函数': ('exp2_activation', ['[2048, 2048]_ReLU', '[2048, 2048]_Tanh',
                                       '[2048, 2048]_Sigmoid', '[2048, 2048]_GELU']),
    '3 损失函数': ('exp3_criterion', ['[2048, 2048]_CrossEntropyLoss', '[2048, 2048]_MSELoss']),
    '4 超参数': ('exp4_hyper', [f'[2048, 2048]_lr_{lr}_batch_{bs}'
                                for lr in ('0.01', '0.0001', '1e-06') for bs in (16, 64, 256)]),
    '5 Dropout': ('exp5_dropout', [f'[2048, 2048]_dropout_{d}' for d in ('0', '0.1', '0.3', '0.5')]),
}
for gname, (fname, ids) in GROUP_FILES.items():
    ids = [r for r in ids if r in curves]
    labels = [r if not r.startswith('[2048, 2048]_') else r.split(']_')[-1] for r in ids]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    logy = fname == 'exp3_criterion'
    title = f'{gname}：验证损失曲线' + ('（对数轴，两种损失尺度不可直接比较）' if logy else '')
    plot_lines(ax, list(zip(ids, labels)), 'val_loss', '验证损失', title, logy=logy)
    save(fig, f'{fname}_loss.png')
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    plot_lines(ax, list(zip(ids, labels)), 'val_acc', '验证准确率', f'{gname}：验证准确率曲线')
    save(fig, f'{fname}_acc.png')

# ------------------------------------------------------------------ 4. batch size @ baseline lr
bs_ids = [f'[2048, 2048]_lr_{BASELINE_LR}_batch_{b}' for b in (16, 64, 256)]
bs_ids = [r for r in bs_ids if r in curves]
labels = [f'batch={b}' for b in (16, 64, 256)][:len(bs_ids)]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
plot_lines(ax, list(zip(bs_ids, labels)), 'val_loss', '验证损失',
           f'批量大小：验证损失曲线（lr={float(BASELINE_LR):g}）')
save(fig, 'exp4_bs_loss.png')
fig, ax = plt.subplots(figsize=(7.2, 4.2))
plot_lines(ax, list(zip(bs_ids, labels)), 'val_acc', '验证准确率',
           f'批量大小：验证准确率曲线（lr={float(BASELINE_LR):g}）')
save(fig, 'exp4_bs_acc.png')

# ------------------------------------------------------------------ 4. lr @ baseline bs
lr_ids = [f'[2048, 2048]_lr_{v}_batch_{BASELINE_BS}' for v in ('0.01', '0.0001', '1e-06')]
lr_ids = [r for r in lr_ids if r in curves]
labels = ['lr=1e-2', 'lr=1e-4', 'lr=1e-6'][:len(lr_ids)]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
plot_lines(ax, list(zip(lr_ids, labels)), 'val_loss', '验证损失',
           f'学习率：验证损失曲线（batch={BASELINE_BS}）')
save(fig, 'exp4_lr_loss.png')
fig, ax = plt.subplots(figsize=(7.2, 4.2))
plot_lines(ax, list(zip(lr_ids, labels)), 'val_acc', '验证准确率',
           f'学习率：验证准确率曲线（batch={BASELINE_BS}）')
save(fig, 'exp4_lr_acc.png')

# ------------------------------------------------------------------ 3b. delta vs baseline
# 差值曲线：纵轴单位是百分点，基线恒为 0。
# 验证准确率逐轮抖动约 ±0.3pp，直接画差值全是噪声，所以做 5 轮滑动平均。
def smooth(v, k=5):
    out = []
    for i in range(len(v)):
        lo, hi = max(0, i - k // 2), min(len(v), i + k // 2 + 1)
        out.append(sum(v[lo:hi]) / (hi - lo))
    return out


base_acc = {d['epoch']: d['val_acc'] for d in curves[BASELINE]}


def delta_fig(fname, title, ids, labels, xmin=10, ylim=None, note=None):
    ids = [r for r in ids if r in curves]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    lo, hi = 0.0, 0.0
    for k, (rid, lab) in enumerate(zip(ids, labels)):
        ep, y = xy(rid, 'val_acc')
        d = smooth([(v - base_acc[e]) * 100 for e, v in zip(ep, y)])
        pts = [x for e, x in zip(ep, d) if e >= xmin]
        lo, hi = min(lo, min(pts)), max(hi, max(pts))
        ax.plot(ep, d, label=lab, linewidth=1.7, color=CYCLE[k % len(CYCLE)])
    ax.axhline(0, color='black', linewidth=0.9, linestyle='--')
    ax.set_xlabel('epoch')
    ax.set_ylabel('验证准确率差值（百分点）')
    ax.set_title(title)
    ax.set_xlim(xmin, 50)
    if ylim:
        ax.set_ylim(*ylim)
        if note:
            ax.text(0.015, 0.965, note, transform=ax.transAxes, fontsize=7.5,
                    color='#666666', va='top')
    else:
        if hi - lo < 0.3:                 # 保证最小跨度，避免把噪声放大成趋势
            c = (hi + lo) / 2
            lo, hi = c - 0.15, c + 0.15
        pad = (hi - lo) * 0.10
        ax.set_ylim(lo - pad, hi + pad)
    ax.legend(fontsize=7.5, loc='best')
    save(fig, f'{fname}_delta.png')


for gname, (fname, ids) in GROUP_FILES.items():
    if fname not in ('exp1_structure', 'exp2_activation', 'exp5_dropout'):
        continue
    ids = [r for r in ids if r in curves]
    labels = [r if not r.startswith('[2048, 2048]_') else r.split(']_')[-1] for r in ids]
    delta_fig(fname, f'{gname}：相对基线的验证准确率差值（5 轮滑动平均）', ids, labels)

# 超参组：整体（纵轴会自动放宽到 -25pp 量级，因为 lr 选错的会崩）
ids_h = [f'[2048, 2048]_lr_{lr}_batch_{bs}'
         for lr in ('0.01', '0.0001', '1e-06') for bs in (16, 64, 256)]
delta_fig('exp4_hyper', '4 超参数：相对基线的验证准确率差值（5 轮滑动平均）',
          ids_h, [r.split(']_')[-1] for r in ids_h],
          ylim=(-30, 4), note='lr=1e-2,batch=16 低于下限（约 -87pp），已裁切')

# 超参组的"好"切片：固定 lr=1e-4 时的三个 batch size，差异只有 0.16pp
ids_bs = [f'[2048, 2048]_lr_0.0001_batch_{b}' for b in (16, 64, 256)]
delta_fig('exp4_bs', '批量大小：相对基线的验证准确率差值（lr=1e-4）',
          ids_bs, [f'batch={b}' for b in (16, 64, 256)])

# 超参组的 lr 切片：固定 batch=64，三种失败模式
ids_lr = [f'[2048, 2048]_lr_{v}_batch_64' for v in ('0.01', '0.0001', '1e-06')]
delta_fig('exp4_lr', '学习率：相对基线的验证准确率差值（batch=64）',
          ids_lr, ['lr=1e-2', 'lr=1e-4', 'lr=1e-6'])

# ------------------------------------------------------------------ 3c. best-val bar per group
def best_bars(fname, gname, ids, labels):
    def is_base(rid):
        return (rid == BASELINE
                or rid.endswith('_GELU')
                or rid.endswith('_CrossEntropyLoss')
                or rid.endswith('_lr_0.0001_batch_64')
                or rid.endswith('_dropout_0.1'))
    vals = [best_val(r)[0] * 100 for r in ids]
    cols = ['#E45756' if is_base(r) else '#4C78A8' for r in ids]
    if len(ids) > 6:                      # 横条，标签好放
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        y = list(range(len(ids)))
        ax.barh(y, vals, color=cols)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('最优验证准确率（%）')
        ax.set_xlim(min(vals) - 0.6, max(vals) + 1.6)
        for i, v in enumerate(vals):
            ax.text(v + 0.12, i, f'{v:.2f}', va='center', fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        x = list(range(len(ids)))
        ax.bar(x, vals, color=cols, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8.5)
        ax.set_ylabel('最优验证准确率（%）')
        ax.set_ylim(min(vals) - 0.5, max(vals) + 0.25)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_title(f'{gname}：最优验证准确率对比（红色为基线）')
    save(fig, f'{fname}_best.png')


for gname, (fname, ids) in GROUP_FILES.items():
    ids = [r for r in ids if r in curves]
    labels = [r if not r.startswith('[2048, 2048]_') else r.split(']_')[-1] for r in ids]
    best_bars(fname, gname, ids, labels)

print('done')


