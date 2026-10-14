"""Summarize the Prompt-2 ablation (p2_ablation.py runs) once runs are DONE.

Per (K, b) and variant: seeds done, best epoch, validation rate, frozen-test mean / p5 at 20 dB (mean +- std over
seeds), 30 dB mean, train hours, failed runs. Paired comparisons on the SAME test channels and symbols: for each
pair of variants and each seed index, the per-channel difference; reported as the mean over seeds of the mean
difference, with a 95% interval from a bootstrap over channels of the seed-averaged per-channel difference.
Effects (seed-averaged test mean at 20 dB):
  training strategy on the original representation      B - A*   (A* = A_orig or A_orig_lr1e-3, chosen by validation)
  F alone (no R)                                          C - B
  R alone (raw channel)                                   D - B
  F given R                                               F - D
  R given F                                               F - C
  interaction F x R                                       (F - D) - (C - B)
  logit L2 given F + R                                    F - E
  total vs. original                                      F - A*
Writes ablation_p2/summary.json, ablation_p2/summary.md and the training-curve figure ablation_p2/curves.pdf.

  python p2_summarize.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'ablation_p2')
VARIANTS = ['A_orig', 'A_orig_lr1e-3', 'B_orig_ts', 'C_F', 'D_R', 'E_FR_nol2', 'F_full', 'F_ep15', 'F_paper']
LABEL = {'A_orig': 'A Original (lr 5e-3)', 'A_orig_lr1e-3': 'A Original (lr 1e-3)', 'B_orig_ts': 'B Original + TS',
         'C_F': 'C F + TS', 'D_R': 'D R + TS', 'E_FR_nol2': 'E F+R, no L2', 'F_full': 'F Full (retrained)',
         'F_ep15': 'F paper ckpt, 15 ep', 'F_paper': 'F paper ckpt, 15+15 ep'}
PAPER_F = ('F_ep15', 'F_paper')     # existing paper checkpoints scored by p2_eval_paper_F.py (not retrained)
I20 = 10   # index of 20 dB in -30:30:5
I30 = 12
N_BOOT, SEED = 5000, 20261014


def runs(K, b, v):
    out = {}
    if v in PAPER_F:
        d = os.path.join(ROOT, 'paper_F', f'K{K}b{b}', v)
        if os.path.exists(os.path.join(d, 'summary.json')):
            out[0] = {'dir': d, 'done': True, 'failed': False, 'history': [], 'params': None,
                      'summary': dict(json.load(open(os.path.join(d, 'summary.json'))), train_hours=float('nan')),
                      'test': np.load(os.path.join(d, 'test_rates.npz'))}
        return out
    for d in sorted(glob.glob(os.path.join(ROOT, 'runs', f'K{K}b{b}', f'{v}_s*'))):
        if os.path.exists(os.path.join(d, 'STOPPED')):
            continue
        seed = int(d.rsplit('_s', 1)[1])
        h = json.load(open(os.path.join(d, 'history.json'))) if os.path.exists(os.path.join(d, 'history.json')) else None
        rec = {'dir': d, 'done': os.path.exists(os.path.join(d, 'DONE')), 'failed': os.path.exists(os.path.join(d, 'FAILED')),
               'history': h['history'] if h else [], 'params': h.get('params') if h else None}
        if rec['done']:
            rec['summary'] = json.load(open(os.path.join(d, 'summary.json')))
            rec['test'] = np.load(os.path.join(d, 'test_rates.npz'))
        out[seed] = rec
    return out


def boot_ci(x, n=N_BOOT, seed=SEED):
    g = np.random.default_rng(seed)
    m = x[g.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def seed_avg(rs, key=I20):
    """Per-channel test rate at `key` averaged over the DONE seeds, and the seed list."""
    ds = sorted(s for s, r in rs.items() if r['done'])
    if not ds:
        return None, []
    return np.mean([rs[s]['test']['sum_rate'][key] for s in ds], 0), ds


def main():
    summ = {'note': 'test = Htest[6144:8192] (frozen, scored once per run); validation = Htest[4096:6144]', 'configs': {}}
    md = ['# Prompt 2 ablation summary', '',
          'Frozen test set Htest[6144:8192], 2048 channels, 20 dB, LS estimator, argmax decisions. '
          'Mean +- std over seeds; checkpoint chosen on validation only.', '']
    for K, b in [(2, 1), (2, 3), (6, 1), (6, 3)]:
        cfg = {}
        rs = {v: runs(K, b, v) for v in VARIANTS}
        if not any(rs.values()):
            continue
        md += [f'## K={K}, b={b}', '',
               '| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |',
               '|---|---|---|---|---|---|---|---|']
        for v in VARIANTS:
            done = [r for r in rs[v].values() if r['done']]
            if not rs[v]:
                continue
            row = {'seeds_done': [s for s, r in rs[v].items() if r['done']],
                   'seeds_running': [s for s, r in rs[v].items() if not r['done'] and not r['failed']],
                   'seeds_failed': [s for s, r in rs[v].items() if r['failed']]}
            if done:
                f = lambda k: np.array([r['summary'][k] for r in done])  # noqa: E731
                row.update({k: [float(f(k).mean()), float(f(k).std(ddof=1)) if len(done) > 1 else 0.0]
                            for k in ('val_rate_ls', 'test_mean_20db', 'test_p5_20db', 'test_mean_30db', 'train_hours')})
                row['best_epoch'] = [int(r['summary']['best_epoch']) for r in done]
                fmt = lambda k: f"{row[k][0]:.3f} ± {row[k][1]:.3f}"  # noqa: E731
                md.append(f"| {LABEL[v]} | {len(done)} | {fmt('val_rate_ls')} | {fmt('test_mean_20db')} | "
                          f"{fmt('test_p5_20db')} | {fmt('test_mean_30db')} | {row['best_epoch']} | "
                          f"{row['train_hours'][0]:.1f} |")
            else:
                h = [r['history'][-1]['val_rate_ls'] for r in rs[v].values() if r['history']]
                md.append(f"| {LABEL[v]} | 0 (running: val {', '.join(f'{x:.3f}' for x in h)}) | | | | | | |")
            cfg[v] = row
        # choose A* by validation (seed 0 of both lr), never by test
        a0 = {v: rs[v].get(0) for v in ('A_orig', 'A_orig_lr1e-3')}
        astar = None
        if all(r and (r['done'] or r['failed']) for r in a0.values()) and any(r['done'] for r in a0.values()):
            astar = max(a0, key=lambda v: a0[v]['summary']['val_rate_ls'] if a0[v]['done'] else float('-inf'))
        cfg['A_star'] = astar
        # paired effects on the seed-averaged per-channel test rates
        avg = {v: seed_avg(rs[v]) for v in VARIANTS}
        if astar:
            avg['A*'] = avg[astar]
        effects = {'paper F (15 ep) - A* (15 ep), same budget': ('F_ep15', 'A*'),
                   'paper F (15+15 ep) - A* (15 ep)': ('F_paper', 'A*'),
                   'TS on original (B - A*)': ('B_orig_ts', 'A*'), 'F alone (C - B)': ('C_F', 'B_orig_ts'),
                   'R alone (D - B)': ('D_R', 'B_orig_ts'), 'F given R (F - D)': ('F_full', 'D_R'),
                   'R given F (F - C)': ('F_full', 'C_F'), 'logit L2 given F+R (F - E)': ('F_full', 'E_FR_nol2'),
                   'total (F - A*)': ('F_full', 'A*')}
        eff = {}
        for name, (x, y) in effects.items():
            if avg.get(x, (None,))[0] is None or avg.get(y, (None,))[0] is None:
                continue
            d = avg[x][0] - avg[y][0]
            eff[name] = {'mean_bit': float(d.mean()), 'ci95': boot_ci(d), 'rel_pct': float(100 * d.mean() / avg[y][0].mean()),
                         'frac_channels_better': float((d > 0).mean()), 'seeds': [avg[x][1], avg[y][1]]}
        if all(avg[v][0] is not None for v in ('B_orig_ts', 'C_F', 'D_R', 'F_full')):
            inter = (avg['F_full'][0] - avg['D_R'][0]) - (avg['C_F'][0] - avg['B_orig_ts'][0])
            eff['interaction F x R'] = {'mean_bit': float(inter.mean()), 'ci95': boot_ci(inter)}
        cfg['effects'] = eff
        if eff:
            md += ['', '| effect (test, 20 dB) | Δ mean [bit/cu] | 95% CI (channels) | relative | share of channels better |',
                   '|---|---|---|---|---|']
            for name, e in eff.items():
                md.append(f"| {name} | {e['mean_bit']:+.3f} | [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}] | "
                          f"{e.get('rel_pct', float('nan')):+.1f}% | {e.get('frac_channels_better', float('nan')):.2f} |")
        md.append('')
        summ['configs'][f'K{K}b{b}'] = cfg
    json.dump(summ, open(os.path.join(ROOT, 'summary.json'), 'w'), indent=1)
    open(os.path.join(ROOT, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    try:
        curves()
    except Exception as e:  # figure is optional
        print('curves skipped:', e, file=sys.stderr)


def curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfgs = [(2, 1), (2, 3), (6, 1), (6, 3)]
    fig, ax = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    cols = dict(zip(VARIANTS, plt.cm.tab10.colors))
    for j, (K, b) in enumerate(cfgs):
        for v in VARIANTS:
            for s, r in runs(K, b, v).items():
                h = r['history']
                if not h:
                    continue
                ep = [x['epoch'] for x in h]
                kw = dict(color=cols[v], lw=1, alpha=0.9 if s == 0 else 0.5, label=LABEL[v] if s == 0 else None)
                ax[0, j].plot(ep, [x['val_rate_ls'] for x in h], **kw)
                ax[1, j].semilogy(ep, [x['val_logit_gap']['gap_quantiles'][3] for x in h], **kw)
                ax[2, j].plot(ep, [x['train_share_saturated'] for x in h], **kw)
        ax[0, j].set_title(f'K={K}, b={b}')
    ax[0, 0].set_ylabel('val sum rate, 20 dB (LS)')
    ax[1, 0].set_ylabel('median logit gap (val)')
    ax[2, 0].set_ylabel('share saturated (train)')
    for a in ax[-1]:
        a.set_xlabel('epoch')
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'curves.pdf'))
    fig.savefig(os.path.join(ROOT, 'curves.png'), dpi=110)


if __name__ == '__main__':
    main()
