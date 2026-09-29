"""GNN gain over the quantized linear baselines of Figs. 3 and 4 (K = 1: MRT + DAC; K >= 2: Rsum_sweep_vs_snr), in the style of Table II of
arXiv:2606.23141: per (K, b) and SNR, the relative gain in the MEAN sum rate (d_mu) and in the 5th PERCENTILE of the
per-channel sum rate (d_5, the rate exceeded by 95% of the channel realizations).

Baseline per metric: the better of ZF + DAC and WMMSE + DAC in that metric (the strongest quantized linear precoder;
'equal' DAC convention, as in Fig. 4). Per-channel sum rates: LS Bussgang estimator on the 125-symbol blocks after
per-block power normalization, 2048 held-out channels (Htest[4096:6144]), Gaussian symbols (stest) -- the data of
Fig. 4. Every mean is checked against the stored curve it must reproduce (snr_curves.json / wmmse_curves.json).

  python rate_gain_table.py      # -> exp_results/rate_gain_table.json
"""
import glob
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpu_timing as C  # noqa: E402  (load_model: the continued GNN models, argmax decisions)
from exp_cellfree_ablation import sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import load_data, load_levels, linear_quantized, run_model_warm, EVAL, PT  # noqa: E402
from reeval_sweep import nv  # noqa: E402
from wmmse_baseline import best_wmmse, quantized as wmmse_quantized  # noqa: E402

SNRS = (0.0, 10.0, 20.0, 30.0)
DEV = torch.device('cpu')
OUT = os.path.join(HERE, 'exp_results', 'rate_gain_table.json')
TOL = 2e-3


SNR_GRID = [float(x) for x in range(-30, 31, 5)]       # grid of every stored curve


def refs(K, b):
    """Stored curves the recomputed means must reproduce (the sources of Figs. 3 and 4, cf. plot_continue_figures.py).
    K = 1: ZF/MRT + DAC is MRT + DAC and WMMSE coincides with MRT; K = 1, b = 1 uses the separately continued model."""
    if (K, b) == (1, 1):
        sweep = glob.glob(os.path.join(HERE, 'stored_models_cellfree_sweep_K_bits', 'M_40_K_1_1bit_*'))[0]
        k1 = glob.glob(os.path.join(HERE, 'stored_models_cellfree_k1_continue', '*_ep20to40_cosine'))[0]
        ref = {'gnn': json.load(open(os.path.join(k1, 'snr_curves_ls.json')))['gnn'],
               'lin_q': json.load(open(os.path.join(sweep, 'snr_curves.json')))['mrt_1bit']}
    else:
        sweep = glob.glob(os.path.join(HERE, 'stored_models_cellfree_sweep_K_bits', f'M40_K{K}_b{b}_ep15_*'))[0]
        c = json.load(open(glob.glob(os.path.join(HERE, 'stored_models_cellfree_sweep_continue', f'M40_K{K}_b{b}_*',
                                                  'snr_curves.json'))[0]))
        ref = {'gnn': c['gnn'], 'lin_q': c['linear_quantized']}
    wref = json.load(open(os.path.join(sweep, 'wmmse_curves.json')))
    assert wref['snr_db'] == SNR_GRID
    return ref, wref


def per_ch(y, H, s, snr, chunk=512):
    return torch.cat([sumrate_bussgang_ls(y[b:b + chunk], H[b:b + chunk], s[b:b + chunk], nv(snr))
                      for b in range(0, H.shape[0], chunk)])


def stats(r):
    return {'mean': r.mean().item(), 'p5': torch.quantile(r.double(), 0.05).item()}


def main():
    torch.set_num_threads(int(os.environ.get('THREADS', 32)))
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        'snr_db': SNRS, 'metrics': 'mean and 5th percentile of the per-channel LS sum rate over Htest[4096:6144]',
        'baseline': 'per metric, the better of ZF + DAC and WMMSE + DAC (equal convention)', 'results': {}}
    for K in (1, 2, 4, 6):
        if all(f'K{K}b{b}' in out['results'] for b in (1, 2, 3)):
            continue
        t0 = time.time()
        _, _, Hte, Ste = load_data(K, 1, DEV)
        H, s = Hte[EVAL], Ste[EVAL]
        Vs = {x: best_wmmse(H, nv(x)) for x in SNRS}                         # SNR-dependent, float64
        print(f'K={K}: WMMSE done ({time.time() - t0:.0f} s)', flush=True)
        for b in (1, 2, 3):
            lv = load_levels(b, DEV)
            ref, wref = refs(K, b)
            model = C.load_model(K, b, lv)
            with torch.no_grad():
                y_gnn = torch.cat([normalize_power(run_model_warm(model, H[i:i + 256], s[i:i + 256], lv, PT, 16), PT)
                                   for i in range(0, H.shape[0], 256)])
                y_zf = normalize_power(linear_quantized(H, s, lv, PT), PT)
            res = {}
            for x in SNRS:
                i = SNR_GRID.index(x)
                r = {'gnn': per_ch(y_gnn, H, s, x), 'zf_dac': per_ch(y_zf, H, s, x),
                     'wmmse_dac': per_ch(wmmse_quantized(Vs[x], s, lv, False), H, s, x)}
                st = {m: stats(v) for m, v in r.items()}
                for m, key, src in (('gnn', 'gnn', ref), ('zf_dac', 'lin_q', ref),
                                    ('wmmse_dac', 'wmmse_quantized_equal', wref)):
                    assert abs(st[m]['mean'] - src[key][i]) < TOL, (K, b, x, m, st[m]['mean'], src[key][i])
                best = {q: max(st['zf_dac'][q], st['wmmse_dac'][q]) for q in ('mean', 'p5')}
                res[f'{x:g}'] = {**st, 'best_baseline': best,
                                 'best_is': {q: max(('zf_dac', 'wmmse_dac'), key=lambda m: st[m][q]) for q in ('mean', 'p5')},
                                 'd_mu_pct': 100 * (st['gnn']['mean'] / best['mean'] - 1),
                                 'd_5_pct': 100 * (st['gnn']['p5'] / best['p5'] - 1)}
            out['results'][f'K{K}b{b}'] = res
            print(f'K={K} b={b}: ' + ' | '.join(f"{x} dB ({v['d_mu_pct']:+.1f}%, {v['d_5_pct']:+.1f}%)"
                                               for x, v in res.items()) + f'  ({time.time() - t0:.0f} s)', flush=True)
            json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
