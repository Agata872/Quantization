"""Sum rate under residual inter-AP carrier phase offsets (Sec. VI-A, eq. (tp_sindr_drift)): the final GNN models,
trained with sigma_theta = 0, vs. ZF/MRT + DAC, WMMSE + DAC, IDE with calibrated beta and the unquantized ZF/MRT and
WMMSE precoders, on the channels and symbols of Figs. 3 and 4 (Htest[4096:6144], 125 Gaussian symbols per channel).

IDE with calibrated beta (ide_cal; Wang et al., TWC 2018, Alg. 1, as in ide_calibrated_check.py and Fig. 6): per
channel realization and SNR, beta = ide_baseline.calibrated_beta(...), the value of the block update (26) on an
independent training block of 125 Gaussian symbol vectors (seed SEED_CAL, never the data), then every data symbol
vector is precoded independently with this fixed beta. Like WMMSE, it is designed separately for every SNR.

Offsets: theta_m ~ N(0, sigma_theta^2) i.i.d. over the M APs, constant over the 125-symbol block and unknown to the
precoder; the radiated block is A y with A = diag(exp(j theta)). A is diagonal, so the LS Bussgang fit of A y is A G
and A Cq A^H, and the sum rate of A y over H equals that of y over the rotated channel A H. The rates below rotate
the channel (checked against rotating the samples); sigma_theta = 0 reproduces the stored curves of Figs. 3 and 4
(asserted, as in rate_gain_table.py).

Per (K, b, method, sigma_theta, SNR), over the 2048 channels x N_THETA offset draws per channel (common random
numbers: the same standard-normal draws z for every method and every sigma_theta, theta = sigma_theta z):
  mean          mean sum rate
  std           standard deviation over all (channel, draw) pairs; by the law of total variance
                std^2 = std_channel^2 + std_offset^2 with
  std_channel   sqrt(Var_H(E_theta[R | H]))   spread over the channel realizations
  std_offset    sqrt(E_H[Var_theta(R | H)])  spread over the offset draws for a fixed channel
  p5            5th percentile over all (channel, draw) pairs
  kappa         mean over the channels of the rate with the offset-averaged signal, interference and distortion
                powers of eq. (tp_kappa), kappa = exp(-sigma_theta^2) -- the analysis of Sec. VI-A
and, per (K, b, sigma_theta, SNR), the paired gain of the GNN over ZF/MRT + DAC, WMMSE + DAC and IDE with
calibrated beta per (channel, draw): mean, std, p5 and the fraction of pairs in which the GNN is better.

  python phase_offset_eval.py            # -> exp_results/phase_offset_eval.json
  python phase_offset_eval.py --K 2 --ide-device cuda:0   # only K = 2 (workers per K share the json), IDE on GPU
  python phase_offset_eval.py --quick    # 128 channels, K = 2, b = 1, no output file
An existing json is resumed: only the missing methods are computed (the GNN is always recomputed, since every gain is
paired with it, and must reproduce its stored statistics).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpu_timing as C  # noqa: E402  (load_model: the continued GNN models, argmax decisions)
from exp_cellfree_ablation import sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import (load_data, load_levels, linear_quantized, linear_unquantized,  # noqa: E402
                         run_model_warm, EVAL, PT)
from reeval_sweep import nv  # noqa: E402
from wmmse_baseline import best_wmmse, quantized as wmmse_quantized  # noqa: E402
import ide_baseline  # noqa: E402

SIGMA_DEG = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0)
SNRS = tuple(float(x) for x in range(-10, 31, 5))      # the SNR range of Figs. 3 and 4
SNR_GRID = [float(x) for x in range(-30, 31, 5)]       # grid of every stored curve
N_THETA = 32
SEED = 20260930
SEED_CAL = 777         # training block of the calibrated beta of IDE (as ide_calibrated_check.py)
CHUNK = 256
TOL = 2e-3
DEV = torch.device('cpu')
OUT = os.path.join(HERE, 'exp_results', 'phase_offset_eval.json')
IDE_CHECK = os.path.join(HERE, 'exp_results', 'ide_calibrated_check.json')   # K = 2, 20 dB, first channels
METHODS = {'gnn': 'final GNN models (continued; argmax), trained with sigma_theta = 0',
           'zf_dac': 'ZF (MRT for K = 1) + b-bit DAC', 'wmmse_dac': 'WMMSE + b-bit DAC (equal convention)',
           'zf_unq': 'unquantized ZF (MRT for K = 1)', 'wmmse_unq': 'unquantized WMMSE (K > 1)',
           'ide_cal': 'IDE (Wang et al., TWC 2018, Alg. 1, T = 100, alpha = 0.95) with calibrated fixed beta: per '
                      'channel and SNR, the block update (26) on an independent block of 125 Gaussian symbol '
                      f'vectors (seed {SEED_CAL}), then fixed; every symbol vector precoded independently'}


def one(pattern):
    ds = glob.glob(pattern)
    assert len(ds) == 1, (pattern, ds)
    return ds[0]


def refs(K, b):
    """Stored mean curves (SNR_GRID) that sigma_theta = 0 must reproduce; the sources of Figs. 3 and 4
    (cf. plot_continue_figures.load)."""
    if (K, b) == (1, 1):
        src = one(os.path.join(HERE, 'stored_models_cellfree_sweep_K_bits', 'M_40_K_1_1bit_*'))
        k1 = one(os.path.join(HERE, 'stored_models_cellfree_k1_continue', '*_ep20to40_cosine'))
        s = json.load(open(os.path.join(src, 'snr_curves.json')))
        r = {'gnn': json.load(open(os.path.join(k1, 'snr_curves_ls.json')))['gnn'],
             'zf_dac': s['mrt_1bit'], 'zf_unq': s['mrt_unquantized']}
    else:
        src = one(os.path.join(HERE, 'stored_models_cellfree_sweep_K_bits', f'M40_K{K}_b{b}_ep15_*'))
        c = json.load(open(one(os.path.join(HERE, 'stored_models_cellfree_sweep_continue', f'M40_K{K}_b{b}_*',
                                            'snr_curves.json'))))
        r = {'gnn': c['gnn'], 'zf_dac': c['linear_quantized'], 'zf_unq': c['linear_unquantized']}
    w = json.load(open(os.path.join(src, 'wmmse_curves.json')))
    assert w['snr_db'] == SNR_GRID
    r.update(wmmse_dac=w['wmmse_quantized_equal'], wmmse_unq=w['wmmse_unquantized'])
    return r


def bussgang_ls(y, s):
    """LS Bussgang gain G (N x M x K) and distortion covariance Cq (N x M x M), as in sumrate_bussgang_ls."""
    ns = y.shape[-1]
    Rys = (y @ s.conj().transpose(1, 2)) / ns
    Rss = (s @ s.conj().transpose(1, 2)) / ns
    G = Rys @ torch.linalg.inv(Rss)
    q = y - G @ s
    return G, (q @ q.conj().transpose(1, 2)) / ns


def rotated_terms(H, G, Cq, rot):
    """Signal, interference and distortion power of every user over the rotated channels A H.
    H, G: n x M x K; Cq: n x M x M; rot = exp(j theta): n x D x M. Returns three n x D x K tensors."""
    Hr = rot[..., None] * H[:, None]                                  # n x D x M x K
    HTG = torch.einsum('ndmk,nmj->ndkj', Hr, G)                        # n x D x K x K
    sig = HTG.diagonal(dim1=-2, dim2=-1).abs() ** 2
    itf = (HTG.abs() ** 2).sum(-1) - sig
    dst = (Hr * (Cq[:, None] @ Hr.conj())).sum(2).real                 # h_k^T A Cq A^H h_k^*
    return sig, itf, dst


def kappa_terms(H, G, Cq, kappa):
    """Offset-averaged signal, interference and distortion powers of eq. (tp_kappa): n x K each.
    E|h_k^T A g_j|^2 = kappa |h_k^T g_j|^2 + (1 - kappa) sum_m |h_mk|^2 |G_mj|^2, and
    E h_k^T A Cq A^H h_k^* = kappa h_k^T Cq h_k^* + (1 - kappa) sum_m |h_mk|^2 [Cq]_mm."""
    H2 = H.abs() ** 2
    P = kappa * (H.transpose(1, 2) @ G).abs() ** 2 + (1 - kappa) * (H2.transpose(1, 2) @ G.abs() ** 2)
    sig = P.diagonal(dim1=1, dim2=2)
    coh = (H.transpose(1, 2) @ Cq @ H.conj()).diagonal(dim1=1, dim2=2).real
    inc = (H2 * Cq.diagonal(dim1=1, dim2=2).real[..., None]).sum(1)
    return sig, P.sum(-1) - sig, kappa * coh + (1 - kappa) * inc


def offset_rates(y, s, H, z, snrs):
    """Sum rate per (sigma_theta, SNR, channel, draw) of the power-normalized samples y (N x M x Ns), and the
    kappa-model rate per (sigma_theta, SNR, channel)."""
    Hd = H.to(torch.complex128)
    G, Cq = bussgang_ls(y.to(torch.complex128), s.to(torch.complex128))
    N, D = z.shape[:2]
    nvs = torch.tensor([nv(x) for x in snrs], dtype=torch.float64)
    R = torch.empty(len(SIGMA_DEG), len(snrs), N, D, dtype=torch.float64)
    Rk = torch.empty(len(SIGMA_DEG), len(snrs), N, dtype=torch.float64)
    for i, sd in enumerate(SIGMA_DEG):
        sr = float(np.deg2rad(sd))
        for c in range(0, N, CHUNK):
            sl = slice(c, c + CHUNK)
            rot = torch.polar(torch.ones_like(z[sl]), sr * z[sl])
            sig, itf, dst = rotated_terms(Hd[sl], G[sl], Cq[sl], rot)
            R[i, :, sl] = torch.log2(1 + sig[None] / ((itf + dst)[None] + nvs[:, None, None, None])).sum(-1)
        sig, itf, dst = kappa_terms(Hd, G, Cq, float(np.exp(-sr ** 2)))
        Rk[i] = torch.log2(1 + sig[None] / ((itf + dst)[None] + nvs[:, None, None])).sum(-1)
    return R, Rk


def stats(R):
    """R: channels x draws."""
    within = R.var(1, unbiased=False).mean()          # E_H[Var_theta(R | H)]
    between = R.mean(1).var(unbiased=False)           # Var_H(E_theta[R | H])
    return {'mean': R.mean().item(), 'std': R.flatten().var(unbiased=False).sqrt().item(),
            'std_channel': between.sqrt().item(), 'std_offset': within.sqrt().item(),
            'p5': torch.quantile(R.flatten(), 0.05).item()}


def gain_stats(Rg, Rb):
    d = (Rg - Rb).flatten()
    return {'mean': d.mean().item(), 'std': d.var(unbiased=False).sqrt().item(),
            'p5': torch.quantile(d, 0.05).item(), 'frac_better': (d > 0).double().mean().item()}


def grid(fn):
    """[sigma][snr] nested list."""
    return [[fn(i, j) for j in range(len(SNRS))] for i in range(len(SIGMA_DEG))]


def gaussian_block(n, K, seed):
    """Training block of the calibrated beta of IDE: n x K x 125 i.i.d. CN(0, 1) symbols, the distribution of the
    data (as ide_calibrated_check.py)."""
    x = torch.randn((n, K, ide_baseline.NS, 2), generator=torch.Generator().manual_seed(seed)) / np.sqrt(2)
    return torch.complex(x[..., 0], x[..., 1])


def ide_cal(H, s, train, lv, snr_db, device=None):
    beta = ide_baseline.calibrated_beta(H, train, lv, snr_db, device=device)
    return ide_baseline.precode(H, s, lv, snr_db, 'ide', 'fixed', beta_fixed=beta, device=device)


def methods_of(K):
    return [m for m in METHODS if K > 1 or not m.startswith('wmmse')]


def evaluate(K, b, H, s, z, Vs, quick, methods=METHODS, ide_device=None):
    """Statistics of `methods` (the GNN is always evaluated, since every gain is paired with it) and the gains of the
    GNN over the baselines among them. Vs: WMMSE precoders per SNR (needed for the WMMSE methods only).
    ide_device: where IDE runs (default CPU; the GPU gave identical decisions)."""
    lv = load_levels(b, DEV)
    model = C.load_model(K, b, lv)
    with torch.no_grad():
        ys = {'gnn': torch.cat([normalize_power(run_model_warm(model, H[i:i + 256], s[i:i + 256], lv, PT, 16), PT)
                                for i in range(0, H.shape[0], 256)])}
        if 'zf_dac' in methods:
            ys['zf_dac'] = normalize_power(linear_quantized(H, s, lv, PT), PT)
        if 'zf_unq' in methods:
            ys['zf_unq'] = normalize_power(linear_unquantized(H, s, PT), PT)
    R, Rk = {}, {}
    for m, y in ys.items():                                            # samples independent of the SNR
        R[m], Rk[m] = offset_rates(y, s, H, z, SNRS)
    train = gaussian_block(H.shape[0], K, SEED_CAL)
    per_snr = (('wmmse_dac', lambda x: wmmse_quantized(Vs[x], s, lv, False)),
               ('wmmse_unq', lambda x: normalize_power(Vs[x] @ s.to(Vs[x].dtype), PT)),
               ('ide_cal', lambda x: ide_cal(H, s, train, lv, x, ide_device)))
    for m, yfn in per_snr:                                             # WMMSE and IDE: one precoder per SNR
        if m not in methods or (K == 1 and m.startswith('wmmse')):
            continue
        parts, ys[m] = [], {}
        for x in SNRS:
            y = yfn(x)
            parts.append(offset_rates(y, s, H, z, (x,)))
            direct = sumrate_bussgang_ls(y, H, s, nv(x)).double()
            assert (parts[-1][0][SIGMA_DEG.index(0.0), 0, :, 0] - direct).abs().max() < 1e-3, (m, x)
            if x == 20.0:
                ys[m][x] = y
        R[m] = torch.cat([p[0] for p in parts], 1)
        Rk[m] = torch.cat([p[1] for p in parts], 1)

    # -- checks: sigma_theta = 0 equals the reference rate function (and, on the full set, the stored curves);
    #    rotating the channel equals rotating the samples
    ref = None if quick else refs(K, b)
    i0, i20, j20 = SIGMA_DEG.index(0.0), SIGMA_DEG.index(20.0), SNRS.index(20.0)
    for m in R:
        assert R[m][i0].var(2).max() < 1e-20, m                         # no offsets: all draws coincide
        for j, x in enumerate(SNRS):
            if m in ys and not isinstance(ys[m], dict):
                direct = sumrate_bussgang_ls(ys[m], H, s, nv(x)).double()
                assert (R[m][i0, j, :, 0] - direct).abs().max() < 1e-3, (m, x)
            if ref is not None and m in ref:
                stored = ref[m][SNR_GRID.index(x)]
                assert abs(R[m][i0, j].mean().item() - stored) < TOL, (K, b, m, x, R[m][i0, j].mean().item(), stored)
    if 'ide_cal' in R and K == 2 and os.path.exists(IDE_CHECK):          # the independent quick check, same symbols
        chk = json.load(open(IDE_CHECK))
        n = chk['n_channels']
        if n <= H.shape[0]:
            for m in ('ide_cal', 'gnn'):
                got, want = R[m][i0, j20, :n, 0].mean().item(), chk['results'][f'b{b}']['sum_rate'][m]
                assert abs(got - want) < TOL, (m, b, got, want)
    rot = torch.polar(torch.ones_like(z[:, 0]), float(np.deg2rad(20.0)) * z[:, 0]).to(torch.complex64)
    for m in ('gnn', 'zf_dac', 'wmmse_dac', 'ide_cal'):
        if m not in ys:
            continue
        y = ys[m][20.0] if isinstance(ys[m], dict) else ys[m]
        direct = sumrate_bussgang_ls(rot[..., None] * y, H, s, nv(20.0)).double()
        assert (R[m][i20, j20, :, 0] - direct).abs().max() < 1e-3, m

    res = {'methods': {}, 'gains': {}}
    for m in R:
        st = grid(lambda i, j: stats(R[m][i, j]))
        res['methods'][m] = {k: [[c[k] for c in row] for row in st] for k in st[0][0]}
        res['methods'][m]['kappa'] = grid(lambda i, j: Rk[m][i, j].mean().item())
    for base in ('zf_dac', 'wmmse_dac', 'ide_cal'):
        if base in R:
            st = grid(lambda i, j: gain_stats(R['gnn'][i, j], R[base][i, j]))
            res['gains'][f'vs_{base}'] = {k: [[c[k] for c in row] for row in st] for k in st[0][0]}
    return res


def summary(K, b, res):
    i10, i20, j20 = SIGMA_DEG.index(10.0), SIGMA_DEG.index(20.0), SNRS.index(20.0)
    parts = []
    for m, d in res['methods'].items():
        parts.append(f"{m} {d['mean'][0][j20]:.2f}->{d['mean'][i10][j20]:.2f}->{d['mean'][i20][j20]:.2f} "
                     f"(std_off@20deg {d['std_offset'][i20][j20]:.3f}, kappa {d['kappa'][i20][j20]:.2f})")
    return f'K={K} b={b} @20 dB, sigma 0/10/20 deg: ' + ' | '.join(parts)


def load_out():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        'sigma_deg': SIGMA_DEG, 'snr_db': SNRS, 'n_theta': N_THETA, 'seed': SEED,
        'channels': 'Htest[4096:6144], 125 Gaussian symbols per channel',
        'estimator': 'LS Bussgang estimator (sumrate_bussgang_ls) over the rotated channel A H',
        'layout': 'every statistic is a [sigma][snr] list', 'results': {}}
    out.update(methods=METHODS, seed_cal=SEED_CAL)
    return out


def save(key, res):
    """Re-read the json and replace only `key`, written atomically, so that workers for different K can share it."""
    out = load_out()
    out['results'][key] = res
    json.dump(out, open(OUT + '.tmp', 'w'), indent=1)
    os.replace(OUT + '.tmp', OUT)


def merge(key, old, new, todo):
    """Add the newly computed methods to a stored entry; the recomputed GNN must reproduce its stored statistics."""
    for k, g in old['methods']['gnn'].items():
        dev = np.abs(np.array(g) - np.array(new['methods']['gnn'][k])).max()
        assert dev < TOL, (key, 'gnn', k, dev)
    old['methods'].update({m: new['methods'][m] for m in todo})
    old['gains'].update({g: v for g, v in new['gains'].items() if g not in old['gains']})
    return old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--K', default='2,4,6,1', help='comma-separated numbers of UEs')
    ap.add_argument('--ide-device', default=None, help='e.g. cuda:0 (IDE only; default CPU)')
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', 32)))
    for K in ((2,) if args.quick else tuple(int(k) for k in args.K.split(','))):
        bits = (1,) if args.quick else (1, 2, 3)
        stored = {} if args.quick else load_out()['results']
        todo = {b: [m for m in methods_of(K) if m not in stored.get(f'K{K}b{b}', {}).get('methods', {})]
                for b in bits}
        if not any(todo.values()):
            continue
        t0 = time.time()
        _, _, Hte, Ste = load_data(K, 1, DEV)
        H, s = Hte[EVAL], Ste[EVAL]
        if args.quick:
            H, s = H[:128], s[:128]
        z = torch.randn(H.shape[0], N_THETA, H.shape[1], dtype=torch.float64,
                        generator=torch.Generator().manual_seed(SEED + K))
        need_wmmse = K > 1 and any(m.startswith('wmmse') for ms in todo.values() for m in ms)
        Vs = {x: best_wmmse(H, nv(x)) for x in SNRS} if need_wmmse else {}
        print(f'K={K}: data{" and WMMSE" if need_wmmse else ""} done ({time.time() - t0:.0f} s)', flush=True)
        for b in bits:
            if not todo[b]:
                continue
            key = f'K{K}b{b}'
            res = evaluate(K, b, H, s, z, Vs, args.quick, todo[b], args.ide_device)
            if key in stored:
                res = merge(key, stored[key], res, todo[b])
            print(summary(K, b, res) + f'  ({time.time() - t0:.0f} s)', flush=True)
            if not args.quick:
                save(key, res)


if __name__ == '__main__':
    main()
