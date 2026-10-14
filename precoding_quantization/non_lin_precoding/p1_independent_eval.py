"""Prompt 1: the frozen paper checkpoints and every baseline re-evaluated on an independent test set.

Test set: Htest[6144:8192] of the M_40_K_{K} dataset of each K, with its stored Gaussian symbols (stest, 125
per channel). The paper selected checkpoints and reported results on Htest[4096:6144] ('EVAL'); tuning used
Htest[0:128] / [0:256]. Stage 'independence' documents that no script reads [6144:8192] and that the slice
has no duplicate in the training data. Nothing here trains, tunes or selects: the checkpoints, the IDE /
SQUID / WMMSE / GNN-GD recipes and their tuning rules are those of the paper, applied unchanged.

Stages (each resumable, outputs in exp_results/independent_test/):
  independence  code scan of every .py file, duplicate check against Htrain/Hval/other Htest rows,
                two-sample KS tests of the channel statistics             -> independence.json
  rates         per-channel LS sum rate on -30:30:5 dB of GNN, ZF/MRT + DAC, unquantized ZF/MRT,
                WMMSE + DAC (equal convention) and unquantized WMMSE, per-user rates at 20/30 dB.
                First reproduces the stored 20 dB values on EVAL with the same code (check).
                                                                            -> rates_K{K}b{b}.npz
  gains         Table I (gain over WMMSE + DAC, mean and 5th percentile) at 0/10/20/30 dB with paired
                bootstrap 95% CIs over channels (ratio of means / ratio of p5s per resample)
                                                                            -> gains.json
  ber           16-QAM BER at 20 dB (Fig. 6): GNN, IDE calibrated / block beta, ZF/MRT + DAC, SQUID (b=1;
                gain re-selected by the paper rule on Htest[0:128]); per-channel BER  -> ber_K{K}b{b}.npz
  phase         Fig. 7 protocol at 20 dB (32 offset draws per channel, sigma 0..30 deg)  -> phase.json
  idrates       IDE calibrated / block beta and SQUID sum rates at 20 and 30 dB   -> idrates_K{K}b{b}.npz
  gd            GNN-GD (200 Adam steps from the GNN output, recipe of gd_refine_sweep) at 20 dB (Fig. 5)
                                                                            -> gd_K{K}b{b}.npz
  report        old (EVAL) vs. new (TEST) tables                            -> report.md, summary.json

  python p1_independent_eval.py --stage rates            # CPU; THREADS env sets torch threads
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpu_timing as C  # noqa: E402  (load_model: continued GNN checkpoints, argmax)
from exp_cellfree_ablation import sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import (load_levels, linear_quantized, linear_unquantized, run_model_warm,  # noqa: E402
                         EVAL, PT, M_FULL)
from reeval_sweep import nv  # noqa: E402
from wmmse_baseline import best_wmmse, quantized as wmmse_quantized  # noqa: E402

TEST = slice(6144, 8192)
TUNE = slice(0, 128)
NS = 125
SNR_GRID = [float(x) for x in range(-30, 31, 5)]
I20, I30 = SNR_GRID.index(20.0), SNR_GRID.index(30.0)
OUT = os.path.join(HERE, 'exp_results', 'independent_test')
DEV = torch.device('cpu')
CONFIGS = [(K, b) for K in (1, 2, 4, 6) for b in (1, 2, 3)]
SEED_QAM_TEST = 6144          # 16-QAM data symbols of the test set (the paper's EVAL used 1234)
SEED_BOOT = 20261014
N_BOOT = 10000
TOL = 2e-3


def dataset_dir(K):
    ds = glob.glob(os.path.join(HERE, 'datasets', 'cellfree', f'M_{M_FULL}_K_{K}_Ntr_200000_*'))
    assert len(ds) == 1, ds
    return ds[0]


def load_slice(K, sl):
    d = dataset_dir(K)
    H = np.load(os.path.join(d, 'Htest.npy'), mmap_mode='r')
    s = np.load(os.path.join(d, 'stest.npy'), mmap_mode='r')
    n = H.shape[0]
    s = s.reshape(K, n, NS)[:, sl].transpose(1, 0, 2)
    return (torch.from_numpy(np.ascontiguousarray(H[sl]).astype(np.complex64)),
            torch.from_numpy(np.ascontiguousarray(s).astype(np.complex64)))


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def ckpt_path(K, b):
    pat = ('stored_models_cellfree_k1_continue/*/model_best.pt' if (K, b) == (1, 1)
           else f'stored_models_cellfree_sweep_continue/M40_K{K}_b{b}_*/model_best.pt')
    ps = glob.glob(os.path.join(HERE, pat))
    assert len(ps) == 1, ps
    return ps[0]


def per_ch(y, H, s, snr, chunk=512):
    return torch.cat([sumrate_bussgang_ls(y[i:i + chunk], H[i:i + chunk], s[i:i + chunk], nv(snr))
                      for i in range(0, H.shape[0], chunk)])


def per_user(y, H, s, snr):
    ns = y.shape[-1]
    G = ((y @ s.conj().transpose(1, 2)) / ns) @ torch.linalg.inv((s @ s.conj().transpose(1, 2)) / ns)
    q = y - G @ s
    HT = H.transpose(1, 2)
    HTG = HT @ G
    sig = HTG.diagonal(dim1=1, dim2=2).abs() ** 2
    itf = (HTG.abs() ** 2).sum(-1) - sig
    dst = (HT @ ((q @ q.conj().transpose(1, 2)) / ns) @ H.conj()).diagonal(dim1=1, dim2=2).real
    return torch.log2(1 + sig / (itf + dst + nv(snr)))


@torch.no_grad()
def gnn_output(K, b, H, s, lv):
    model = C.load_model(K, b, lv)
    return torch.cat([normalize_power(run_model_warm(model, H[i:i + 256], s[i:i + 256], lv, PT, 16), PT)
                      for i in range(0, H.shape[0], 256)])


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


# ------------------------------------------------------------------------------------------- independence
def stage_independence():
    rep = {'test_slice': [TEST.start, TEST.stop], 'checked': f'{datetime.now():%Y-%m-%d %H:%M:%S}'}
    # 1) code scan: every Htest slice any script uses
    pat_slice = re.compile(r'slice\(\s*(\d+)\s*,\s*(\d+)\s*\)')
    pat_idx = re.compile(r'\[\s*(\d+)\s*:\s*(\d+)\s*\]')
    uses, full = [], []
    root = os.path.dirname(HERE)
    for dp, _, fs in os.walk(root):
        if any(x in dp for x in ('/env', '__pycache__', '/.git')):
            continue
        for f in fs:
            if not f.endswith('.py') or f in ('p1_independent_eval.py', 'p2_ablation.py'):
                continue
            path = os.path.join(dp, f)
            for ln, line in enumerate(open(path, errors='replace'), 1):
                for m in list(pat_slice.finditer(line)) + list(pat_idx.finditer(line)):
                    a, z = int(m.group(1)), int(m.group(2))
                    if z >= 128 and ('Hte' in line or 'Htest' in line or 'EVAL' in line or 'TUNE' in line
                                     or 'slice(' in line and ('=' in line)):
                        uses.append({'file': os.path.relpath(path, root), 'line': ln, 'range': [a, z],
                                     'overlaps_test': a < TEST.stop and z > TEST.start})
                if re.search(r'ChannelSymbolsDataset\(\s*Htest', line):
                    full.append({'file': os.path.relpath(path, root), 'line': ln})
    rep['code_slices'] = uses
    rep['code_full_Htest_users'] = full
    rep['code_full_Htest_note'] = (
        'old training.py pipeline (Feys-style raw-input GNNs): it evaluates whole Htest after training with Hval '
        'as validation; its stored models are M=16 K=2, M=8 K=1, M=32 K=1, M=40 K=1 '
        '(stored_models_cellfree_generalized_bussgang_loss*), none of them a paper model, and no such run '
        'exists for M=40 with K>=2. Hyperparameters of the paper recipe were chosen on Htest[4096:6144] '
        '(exp_cellfree_ablation.py rounds 1-6, K=1) and in train_sweep.py short runs (same slice).')
    rep['any_code_slice_overlaps_test'] = any(u['overlaps_test'] for u in uses)
    # 2) data: duplicates and distribution
    from scipy.stats import ks_2samp
    per_k = {}
    for K in (1, 2, 4, 6):
        d = dataset_dir(K)
        Htr = np.load(os.path.join(d, 'Htrain.npy'), mmap_mode='r')
        Hva = np.load(os.path.join(d, 'Hval.npy'), mmap_mode='r')
        Hte = np.load(os.path.join(d, 'Htest.npy'))
        T = Hte[TEST]
        key = lambda A: {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in A}  # noqa: E731
        kt = key(T)
        others = {'Htrain': key(Htr), 'Hval': key(Hva),
                  'Htest_outside_test': key(np.concatenate([Hte[:TEST.start], Hte[TEST.stop:]]))}
        dup = {n: len(kt & k) for n, k in others.items()}

        def feats(A):
            A = np.asarray(A)
            p = (np.abs(A) ** 2).sum(-1)                                   # per-AP power, n x M
            tot = p.sum(-1)
            return {'log10_fro2': np.log10(tot), 'strongest_ap_share': p.max(-1) / tot}
        fT, fE, fR = feats(T), feats(Hte[EVAL]), feats(Htr[:100000])
        ks = {f: {'vs_eval': float(ks_2samp(fT[f], fE[f]).pvalue), 'vs_train100k': float(ks_2samp(fT[f], fR[f]).pvalue)}
              for f in fT}
        per_k[f'K{K}'] = {'dataset': os.path.relpath(d, HERE), 'unique_test_rows': len(kt),
                          'duplicates_of_test_rows_in': dup, 'ks_pvalues': ks,
                          'sha256_Htest': sha256_file(os.path.join(d, 'Htest.npy')),
                          'sha256_stest': sha256_file(os.path.join(d, 'stest.npy')),
                          'mean_abs_h2_test': float(np.mean(np.abs(T) ** 2))}
        log(f'K={K}: duplicates {dup}, KS p {ks}')
    rep['data'] = per_k
    rep['generation'] = ('data_handling.getdata_nonlinprec: one call per K draws Ntr+Nval+Nte channels i.i.d. '
                         '(fixed AP layout of that call, new UE positions, shadowing and fading per realization), '
                         'scales all of them by one common constant, then splits train/val/test in order; symbols '
                         'i.i.d. CN(0,1). The test rows are therefore independent draws from the training '
                         'distribution; the AP layout is shared by design (one deployment per K).')
    json.dump(rep, open(os.path.join(OUT, 'independence.json'), 'w'), indent=1)
    log(f"code slices overlapping the test slice: {rep['any_code_slice_overlaps_test']}")


# ------------------------------------------------------------------------------------------- rates
def stage_rates(configs):
    for K in sorted({k for k, _ in configs}):
        bs = [b for k, b in configs if k == K and not os.path.exists(os.path.join(OUT, f'rates_K{K}b{b}.npz'))]
        if not bs:
            continue
        t0 = time.time()
        He, se = load_slice(K, EVAL)
        H, s = load_slice(K, TEST)
        Vs = {x: best_wmmse(H, nv(x)) for x in SNR_GRID} if K > 1 else None
        Ve20 = best_wmmse(He, nv(20.0)) if K > 1 else None
        log(f'K={K}: WMMSE done ({time.time() - t0:.0f} s)')
        for b in bs:
            lv = load_levels(b, DEV)
            # -- check on EVAL: the same code reproduces the stored 20 dB values of the paper
            yg_e = gnn_output(K, b, He, se, lv)
            chk = {'gnn': per_ch(yg_e, He, se, 20.0).mean().item(),
                   'lin_q': per_ch(normalize_power(linear_quantized(He, se, lv, PT), PT), He, se, 20.0).mean().item()}
            if K > 1:
                chk['wmmse_q'] = per_ch(wmmse_quantized(Ve20, se, lv, False), He, se, 20.0).mean().item()
            old = json.load(open(os.path.join(HERE, 'exp_results', 'rate_gain_table.json')))['results'][f'K{K}b{b}']['20']
            ref = {'gnn': old['gnn']['mean'], 'lin_q': old['zf_dac']['mean'], 'wmmse_q': old['wmmse_dac']['mean']}
            for m, v in chk.items():
                assert abs(v - ref[m]) < TOL, (K, b, m, v, ref[m])
            # -- test slice
            with torch.no_grad():
                y = {'gnn': gnn_output(K, b, H, s, lv),
                     'lin_q': normalize_power(linear_quantized(H, s, lv, PT), PT),
                     'lin_unq': normalize_power(linear_unquantized(H, s, PT), PT)}
            R = {m: torch.stack([per_ch(v, H, s, x) for x in SNR_GRID]).numpy() for m, v in y.items()}
            PU = {f'{m}_pu{x:g}': per_user(y[m], H, s, x).numpy() for m in y for x in (20.0, 30.0)}
            if K > 1:
                R['wmmse_q'] = torch.stack([per_ch(wmmse_quantized(Vs[x], s, lv, False), H, s, x)
                                            for x in SNR_GRID]).numpy()
                R['wmmse_unq'] = torch.stack([per_ch(normalize_power(Vs[x] @ s.to(Vs[x].dtype), PT).to(torch.complex64),
                                                     H, s, x) for x in SNR_GRID]).numpy()
                for x in (20.0, 30.0):
                    PU[f'wmmse_q_pu{x:g}'] = per_user(wmmse_quantized(Vs[x], s, lv, False), H, s, x).numpy()
            else:                                   # K = 1: WMMSE reduces to MRT
                R['wmmse_q'], R['wmmse_unq'] = R['lin_q'], R['lin_unq']
            np.savez(os.path.join(OUT, f'rates_K{K}b{b}.npz'), snr_db=np.array(SNR_GRID), **R, **PU,
                     eval_check=json.dumps({'recomputed_eval_20db': chk, 'stored_eval_20db': ref}),
                     checkpoint=ckpt_path(K, b), checkpoint_sha256=sha256_file(ckpt_path(K, b)))
            log(f'K={K} b={b} @20 dB test: GNN {R["gnn"][I20].mean():.3f} | lin+DAC {R["lin_q"][I20].mean():.3f} | '
                f'WMMSE+DAC {R["wmmse_q"][I20].mean():.3f} | WMMSE unq {R["wmmse_unq"][I20].mean():.3f} '
                f'(EVAL check {chk})  ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------- gains
def boot_ratio(a, b, stat, n=N_BOOT, seed=SEED_BOOT):
    """Paired bootstrap over channels of 100 (stat(a) / stat(b) - 1); a, b: per-channel values."""
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(a), size=(n, len(a)))
    if stat == 'mean':
        r = a[idx].mean(1) / b[idx].mean(1)
    else:
        r = np.percentile(a[idx], 5, axis=1) / np.percentile(b[idx], 5, axis=1)
    lo, hi = np.percentile(100 * (r - 1), [2.5, 97.5])
    return float(lo), float(hi)


def gain_entry(a, b):
    mu = 100 * (a.mean() / b.mean() - 1)
    p5 = 100 * (np.percentile(a, 5) / np.percentile(b, 5) - 1)
    return {'d_mu_pct': float(mu), 'd_mu_ci95': boot_ratio(a, b, 'mean'),
            'd_5_pct': float(p5), 'd_5_ci95': boot_ratio(a, b, 'p5'),
            'mean': [float(a.mean()), float(b.mean())], 'p5': [float(np.percentile(a, 5)), float(np.percentile(b, 5))],
            'frac_channels_better': float((a > b).mean())}


def stage_gains():
    old = json.load(open(os.path.join(HERE, 'exp_results', 'rate_gain_table.json')))['results']
    out = {'definition': 'Delta = 100 (T_GNN / T_base - 1); T = mean (d_mu) or 5th percentile (d_5) of the per-channel '
                         'LS sum rate over the 2048 test channels; CI: paired bootstrap over channels, '
                         f'{N_BOOT} resamples, percentile 2.5/97.5, seed {SEED_BOOT}',
           'test_slice': 'Htest[6144:8192]', 'results': {}}
    for K, b in CONFIGS:
        f = os.path.join(OUT, f'rates_K{K}b{b}.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        res = {}
        for x in (0.0, 10.0, 20.0, 30.0):
            i = SNR_GRID.index(x)
            g, w, l = z['gnn'][i], z['wmmse_q'][i], z['lin_q'][i]
            e = {'vs_wmmse_dac': gain_entry(g, w), 'vs_lin_dac': gain_entry(g, l)}
            o = old.get(f'K{K}b{b}', {}).get(f'{x:g}')
            if o:
                e['old_eval'] = {'gnn_mean': o['gnn']['mean'], 'wmmse_mean': o['wmmse_dac']['mean'],
                                 'gnn_p5': o['gnn']['p5'], 'wmmse_p5': o['wmmse_dac']['p5'],
                                 'd_mu_pct_vs_wmmse': 100 * (o['gnn']['mean'] / o['wmmse_dac']['mean'] - 1),
                                 'd_5_pct_vs_wmmse': 100 * (o['gnn']['p5'] / o['wmmse_dac']['p5'] - 1)}
            res[f'{x:g}'] = e
        out['results'][f'K{K}b{b}'] = res
        e = res['20']['vs_wmmse_dac']
        log(f"K{K}b{b} 20 dB vs WMMSE+DAC: d_mu {e['d_mu_pct']:.1f}% [{e['d_mu_ci95'][0]:.1f}, {e['d_mu_ci95'][1]:.1f}]"
            f" (old {res['20'].get('old_eval', {}).get('d_mu_pct_vs_wmmse', float('nan')):.1f}) | d_5 {e['d_5_pct']:.1f}% "
            f"[{e['d_5_ci95'][0]:.1f}, {e['d_5_ci95'][1]:.1f}] (old {res['20'].get('old_eval', {}).get('d_5_pct_vs_wmmse', float('nan')):.1f})")
    json.dump(out, open(os.path.join(OUT, 'gains.json'), 'w'), indent=1)


# ------------------------------------------------------------------------------------------- BER
def stage_ber(configs):
    import ber_16qam_20dB as B
    import squid_baseline
    for K in sorted({k for k, _ in configs}):
        H, _ = load_slice(K, TEST)
        Ht, _ = load_slice(K, TUNE)
        s, st = B.qam16(H.shape[0], K, SEED_QAM_TEST), B.qam16(Ht.shape[0], K, 4321)
        for b in [b for k, b in configs if k == K]:
            f = os.path.join(OUT, f'ber_K{K}b{b}.npz')
            if os.path.exists(f):
                continue
            t0 = time.time()
            lv = load_levels(b, DEV)
            res, extra = {}, {}
            for m in ('gnn', 'ide_cal', 'ide_block', 'zf_dac'):
                y = B.METHODS[m](K, b, H, s, lv)
                res[m] = ber_per_channel(y, H, s, B)
                if m == 'ide_cal':
                    extra['ide_cal_beta'] = B.CAL_STATS.pop((K, b))
            if b == 1:
                N0 = 10 ** (-B.SNR_DB / 10)
                tune = {gn: B.ber(normalize_power(squid_baseline.squid(Ht, st, N0, gn), PT), Ht, st) for gn in B.GAINS}
                g_best = min(tune, key=tune.get)
                y = torch.cat([squid_baseline.squid(H[i:i + 256], s[i:i + 256], N0, g_best)
                               for i in range(0, H.shape[0], 256)])
                res['squid'] = ber_per_channel(normalize_power(y, PT), H, s, B)
                old_gain = json.load(open(B.OUT))['results'][f'K{K}b{b}']['squid_gain']
                extra.update(squid_gain=g_best, squid_gain_paper=old_gain)
            np.savez(f, **res, extra=json.dumps(extra), symbols_seed=SEED_QAM_TEST)
            log(f'BER K{K} b{b}: ' + ' | '.join(f'{m} {v.mean():.3e}' for m, v in res.items())
                + f' {extra.get("squid_gain", "")} ({time.time() - t0:.0f} s)')


def ber_per_channel(y, H, s, B):
    r = torch.einsum('nmk,nmt->nkt', H, y)
    g = (r * s.conj()).sum(-1) / (s.abs() ** 2).sum(-1)
    z = r / g[..., None]
    sd = torch.sqrt(B.NV / (2 * g.abs() ** 2))[..., None]
    pe = 0.5 * (B._pe_dim(z.real, s.real, sd) + B._pe_dim(z.imag, s.imag, sd))
    out = pe.double().mean(dim=(1, 2)).numpy()                       # per channel, all UEs and symbols
    ref = B.ber(y, H, s)                                             # float32 mean over all bits (paper code)
    assert abs(out.mean() - ref) <= 1e-5 * ref + 1e-12, (out.mean(), ref)
    return out


# ------------------------------------------------------------------------------------------- phase offsets
def stage_phase(Ks):
    import phase_offset_eval as P
    path = os.path.join(OUT, 'phase.json')
    out = json.load(open(path)) if os.path.exists(path) else {
        'sigma_deg': P.SIGMA_DEG, 'snr_db': [20.0], 'n_theta': P.N_THETA, 'seed_rule': 'SEED + K (as the paper)',
        'channels': 'Htest[6144:8192]', 'results': {}}
    for K in Ks:
        todo = [b for b in (1, 2, 3) if f'K{K}b{b}' not in out['results']]
        if not todo:
            continue
        t0 = time.time()
        H, s = load_slice(K, TEST)
        z = torch.randn(H.shape[0], P.N_THETA, H.shape[1], dtype=torch.float64,
                        generator=torch.Generator().manual_seed(P.SEED + K))
        V20 = best_wmmse(H, nv(20.0)) if K > 1 else None
        train = P.gaussian_block(H.shape[0], K, P.SEED_CAL)
        for b in todo:
            lv = load_levels(b, DEV)
            ys = {'gnn': gnn_output(K, b, H, s, lv), 'zf_dac': normalize_power(linear_quantized(H, s, lv, PT), PT),
                  'ide_cal': P.ide_cal(H, s, train, lv, 20.0)}
            if K > 1:
                ys['wmmse_dac'] = wmmse_quantized(V20, s, lv, False)
                ys['wmmse_unq'] = normalize_power(V20 @ s.to(V20.dtype), PT).to(torch.complex64)
            else:
                ys['mrt_unq'] = normalize_power(linear_unquantized(H, s, PT), PT)
            R, Rk = {}, {}
            for m, y in ys.items():
                r, rk = P.offset_rates(y, s, H, z, (20.0,))
                R[m], Rk[m] = r[:, 0], rk[:, 0]                       # sigma x channels x draws
                direct = sumrate_bussgang_ls(y, H, s, nv(20.0)).double()
                assert (R[m][0, :, 0] - direct).abs().max() < 1e-3, m
            res = {'methods': {}, 'gains': {}, 'boot': {}}
            for m in R:
                st = [P.stats(R[m][i]) for i in range(len(P.SIGMA_DEG))]
                res['methods'][m] = {k: [c[k] for c in st] for k in st[0]}
                res['methods'][m]['kappa'] = [Rk[m][i].mean().item() for i in range(len(P.SIGMA_DEG))]
            for base in ('ide_cal', 'zf_dac', 'wmmse_dac'):
                if base in R:
                    st = [P.gain_stats(R['gnn'][i], R[base][i]) for i in range(len(P.SIGMA_DEG))]
                    res['gains'][f'vs_{base}'] = {k: [c[k] for c in st] for k in st[0]}
            # paired bootstrap over channels: relative loss at 10/20 deg and gain retained over IDE
            g = np.random.default_rng(SEED_BOOT)
            idx = g.integers(0, H.shape[0], size=(2000, H.shape[0]))
            cm = {m: R[m].mean(-1).numpy() for m in R}                 # sigma x channels (offset-averaged)
            for sd in (10.0, 20.0, 30.0):
                i = P.SIGMA_DEG.index(sd)
                for m in ('gnn', 'ide_cal'):
                    loss = 100 * (1 - cm[m][i][idx].mean(1) / cm[m][0][idx].mean(1))
                    res['boot'][f'{m}_loss_pct_{sd:g}deg'] = [float(100 * (1 - cm[m][i].mean() / cm[m][0].mean())),
                                                               *np.percentile(loss, [2.5, 97.5]).tolist()]
                ret = ((cm['gnn'][i][idx].mean(1) - cm['ide_cal'][i][idx].mean(1))
                       / (cm['gnn'][0][idx].mean(1) - cm['ide_cal'][0][idx].mean(1)))
                pt_ = (cm['gnn'][i].mean() - cm['ide_cal'][i].mean()) / (cm['gnn'][0].mean() - cm['ide_cal'][0].mean())
                res['boot'][f'gain_retained_vs_ide_{sd:g}deg'] = [float(pt_), *np.percentile(ret, [2.5, 97.5]).tolist()]
            out['results'][f'K{K}b{b}'] = res
            json.dump(out, open(path + '.tmp', 'w'), indent=1)
            os.replace(path + '.tmp', path)
            m = res['methods']
            log(f"phase K{K} b{b}: GNN " + '/'.join(f"{v:.2f}" for v in np.array(m['gnn']['mean'])[[0, 4, 8, 10]])
                + " | IDE " + '/'.join(f"{v:.2f}" for v in np.array(m['ide_cal']['mean'])[[0, 4, 8, 10]])
                + f" (0/10/20/30 deg)  ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------------------------------- IDE / SQUID rates
def stage_idrates(configs):
    import ide_baseline
    import squid_baseline
    import phase_offset_eval as P
    for K, b in configs:
        f = os.path.join(OUT, f'idrates_K{K}b{b}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        H, s = load_slice(K, TEST)
        lv = load_levels(b, DEV)
        train = P.gaussian_block(H.shape[0], K, P.SEED_CAL)
        res, extra = {}, {}
        for x in (20.0, 30.0):
            res[f'ide_cal_{x:g}'] = per_ch(P.ide_cal(H, s, train, lv, x), H, s, x).numpy()
            res[f'ide_block_{x:g}'] = per_ch(ide_baseline.precode(H, s, lv, x, 'ide', 'block'), H, s, x).numpy()
            if b == 1:
                Ht, st = load_slice(K, TUNE)
                tune = {gn: squid_baseline.rate(squid_baseline.run(Ht, st, x, gn), Ht, st, x) for gn in squid_baseline.GAINS}
                gb = max(tune, key=tune.get)
                res[f'squid_{x:g}'] = per_ch(squid_baseline.run(H, s, x, gb), H, s, x).numpy()
                extra[f'squid_gain_{x:g}'] = gb
        np.savez(f, **res, extra=json.dumps(extra))
        log(f'idrates K{K} b{b}: ' + ' | '.join(f'{k} {v.mean():.3f}' for k, v in res.items())
            + f' {extra} ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------- GNN-GD
def stage_gd(configs, device):
    from gd_baseline import gd_optimize
    from gd_refine_sweep import recipe
    cfg = recipe()
    dev = torch.device(device)
    for K, b in configs:
        f = os.path.join(OUT, f'gd_K{K}b{b}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        H, s = load_slice(K, TEST)
        lv = load_levels(b, DEV)
        model = C.load_model(K, b, lv)
        with torch.no_grad():
            y0 = torch.cat([run_model_warm(model, H[i:i + 256], s[i:i + 256], lv, PT, 16)
                            for i in range(0, H.shape[0], 256)])        # DAC grid, not normalized
        r0, r = [], []
        for i in range(0, H.shape[0], 256):
            Hc, sc, yc = H[i:i + 256].to(dev), s[i:i + 256].to(dev), y0[i:i + 256].to(dev)
            _, best_r, _ = gd_optimize(Hc, sc, lv.to(dev), nv(20.0), yc, rate='ls', **cfg)
            r.append(best_r.cpu())
            r0.append(sumrate_bussgang_ls(normalize_power(yc, PT), Hc, sc, nv(20.0)).cpu())
        r, r0 = torch.cat(r).numpy(), torch.cat(r0).numpy()
        assert (r >= r0 - 1e-4).all()
        np.savez(f, gnn_gd_20=r, gnn_20=r0, recipe=json.dumps(cfg))
        log(f'GD K{K} b{b}: GNN {r0.mean():.3f} -> GNN-GD {r.mean():.3f} ({100 * r0.mean() / r.mean():.1f}%) '
            f'({time.time() - t0:.0f} s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['independence', 'rates', 'gains', 'ber', 'phase', 'idrates', 'gd'])
    ap.add_argument('--configs', default='', help='e.g. K2b1,K6b3 (default: all 12)')
    ap.add_argument('--K', default='2,1,4,6', help='phase stage only')
    ap.add_argument('--device', default='cpu', help='gd stage only')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('THREADS', 24)))
    cfgs = CONFIGS if not a.configs else [c for c in CONFIGS if f'K{c[0]}b{c[1]}' in a.configs.split(',')]
    if a.stage == 'independence':
        stage_independence()
    elif a.stage == 'rates':
        stage_rates(cfgs)
    elif a.stage == 'gains':
        stage_gains()
    elif a.stage == 'ber':
        stage_ber(cfgs)
    elif a.stage == 'phase':
        stage_phase([int(k) for k in a.K.split(',')])
    elif a.stage == 'idrates':
        stage_idrates(cfgs)
    elif a.stage == 'gd':
        stage_gd(cfgs, a.device)


if __name__ == '__main__':
    main()
