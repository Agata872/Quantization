"""IDE / IDE2 finite-alphabet precoding baselines (Wang, Wen, Jin, Tsai, IEEE TWC 2018, Algorithms 1 and 2)
on the evaluation channels of the K x bits sweep.

Problem (6) of the paper, per symbol vector:  min_{x in X^M, beta>0} ||s - beta A x||^2 + beta^2 K sigma^2,
with A = H^T (K x M; our received signal is r = H^T y) and X = L_b + j L_b, the same per-AP Lloyd-Max grid as
every other method (1 bit: +-l). Implemented as in the paper:
  * T = 100 iterations, damping alpha = 0.95, x_d^0 = 0, gamma_d^0 = 1 (IDE);
  * adaptive beta: beta^0 = 1 and beta updated with (26), beta = Re{s^H A x}/(||A x||^2 + K sigma^2),
    every 10 iterations of x (Sec. IV), after line 6 (IDE) / line 5 (IDE2);
  * output x = x^T (the last hard iterate).
Algorithm 1, line 4 reads (H~^H H~ + gamma^t I)^-1, but only gamma_d is initialised (line 2) and gamma_d is
otherwise unused (line 9); line 4 is therefore implemented with gamma_d^t.
Normalisation of the paper: total transmit power 1, E|h|^2 = 1, sigma^2 = 1/SNR (SNR = N P_tx / sigma^2).
Our channels already have E|h|^2 = 1; the levels are divided by sqrt(P_t) so that the total power is ~1, which
keeps beta^0 = 1 meaningful. The level choice is then mapped back (scale is removed by the per-block power
normalization anyway). IDE depends on the SNR through beta, so it is run separately at every SNR (like WMMSE).

Precoding factor beta (--variants algorithm:beta_mode):
  symbol  (26) per symbol vector, exactly as in the paper, which assumes that the UE knows every beta^t. Under the
          Bussgang rate every other method is scored with (one gain per UE and block), the symbol-to-symbol
          fluctuation of 1/beta^t is distortion; for b >= 2 the rate then DEcreases as IUI converges.
  block   (26) with numerator and denominator summed over the N_s symbols of the block: one beta per channel
          realization, updated every 10 iterations. Strongest variant under our metric, but it couples the
          symbols of a block (the whole block must be available, as for GNN-GD).
  wf      beta fixed to beta_WF of eq. (7) per channel realization: causal, per-symbol processing.

Every rate: LS Bussgang estimator after per-block power normalization, test set [4096:6144], 125 symbols per
channel -- the numbers every other curve of the paper uses.

  python ide_baseline.py                       # all (K, b), full SNR grid -> exp_results/ide_baseline.json
  python ide_baseline.py --snrs 20 --configs K2b1,K6b2
  python ide_baseline.py --time                # CPU time per channel realization (K=2,6, b=2, 20 dB)
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import load_levels, PT  # noqa: E402

SNR = [float(x) for x in range(-30, 31, 5)]
NS = 125
OUT = os.path.join(CURRENT_DIR, 'exp_results', 'ide_baseline.json')
DEV = torch.device('cpu')


def data(K):
    d = glob.glob(os.path.join(CURRENT_DIR, 'datasets', 'cellfree', f'M_40_K_{K}_Ntr_200000_*'))[0]
    H = np.load(os.path.join(d, 'Htest.npy'))
    s = np.load(os.path.join(d, 'stest.npy')).reshape(K, H.shape[0], NS).transpose(1, 0, 2)
    return (torch.from_numpy(H[4096:6144].astype(np.complex64)),
            torch.from_numpy(np.ascontiguousarray(s[4096:6144]).astype(np.complex64)))


def project(x, lv):
    """Pi_X: nearest level, real and imaginary parts separately."""
    def q(v):
        return lv[(v.unsqueeze(-1) - lv).abs().argmin(-1)]
    return torch.complex(q(x.real), q(x.imag))


def beta_wf(A, sig2):
    """Precoding factor of the Wiener-filter precoder, eq. (7), with N P_tx = 1: one value per channel (C)."""
    K = A.shape[1]
    W = A.conj().transpose(1, 2) @ torch.linalg.inv(A @ A.conj().transpose(1, 2)
                                                     + K * sig2 * torch.eye(K, dtype=A.dtype))
    return (W.abs() ** 2).sum((1, 2)).sqrt()


def ide(A, S, lv, sig2, variant='ide', T=100, alpha=0.95, beta_every=10, beta_mode='symbol'):
    """A: C x K x M, S: C x N x K (one row per symbol vector), lv: real levels. Returns x: C x N x M.

    beta_mode  'symbol': (26) per symbol vector, as in the paper (the UE is assumed to know every beta^t);
               'block' : (26) with numerator and denominator summed over the N symbols of the block, so that all
                         symbols of a channel realization share one beta (a UE with one gain per block);
               'wf'    : fixed beta = beta_WF of eq. (7), one per channel realization, never updated."""
    C, K, M = A.shape
    N = S.shape[1]
    AH = A.conj().transpose(1, 2)                                    # C x M x K
    G = A @ AH                                                       # C x K x K  (A A^H)
    trG = G.diagonal(dim1=1, dim2=2).real.sum(-1)                    # tr(A^H A)
    col2 = (A.abs() ** 2).sum(1)                                     # C x M, diag(A^H A)
    eye = torch.eye(K, dtype=A.dtype)
    xd = torch.zeros(C, N, M, dtype=A.dtype)
    if beta_mode == 'wf':
        beta = beta_wf(A, sig2)[:, None].expand(C, N).clone()
    else:
        beta = torch.ones(C, N, dtype=torch.float64)
    gd = torch.ones(C, N, dtype=torch.float64)
    x = xd
    for t in range(T):
        b = beta
        e = S - b[..., None] * torch.einsum('ckm,cnm->cnk', A, xd)  # s - H~ x_d
        if variant == 'ide':
            Q = torch.linalg.inv((b ** 2)[..., None, None] * G[:, None] + gd[..., None, None] * eye)
            We = b[..., None] * torch.einsum('cmk,cnk->cnm', AH, (Q @ e.unsqueeze(-1)).squeeze(-1))
            QA = torch.einsum('cnkl,clm->cnkm', Q, A)
            dg = (b ** 2)[..., None] * torch.einsum('ckm,cnkm->cnm', A.conj(), QA).real   # diag(W H~)
            x = project(xd + We / dg, lv)                                                 # line 6
            r = S - b[..., None] * torch.einsum('ckm,cnm->cnk', A, x)
            g = (b ** 2) * trG[:, None] / (r.abs() ** 2).sum(-1).clamp_min(1e-30)        # line 7
            gd = alpha * gd + (1 - alpha) * g                                             # line 9
        elif variant == 'ide2':
            upd = torch.einsum('cmk,cnk->cnm', AH, e) / (b[..., None] * col2[:, None, :])  # W_u e
            x = project(xd + upd, lv)                                                     # line 5
        else:
            raise ValueError(variant)
        xd = alpha * xd + (1 - alpha) * x                                                 # line 8 / 6
        if beta_mode != 'wf' and (t + 1) % beta_every == 0:                               # (26)
            Ax = torch.einsum('ckm,cnm->cnk', A, x)
            num, den = (S.conj() * Ax).sum(-1).real, (Ax.abs() ** 2).sum(-1) + K * sig2
            if beta_mode == 'block':
                num, den = num.sum(1, keepdim=True).expand(C, N), den.sum(1, keepdim=True).expand(C, N)
            beta = (num / den).clamp_min(1e-6)
    return x


def precode(H, s, levels, snr_db, variant, beta_mode='symbol', chunk=128):
    """H: n x M x K, s: n x K x Ns -> power-normalized y: n x M x Ns (complex64)."""
    lv = (levels.double() / PT ** 0.5)
    sig2 = 10 ** (-snr_db / 10)
    ys = []
    for i in range(0, H.shape[0], chunk):
        A = H[i:i + chunk].transpose(1, 2).to(torch.complex128)
        S = s[i:i + chunk].transpose(1, 2).to(torch.complex128)
        x = ide(A, S, lv, sig2, variant, beta_mode=beta_mode)
        ys.append(x.transpose(1, 2).to(torch.complex64))
    return normalize_power(torch.cat(ys), PT)


def rate(y, H, s, snr_db):
    nv = PT / 10 ** (snr_db / 10)
    return torch.cat([sumrate_bussgang_ls(y[b:b + 512], H[b:b + 512], s[b:b + 512], nv)
                      for b in range(0, H.shape[0], 512)]).mean().item()


def timing(n_ch=10, reps=5, threads=16):
    torch.set_num_threads(threads)
    res = []
    for K in (2, 6):
        H, s = data(K)
        levels = load_levels(2, DEV)
        for variant in ('ide', 'ide2'):
            ts = []
            for i in range(n_ch):
                f = lambda: precode(H[i:i + 1], s[i:i + 1], levels, 20.0, variant, 'block')  # noqa: E731
                f()
                for _ in range(reps):
                    t0 = time.perf_counter()
                    f()
                    ts.append(time.perf_counter() - t0)
            res.append({'K': K, 'bits': 2, 'variant': variant, 'threads': threads,
                        'ms_per_channel': 1e3 * statistics.median(ts)})
            print(json.dumps(res[-1]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default='', help='e.g. K2b1,K6b2; default all K in {1,2,4,6}, b in {1,2,3}')
    ap.add_argument('--snrs', default='', help='comma-separated dB; default -30:30:5')
    ap.add_argument('--variants', default='ide:block,ide2:block,ide:wf,ide2:wf,ide:symbol,ide2:symbol',
                    help='algorithm:beta_mode pairs')
    ap.add_argument('--time', action='store_true')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--out', default=OUT, help='result json (use separate files for parallel workers)')
    a = ap.parse_args()
    out_path = a.out
    if a.time:
        res = timing()
        json.dump(res, open(os.path.join(CURRENT_DIR, 'exp_results', 'ide_timing.json'), 'w'), indent=1)
        return
    torch.set_num_threads(a.threads)
    snrs = [float(x) for x in a.snrs.split(',')] if a.snrs else SNR
    cfgs = [(K, b) for K in (1, 2, 4, 6) for b in (1, 2, 3)]
    if a.configs:
        want = set(a.configs.split(','))
        cfgs = [c for c in cfgs if f'K{c[0]}b{c[1]}' in want]
    variants = [tuple(v.split(':')) for v in a.variants.split(',')]
    out = json.load(open(out_path)) if os.path.exists(out_path) else {
        'snr_db': SNR, 'estimator': 'LS Bussgang, test set [4096:6144], 125 symbols per channel',
        'recipe': 'Wang et al. TWC 2018, Alg. 1/2, T=100, alpha=0.95, beta update (26) every 10 iterations',
        'results': {}}
    for K, b in cfgs:
        H, s = data(K)
        levels = load_levels(b, DEV)
        tag = f'K{K}b{b}'
        cur = out['results'].setdefault(tag, {})
        for variant, bm in variants:
            key = f'{variant}_{bm}'
            for x in snrs:
                t0 = time.perf_counter()
                r = rate(precode(H, s, levels, x, variant, bm), H, s, x)
                cur.setdefault(key, {})[f'{x:g}'] = r
                print(f'{tag} {key:12s} {x:+5.0f} dB: {r:.4f}  ({time.perf_counter() - t0:.0f} s)', flush=True)
                json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
