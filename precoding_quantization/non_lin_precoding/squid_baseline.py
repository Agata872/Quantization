"""SQUID baseline (squared-infinity-norm relaxation with Douglas-Rachford splitting; Jacobsson, Durisi, Coldrey,
Goldstein, Studer, "Quantized precoding for massive MU-MIMO", IEEE TCOM 2017, Sec. IV-B, eqs. (48)-(56), Alg. 1)
on the evaluation channels of the K x bits sweep. 1-bit DACs only: the relaxation relies on |x_m| being constant,
and the authors' reference code refuses L > 2 ("SQUID: only 1 bit (L=2) supported!").

Port of the authors' reference implementation (github.com/quantizedmassivemimo/1bit_precoding, precoders/SQUID.m
and tools/prox_infinity_norm_squared.m, v1.3), batched over channels and symbol vectors:
  real-valued model H_R = [Re H, -Im H; Im H, Re H] (2K x 2M, H = our H^T), s_R = [Re s; Im s];
  b = c = 0; for t = 1..iter:  z = 2b - c;  a = prox_g(z);  b = prox_{lambda ||.||_inf^2}(c + a - b);
                               c = c + rho (a - b),   lambda = 2 K M N0;
  x = sign(b) / sqrt(2M); beta = Re{(Hx)^H s}/(||Hx||^2 + K N0); x -> -x if beta < 0.
Reference-code defaults: iter = 50, rho = 1, gain = 1, where the code notes that gain "must be optimized"
(1 for large systems or low SNR, smaller for small systems or high SNR). gain scales only prox_g, so it also
changes the effective l_inf^2 weight to lambda/gain. With gain = 1, the iteration diverges on our cell-free
channels at 20 dB (relaxed objective > 1e100 after 1000 iterations, K = 2 and 6); with gain = 0.01 it converges
and the rate increases monotonically with the iterations. We therefore report
  squid_default  gain = 1
  squid_tuned    gain chosen per (K, SNR) from GAINS on the disjoint tuning channels Htest[0:128]
Normalization of the reference code: E|h|^2 = 1, total transmit power 1, N0 = 10^(-SNR/10) -- identical to ours
after dividing the power by P_t, so no rescaling is needed. sign(0) (only if the prox returns exactly 0) is mapped
to +1 to stay on the 1-bit grid.

Every rate: LS Bussgang estimator after per-block power normalization, test set [4096:6144], 125 symbols per
channel -- the numbers every other curve of the paper uses.

  python squid_baseline.py            # K in {1,2,4,6}, b = 1, -30:30:5 dB -> exp_results/squid_baseline.json
  python squid_baseline.py --time     # CPU time per channel realization (K = 2, 6; 20 dB; 16 threads)
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
from train_sweep import PT  # noqa: E402

SNR = [float(x) for x in range(-30, 31, 5)]
NS = 125
EVAL, TUNE = slice(4096, 6144), slice(0, 128)
GAINS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
OUT = os.path.join(CURRENT_DIR, 'exp_results', 'squid_baseline.json')


def data(K, sl):
    d = glob.glob(os.path.join(CURRENT_DIR, 'datasets', 'cellfree', f'M_40_K_{K}_Ntr_200000_*'))[0]
    H = np.load(os.path.join(d, 'Htest.npy'))
    s = np.load(os.path.join(d, 'stest.npy')).reshape(K, H.shape[0], NS).transpose(1, 0, 2)
    return (torch.from_numpy(H[sl].astype(np.complex64)),
            torch.from_numpy(np.ascontiguousarray(s[sl]).astype(np.complex64)))


def prox_inf2(w, lam):
    """prox of lam*||u||_inf^2 (Algorithm 1 / prox_infinity_norm_squared.m), along the last dimension."""
    wabs = w.abs()
    n = w.shape[-1]
    srt = wabs.sort(-1, descending=True).values
    ws = srt.cumsum(-1) / (2 * lam + torch.arange(1, n + 1, dtype=w.dtype))
    alpha = ws.max(-1, keepdim=True).values.clamp_min(0)
    return torch.minimum(wabs, alpha) * torch.sign(w)


def squid(H, s, N0, gain=1.0, rho=1.0, iters=50):
    """H: C x M x K (ours, r = H^T y), s: C x K x N  ->  x: C x M x N on {+-1 +- j}/sqrt(2M)."""
    C, M, K = H.shape
    A = H.transpose(1, 2).to(torch.complex128)                       # C x K x M, the paper's H
    HR = torch.cat([torch.cat([A.real, -A.imag], 2), torch.cat([A.imag, A.real], 2)], 1)   # C x 2K x 2M
    HRt = HR.transpose(1, 2)
    Q = HRt @ torch.linalg.inv((0.5 / gain) * torch.eye(2 * K, dtype=torch.float64) + HR @ HRt)  # C x 2M x 2K
    S = s.to(torch.complex128)
    sR = torch.cat([S.real, S.imag], 1)                              # C x 2K x N
    sMF = HRt @ sR                                                   # C x 2M x N
    sREG = (2 * gain) * (sMF - Q @ (HR @ sMF))
    lam = 2 * K * M * N0
    b = torch.zeros_like(sMF)
    c = torch.zeros_like(sMF)
    for _ in range(iters):
        z = 2 * b - c
        a = sREG + z - Q @ (HR @ z)
        b = prox_inf2((c + a - b).transpose(1, 2), lam).transpose(1, 2)
        c = c + rho * (a - b)
    sg = torch.sign(b)
    sg[sg == 0] = 1.0
    x = torch.complex(sg[:, :M], sg[:, M:]) / (2 * M) ** 0.5         # C x M x N
    Hx = torch.einsum('ckm,cmn->ckn', A, x)
    beta = (Hx.conj() * S).sum(1).real / ((Hx.abs() ** 2).sum(1) + K * N0)
    x = torch.where((beta < 0)[:, None, :], -x, x)
    return x.to(torch.complex64)


def run(H, s, snr_db, gain, chunk=256):
    N0 = 10 ** (-snr_db / 10)
    y = torch.cat([squid(H[i:i + chunk], s[i:i + chunk], N0, gain) for i in range(0, H.shape[0], chunk)])
    return normalize_power(y, PT)


def rate(y, H, s, snr_db):
    nv = PT / 10 ** (snr_db / 10)
    return torch.cat([sumrate_bussgang_ls(y[b:b + 512], H[b:b + 512], s[b:b + 512], nv)
                      for b in range(0, H.shape[0], 512)]).mean().item()


def timing(n_ch=10, reps=5, threads=16):
    torch.set_num_threads(threads)
    res = []
    for K in (2, 6):
        H, s = data(K, EVAL)
        ts = []
        for i in range(n_ch):
            f = lambda: run(H[i:i + 1], s[i:i + 1], 20.0, 1.0)  # noqa: E731
            f()
            for _ in range(reps):
                t0 = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t0)
        res.append({'K': K, 'bits': 1, 'threads': threads, 'ms_per_channel': 1e3 * statistics.median(ts)})
        print(json.dumps(res[-1]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ks', default='1,2,4,6')
    ap.add_argument('--snrs', default='')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--time', action='store_true')
    a = ap.parse_args()
    if a.time:
        json.dump(timing(), open(os.path.join(CURRENT_DIR, 'exp_results', 'squid_timing.json'), 'w'), indent=1)
        return
    torch.set_num_threads(a.threads)
    snrs = [float(x) for x in a.snrs.split(',')] if a.snrs else SNR
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        'snr_db': SNR, 'bits': 1, 'estimator': 'LS Bussgang, test set [4096:6144], 125 symbols per channel',
        'recipe': 'Jacobsson et al. TCOM 2017 / reference code v1.3: iter=50, rho=1; gain=1 (default) or tuned '
                  f'per (K, SNR) over {GAINS} on Htest[0:128]', 'results': {}}
    for K in [int(k) for k in a.Ks.split(',')]:
        H, s = data(K, EVAL)
        Ht, st = data(K, TUNE)
        cur = out['results'].setdefault(f'K{K}b1', {'squid_default': {}, 'squid_tuned': {}, 'tuned_gain': {}})
        for x in snrs:
            t0 = time.perf_counter()
            tune = {g: rate(run(Ht, st, x, g), Ht, st, x) for g in GAINS}
            g_best = max(tune, key=tune.get)
            r_def = rate(run(H, s, x, 1.0), H, s, x)
            r_tun = r_def if g_best == 1.0 else rate(run(H, s, x, g_best), H, s, x)
            cur['squid_default'][f'{x:g}'], cur['squid_tuned'][f'{x:g}'] = r_def, r_tun
            cur['tuned_gain'][f'{x:g}'] = g_best
            print(f'K{K} b1 {x:+5.0f} dB: default {r_def:.4f} | tuned {r_tun:.4f} (gain {g_best:g}; tuning '
                  + ' '.join(f'{g:g}:{v:.3f}' for g, v in tune.items()) + f')  ({time.perf_counter() - t0:.0f} s)',
                  flush=True)
            json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
