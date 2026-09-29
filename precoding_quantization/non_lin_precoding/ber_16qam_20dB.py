"""Uncoded BER with Gray-mapped 16-QAM at SNR = 20 dB: GNN vs. IDE with beta_WF (Wang et al., TWC 2018, Alg. 1)
for b = 1, 2, 3, and SQUID (Jacobsson et al., TCOM 2017) for b = 1. M = 40, K in {1, 2, 4, 6}.

Identical to ber_qpsk_20dB.py except for the constellation:
Symbols: Gray 16-QAM, per-dimension levels {-3, -1, 1, 3}/sqrt(10) (unit average energy), i.i.d. (seed 1234 for
the 2048 evaluation channels Htest[4096:6144], seed 4321 for the SQUID tuning channels Htest[0:128]), 125 per
channel realization, the same for every method; the GNN (trained with Gaussian symbols) is applied unchanged.
Precoders: GNN (argmax), IDE_WF = ide_baseline.precode(..., 'ide', 'wf'), IDE_block = ide_baseline.precode(...,
'ide', 'block') (beta of (26) summed over the block, one beta per channel realization, updated every 10
iterations; the figure of the paper uses this variant), SQUID with gain selected from ber_qpsk_20dB.GAINS by the
16-QAM BER on the tuning channels; ZF/MRT + DAC is stored as a reference only.
Receiver: UE k divides by its effective gain per block, g_k = sum_t r_k[t] s_k[t]^* / sum_t |s_k[t]|^2 (noiseless
r = H^T y), and detects each dimension with the PAM-4 thresholds {0, +-2/sqrt(10)}. Per dimension, bit 1 (sign)
errs with Q(sign(a) u / sd) and bit 2 (inner/outer) with Q((d - u)/sd) + Q((d + u)/sd) if an inner level a was
sent and Q((-d - u)/sd) - Q((d - u)/sd) if an outer one was sent (u: noiseless equalized value, d = 2/sqrt(10),
sd^2 = sigma^2 / (2 |g_k|^2)); checked against Monte Carlo.

  python ber_16qam_20dB.py     # -> exp_results/ber_16qam_20dB.json (resumes from an existing file)
"""
import json
import math
import os
import sys
import time

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import normalize_power  # noqa: E402
from train_sweep import load_levels, linear_quantized, PT  # noqa: E402
import ide_baseline  # noqa: E402
import squid_baseline  # noqa: E402
from ber_qpsk_20dB import (SNR_DB, NV, NS, EVAL, TUNE, DEV, GAINS, METHODS, channels,  # noqa: E402
                           gnn_precode)

OUT = os.path.join(CURRENT_DIR, 'exp_results', 'ber_16qam_20dB.json')
A = 1 / math.sqrt(10)


def qam16(n, K, seed):
    g = torch.Generator().manual_seed(seed)
    lv = torch.tensor([-3., -1., 1., 3.]) * A
    i = torch.randint(0, 4, (n, K, NS, 2), generator=g)
    return torch.complex(lv[i[..., 0]], lv[i[..., 1]])


def _q(x):
    return 0.5 * torch.erfc(x / math.sqrt(2))


def _pe_dim(u, a, sd):
    """Mean error probability of the two Gray bits of one PAM-4 dimension."""
    d = 2 * A
    p_sign = _q(torch.sign(a) * u / sd)
    p_amp = torch.where(a.abs() < d, _q((d - u) / sd) + _q((d + u) / sd), _q((-d - u) / sd) - _q((d - u) / sd))
    return 0.5 * (p_sign + p_amp)


def ber(y, H, s, nv=NV):
    """Exact-in-noise uncoded 16-QAM BER; y: n x M x Ns (power-normalized), H: n x M x K, s: n x K x Ns."""
    r = torch.einsum('nmk,nmt->nkt', H, y)                              # noiseless received signal
    g = (r * s.conj()).sum(-1) / (s.abs() ** 2).sum(-1)                 # effective gain per UE and block
    z = r / g[..., None]
    sd = torch.sqrt(nv / (2 * g.abs() ** 2))[..., None]
    return (0.5 * (_pe_dim(z.real, s.real, sd) + _pe_dim(z.imag, s.imag, sd))).mean().item()


def main():
    torch.set_num_threads(int(os.environ.get('THREADS', 16)))
    out = {'snr_db': SNR_DB, 'modulation': '16-QAM (Gray)', 'symbols_seed': 1234, 'n_channels': 2048,
           'receiver': 'per-block effective-gain equalization, PAM-4 thresholds per dimension; BER exact in the AWGN',
           'results': {}}
    if os.path.exists(OUT):
        out['results'] = json.load(open(OUT))['results']
    for K in (1, 2, 4, 6):
        H, Ht = channels(K, EVAL), channels(K, TUNE)
        s, st = qam16(H.shape[0], K, 1234), qam16(Ht.shape[0], K, 4321)
        for b in (1, 2, 3):
            res = out['results'].get(f'K{K}b{b}', {})              # resume: compute only the missing methods
            todo = {m: f for m, f in METHODS.items() if m not in res}
            if not todo and (b > 1 or 'squid' in res):
                continue
            t0 = time.perf_counter()
            levels = load_levels(b, DEV)
            for m, f in todo.items():
                res[m] = ber(f(K, b, H, s, levels), H, s)
            if b == 1 and 'squid' not in res:
                N0 = 10 ** (-SNR_DB / 10)
                tune = {gn: ber(normalize_power(squid_baseline.squid(Ht, st, N0, gn), PT), Ht, st) for gn in GAINS}
                g_best = min(tune, key=tune.get)
                y = torch.cat([squid_baseline.squid(H[i:i + 256], s[i:i + 256], N0, g_best)
                               for i in range(0, H.shape[0], 256)])
                res.update(squid=ber(normalize_power(y, PT), H, s), squid_gain=g_best,
                           squid_tuning={f'{k:g}': v for k, v in tune.items()})
            out['results'][f'K{K}b{b}'] = res
            print(f'K{K} b{b}: ' + ' | '.join(f'{k} {v:.3e}' if k != 'squid_gain' else f'{k} {v}'
                                              for k, v in res.items() if k != 'squid_tuning')
                  + f'  ({time.perf_counter() - t0:.0f} s)', flush=True)
            json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
