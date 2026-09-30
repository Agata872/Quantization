"""Uncoded BER with Gray-mapped 16-QAM at SNR = 20 dB: GNN vs. IDE with a calibrated fixed beta and with block beta
(Wang et al., TWC 2018, Alg. 1) for b = 1, 2, 3, and SQUID (Jacobsson et al., TCOM 2017) for b = 1. M = 40,
K in {1, 2, 4, 6}.

Self-contained (the former QPSK companion script, from which the shared settings below came, was removed).
Symbols: Gray 16-QAM, per-dimension levels {-3, -1, 1, 3}/sqrt(10) (unit average energy), i.i.d. (seed 1234 for
the 2048 evaluation channels Htest[4096:6144], seed 4321 for the SQUID tuning channels Htest[0:128]), 125 per
channel realization, the same for every method; the GNN (trained with Gaussian symbols) is applied unchanged.
Precoders: GNN (best checkpoint of the continued runs, argmax), IDE_WF = ide_baseline.precode(..., 'ide', 'wf')
(T = 100, alpha = 0.95, beta = beta_WF of eq. (7)), IDE_block = ide_baseline.precode(..., 'ide', 'block') (beta of
(26) summed over the block, one beta per channel realization, updated every 10 iterations), SQUID (50 iterations,
rho = 1) with gain selected from GAINS by the 16-QAM BER on the tuning channels; ZF/MRT + DAC is stored as a
reference only. IDE_cal: per channel realization, beta = ide_baseline.calibrated_beta(...), the value of the block
update on an independent training block of 125 16-QAM symbol vectors (seed SEED_CAL, never the data), then
ide_baseline.precode(..., 'ide', 'fixed', beta_fixed=beta): every symbol vector is processed independently with a
fixed beta, as for IDE_WF, but without its mis-scaling on the DAC grid; the ratio to beta_WF is stored.
The figure of the paper shows GNN, IDE_cal, IDE_block and SQUID (IDE_WF, the earlier version, is kept in the json).
Receiver: UE k divides by its effective gain per block, g_k = sum_t r_k[t] s_k[t]^* / sum_t |s_k[t]|^2 (noiseless
r = H^T y), and detects each dimension with the PAM-4 thresholds {0, +-2/sqrt(10)}. Per dimension, bit 1 (sign)
errs with Q(sign(a) u / sd) and bit 2 (inner/outer) with Q((d - u)/sd) + Q((d + u)/sd) if an inner level a was
sent and Q((-d - u)/sd) - Q((d - u)/sd) if an outer one was sent (u: noiseless equalized value, d = 2/sqrt(10),
sd^2 = sigma^2 / (2 |g_k|^2)); checked against Monte Carlo.

  python ber_16qam_20dB.py     # -> exp_results/ber_16qam_20dB.json (resumes from an existing file)
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import GNNv2, normalize_power  # noqa: E402
from train_sweep import load_levels, run_model_warm, linear_quantized, PT, M_FULL  # noqa: E402
import ide_baseline  # noqa: E402
import squid_baseline  # noqa: E402

SNR_DB = 20.0
NV = PT / 10 ** (SNR_DB / 10)
NS = 125
EVAL, TUNE = slice(4096, 6144), slice(0, 128)
DEV = torch.device('cpu')
# the BER-optimal SQUID gain at 20 dB can lie below the rate-tuning grid, which is therefore extended down to 1e-6
GAINS = squid_baseline.GAINS + (3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
OUT = os.path.join(CURRENT_DIR, 'exp_results', 'ber_16qam_20dB.json')
A = 1 / math.sqrt(10)


def channels(K, sl):
    d = glob.glob(os.path.join(CURRENT_DIR, 'datasets', 'cellfree', f'M_40_K_{K}_Ntr_200000_*'))[0]
    return torch.from_numpy(np.load(os.path.join(d, 'Htest.npy'))[sl].astype(np.complex64))


def load_gnn(K, b, levels):
    pat = ('stored_models_cellfree_k1_continue/*/model_best.pt' if (K, b) == (1, 1)
           else f'stored_models_cellfree_sweep_continue/M40_K{K}_b{b}_*/model_best.pt')
    ck = torch.load(glob.glob(os.path.join(CURRENT_DIR, pat))[0], map_location=DEV, weights_only=False)
    m = GNNv2(M_FULL, K, 128, 4, b, 0.25, levels, input_mode='polar4', feat_stats=ck['feat_stats'])
    m.load_state_dict(ck['model'])
    m.output_type = 'argmax'
    m.eval()
    return m


@torch.no_grad()
def gnn_precode(model, H, s, levels, chunk=256):
    return torch.cat([normalize_power(run_model_warm(model, H[i:i + chunk], s[i:i + chunk], levels, PT, 16), PT)
                      for i in range(0, H.shape[0], chunk)])


SEED_CAL = 777          # training symbols for the calibrated beta of IDE (independent of the data, seed 1234)
CAL_STATS = {}


def ide_calibrated(K, b, H, s, lv):
    beta = ide_baseline.calibrated_beta(H, qam16(H.shape[0], K, SEED_CAL), lv, SNR_DB)
    ratio = beta / ide_baseline.beta_wf_values(H, SNR_DB)
    CAL_STATS[K, b] = {'median_over_beta_wf': float(ratio.median()),
                       'p10_p90_over_beta_wf': [float(ratio.quantile(q)) for q in (0.1, 0.9)]}
    return ide_baseline.precode(H, s, lv, SNR_DB, 'ide', 'fixed', beta_fixed=beta)


METHODS = {
    'gnn': lambda K, b, H, s, lv: gnn_precode(load_gnn(K, b, lv), H, s, lv),
    'ide_wf': lambda K, b, H, s, lv: ide_baseline.precode(H, s, lv, SNR_DB, 'ide', 'wf'),
    'ide_block': lambda K, b, H, s, lv: ide_baseline.precode(H, s, lv, SNR_DB, 'ide', 'block'),
    'ide_cal': ide_calibrated,
    'zf_dac': lambda K, b, H, s, lv: normalize_power(linear_quantized(H, s, lv, PT), PT),
}


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
                if m == 'ide_cal':
                    res['ide_cal_beta'] = CAL_STATS.pop((K, b))
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
                                              for k, v in res.items() if not isinstance(v, dict))
                  + f'  ({time.perf_counter() - t0:.0f} s)', flush=True)
            json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
