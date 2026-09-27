"""Unquantized and quantized WMMSE baselines on the evaluation channels of the K x bits sweep.

WMMSE: sum-rate WMMSE (Shi et al., IEEE TSP 2011) per channel and per SNR, total power Pt, in a K x K
form (Woodbury: the M x M matrix A = Ht D Ht^H has rank K); per channel the best of three starts
(RZF, MRT, ZF) by closed-form sum rate. It is the sum-rate-optimized linear precoder, i.e. the proper
unquantized linear reference: unquantized ZF equalizes the users' gains, which in cell-free (strongest /
weakest user ||h_k||^2 ~18x, median) wastes power, most of all at low SNR.

Quantized WMMSE: x = V s through the same b-bit DAC as the other baselines (Lloyd-Max levels, 1 bit:
+-sqrt(Pt/2M)), in two conventions:
  equal   per-AP input normalization x_m/||v_m||, nearest level, NOT de-normalized: every AP transmits
          the same power. This is linear_quantized (the "ZF + b-bit DAC" baseline) and the constraint
          the GNN works under -- its outputs are DAC levels shared by all APs.
  denorm  the same, then y_m = ||v_m|| Q(x_m/||v_m||): the amplification stage restores the precoder's
          per-AP power (Feys et al., JSTSP 2025, Sec. II-B3). A degree of freedom the GNN does not have;
          reported for ZF/MRT as well ('lin_q_denorm').
All rates: the LS Bussgang estimator on the 125-symbol blocks after per-block power normalization,
test set [4096:6144] -- the numbers every other curve uses. K=1: WMMSE reduces to MRT.

Writes <run>/wmmse_curves.json into every run folder of stored_models_cellfree_sweep_K_bits (the
baselines do not depend on the GNN; the continued runs use the same channels) and
figures_ls/wmmse_summary.json.

  CUDA_VISIBLE_DEVICES=0 python wmmse_baseline.py
"""
import glob
import json
import os
import sys
import time

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import load_data, load_levels, linear_precoder, quantize_to_levels, EVAL, PT  # noqa: E402
from reeval_sweep import SNR, LS_NOTE, nv, curve  # noqa: E402

MARK = 'wmmse_baseline.py'
SWEEP_DIR = os.path.join(CURRENT_DIR, 'stored_models_cellfree_sweep_K_bits')


def closed_rate(V, H, noise_var):
    """Exact expected sum rate of the linear precoder V (r_k = h_k^T x): bs."""
    HV = H.transpose(1, 2) @ V
    sig = HV.diagonal(dim1=1, dim2=2).abs() ** 2
    return torch.log2(1 + sig / ((HV.abs() ** 2).sum(-1) - sig + noise_var)).sum(-1)


def wmmse(H, noise_var, V0, iters=150):
    """Sum-rate WMMSE. V = (A + mu I)^-1 Ht diag(u w) = Ht (Q + mu D^-1)^-1 D^-1 diag(u w), Q = Ht^H Ht,
    D = diag(w |u|^2). With Qt = D^1/2 Q D^1/2 = U L U^H and Z = U^H D^-1/2 diag(u w):
    ||V||_F^2 = sum_i L_i ||Z_i||^2 / (L_i + mu)^2, bisected for sum ||v||^2 = Pt. Float64: Q is
    ill-conditioned when users share their dominant APs."""
    dev = H.device
    H, V0 = H.to(torch.complex128), V0.to(torch.complex128)
    Ht = H.conj()                                          # columns h~_k, r_k = h~_k^H x
    bs = H.shape[0]
    Q = Ht.conj().transpose(1, 2) @ Ht
    V = V0 * (PT / (V0.abs() ** 2).sum((1, 2), keepdim=True)).sqrt()
    for _ in range(iters):
        HV = Ht.conj().transpose(1, 2) @ V
        den = (HV.abs() ** 2).sum(-1) + noise_var
        d = HV.diagonal(dim1=1, dim2=2)
        u = d / den                                        # MMSE receivers
        w = 1 / (1 - (u.conj() * d).real).clamp_min(1e-12)  # MSE weights
        sq = (w * u.abs() ** 2).clamp_min(1e-30).sqrt()
        lam, U = torch.linalg.eigh(sq[:, :, None] * Q * sq[:, None, :])
        lam = lam.clamp_min(0)
        Z = U.conj().transpose(1, 2) @ torch.diag_embed((u * w) / sq)
        z2 = (Z.abs() ** 2).sum(-1)
        lo = torch.zeros(bs, device=dev, dtype=torch.float64)
        hi = torch.full((bs,), 1e8, device=dev, dtype=torch.float64)
        for _ in range(80):
            mu = (lo + hi) / 2
            big = (lam * z2 / (lam + mu[:, None]) ** 2).sum(-1) > PT
            lo, hi = torch.where(big, mu, lo), torch.where(big, hi, mu)
        V = Ht @ (sq[:, :, None] * (U @ (Z / (lam + hi[:, None])[:, :, None])))
    return V.to(torch.complex64)


def best_wmmse(H, noise_var):
    K = H.shape[2]
    Hc = H.conj()
    starts = [Hc @ torch.linalg.inv(H.transpose(1, 2) @ Hc + (K * noise_var / PT) * torch.eye(K, device=H.device)),
              Hc, linear_precoder(H, PT)]
    best, bestV = None, None
    for V0 in starts:
        V = wmmse(H, noise_var, V0)
        r = closed_rate(V, H, noise_var)
        if best is None:
            best, bestV = r, V
        else:
            better = r > best
            best, bestV[better] = torch.where(better, r, best), V[better]
    return bestV


def rate_at(y, H, s, x):
    """Mean LS sum rate at one SNR (for the SNR-dependent WMMSE precoders)."""
    return torch.cat([sumrate_bussgang_ls(y[b:b + 512], H[b:b + 512], s[b:b + 512], nv(x))
                      for b in range(0, H.shape[0], 512)]).mean().item()


def quantized(V, s, levels, denorm):
    x = V @ s
    rn = torch.linalg.vector_norm(V, dim=2, keepdim=True).clamp_min(1e-12)   # ||v_m||
    y = quantize_to_levels(x / rn, levels)
    return normalize_power(y * rn if denorm else y, PT)


def run_folders(K, bits):
    ds = glob.glob(os.path.join(SWEEP_DIR, f'M40_K{K}_b{bits}_ep15_*'))
    if K == 1 and bits == 1:
        ds = glob.glob(os.path.join(SWEEP_DIR, 'M_40_K_1_1bit_*'))
    return ds


def main():
    dev = torch.device('cuda')
    summary = {'snr_db': SNR, 'estimator': LS_NOTE, 'configs': []}
    for K in (1, 2, 4, 6):
        t0 = time.time()
        _, _, Hte, Ste = load_data(K, 1000, dev)
        H, s = Hte[EVAL], Ste[EVAL]
        W_lin = linear_precoder(H, PT)                     # MRT for K=1, ZF otherwise
        Vs = [best_wmmse(H, nv(x)) for x in SNR]           # SNR-dependent
        unq = [rate_at(normalize_power(V @ s, PT), H, s, x) for V, x in zip(Vs, SNR)]
        unq_closed = [closed_rate(V, H, nv(x)).mean().item() for V, x in zip(Vs, SNR)]
        print(f'K={K}: WMMSE done ({time.time() - t0:.0f}s); unquantized @20dB {unq[SNR.index(20.0)]:.3f}', flush=True)
        for bits in (1, 2, 3):
            levels = load_levels(bits, dev)
            res = {'K': K, 'bits': bits, 'snr_db': SNR, 'estimator': LS_NOTE, 'written_by': MARK,
                   'wmmse_unquantized': unq, 'wmmse_unquantized_closed_form': unq_closed,
                   'wmmse_quantized_equal': [rate_at(quantized(V, s, levels, False), H, s, x) for V, x in zip(Vs, SNR)],
                   'wmmse_quantized_denorm': [rate_at(quantized(V, s, levels, True), H, s, x) for V, x in zip(Vs, SNR)],
                   'lin_q_equal': curve(quantized(W_lin, s, levels, False), H, s),
                   'lin_q_denorm': curve(quantized(W_lin, s, levels, True), H, s),
                   'linear_precoder': 'MRT' if K == 1 else 'ZF',
                   'conventions': {'equal': 'per-AP normalization, no de-normalization: every AP transmits the '
                                            'same power (= linear_quantized, the GNN constraint)',
                                   'denorm': 'de-normalized y_m = ||v_m|| Q(x_m/||v_m||) (Feys et al. JSTSP 2025, '
                                             'Sec. II-B3)'}}
            folders = run_folders(K, bits)
            for d in folders:                              # consistency: 'equal' ZF/MRT == the stored baseline
                c = json.load(open(os.path.join(d, 'snr_curves.json')))
                stored = c.get('linear_quantized', c.get('mrt_1bit'))
                res['check_lin_q_vs_stored_max_abs_diff'] = max(abs(a - b) for a, b in zip(res['lin_q_equal'], stored))
                json.dump(res, open(os.path.join(d, 'wmmse_curves.json'), 'w'), indent=1)
            i20 = SNR.index(20.0)
            print(f'  K={K} b={bits} @20dB: WMMSE unq {unq[i20]:.3f} | WMMSE+DAC equal {res["wmmse_quantized_equal"][i20]:.3f} '
                  f'denorm {res["wmmse_quantized_denorm"][i20]:.3f} | {res["linear_precoder"]}+DAC equal '
                  f'{res["lin_q_equal"][i20]:.3f} denorm {res["lin_q_denorm"][i20]:.3f} | '
                  f'check vs stored {res.get("check_lin_q_vs_stored_max_abs_diff", float("nan")):.1e} -> {len(folders)} folder(s)',
                  flush=True)
            summary['configs'].append({k: res[k] for k in ('K', 'bits', 'wmmse_unquantized', 'wmmse_quantized_equal',
                                                             'wmmse_quantized_denorm', 'lin_q_equal', 'lin_q_denorm')})
        del H, s, Hte, Ste, Vs
        torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(SWEEP_DIR, 'figures_ls', 'wmmse_summary.json'), 'w'), indent=1)
    print('wrote figures_ls/wmmse_summary.json', flush=True)


if __name__ == '__main__':
    main()
