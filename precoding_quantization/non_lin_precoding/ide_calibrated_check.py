"""Quick check: IDE (Wang et al., TWC 2018, Alg. 1) with a calibrated fixed beta versus beta_WF and block beta.

Calibrated fixed beta: per channel realization, beta is determined once with the block update (26) of IDE on an
independent training block of 125 symbols (same distribution as the data, never the data themselves) and then
kept fixed while every data symbol vector is processed independently, as with beta_WF. It thus needs the same
information as beta_WF (the channel) and the same per-symbol processing as the GNN.
Setting: K = 2, b = 1, 2, 3, SNR 20 dB, the first --n-channels channels of the evaluation set Htest[4096:6144].
  sum rate  - LS Bussgang estimator with the Gaussian test symbols (ide_baseline.rate, as the rate figures);
  BER       - uncoded Gray 16-QAM, exact in the AWGN, symbols of ber_16qam_20dB.py (as Fig. 6).
References on the same channels and symbols: GNN (argmax, continued runs) and WMMSE + DAC (per-AP
normalization, nearest DAC level).

  python ide_calibrated_check.py           # -> exp_results/ide_calibrated_check.json
"""
import argparse
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

import ber_16qam_20dB as ber16  # noqa: E402
import ide_baseline  # noqa: E402
from exp_cellfree_ablation import normalize_power  # noqa: E402
from train_sweep import PT, load_levels, quantize_to_levels  # noqa: E402
from wmmse_baseline import best_wmmse  # noqa: E402

K = 2
SNR_DB = 20.0
DEV = torch.device('cpu')


def ide_run(H, S, levels, snr_db, beta_mode, beta_fixed=None, chunk=128):
    """As ide_baseline.precode(), plus the final beta per channel. H: n x M x K, S: n x K x Ns."""
    lv = levels.double() / PT ** 0.5
    sig2 = 10 ** (-snr_db / 10)
    ys, betas = [], []
    for i in range(0, H.shape[0], chunk):
        A = H[i:i + chunk].transpose(1, 2).to(torch.complex128)
        Sc = S[i:i + chunk].transpose(1, 2).to(torch.complex128)
        bf = None if beta_fixed is None else beta_fixed[i:i + chunk]
        x, beta = ide_baseline.ide(A, Sc, lv, sig2, 'ide', beta_mode=beta_mode, beta_fixed=bf, return_beta=True)
        ys.append(x.transpose(1, 2).to(torch.complex64))
        betas.append(beta[:, 0])
    return normalize_power(torch.cat(ys), PT), torch.cat(betas)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n-channels', type=int, default=512)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--seed-train', type=int, default=777)
    ap.add_argument('--out', default=os.path.join(CURRENT_DIR, 'exp_results', 'ide_calibrated_check.json'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    n = args.n_channels
    H_all, s_all = ide_baseline.data(K)
    H, s_gauss = H_all[:n], s_all[:n]
    s_qam = ber16.qam16(n, K, 1234)            # identical to the Fig. 6 symbols on these channels
    g = torch.Generator().manual_seed(args.seed_train)
    normal = torch.randn((n, K, ide_baseline.NS, 2), generator=g) / math.sqrt(2)
    train_gauss = torch.complex(normal[..., 0], normal[..., 1])
    train_qam = ber16.qam16(n, K, args.seed_train)
    nv = PT / 10 ** (SNR_DB / 10)
    V = best_wmmse(H, nv)
    row_norm = torch.linalg.vector_norm(V, dim=2, keepdim=True).clamp_min(1e-12)

    # backward compatibility of ide(): the new code path equals ide_baseline.precode() bit for bit
    lv1 = load_levels(1, DEV)
    reference = ide_baseline.precode(H[:32], s_qam[:32], lv1, SNR_DB, 'ide', 'block')
    assert torch.equal(ide_run(H[:32], s_qam[:32], lv1, SNR_DB, 'block')[0], reference)

    results = {'n_channels': n, 'snr_db': SNR_DB, 'k': K, 'results': {}}
    for b in (1, 2, 3):
        levels = load_levels(b, DEV)
        model = ber16.load_gnn(K, b, levels)
        entry = {}
        for name, S, train, metric in (('sum_rate', s_gauss, train_gauss,
                                        lambda y, S=s_gauss: ide_baseline.rate(y, H, S, SNR_DB)),
                                       ('ber_16qam', s_qam, train_qam, lambda y, S=s_qam: ber16.ber(y, H, S))):
            t0 = time.perf_counter()
            y_wf, beta_wf = ide_run(H, S, levels, SNR_DB, 'wf')
            y_blk, beta_blk = ide_run(H, S, levels, SNR_DB, 'block')
            _, beta_cal = ide_run(H, train, levels, SNR_DB, 'block')           # calibration on training symbols
            y_cal, _ = ide_run(H, S, levels, SNR_DB, 'fixed', beta_fixed=beta_cal)
            with torch.no_grad():
                y_gnn = ber16.gnn_precode(model, H, S, levels)
                y_wmmse = normalize_power(quantize_to_levels((V @ S) / row_norm, levels), PT)
            entry[name] = {'ide_wf': metric(y_wf), 'ide_block': metric(y_blk), 'ide_cal': metric(y_cal),
                           'gnn': metric(y_gnn), 'wmmse_dac': metric(y_wmmse)}
            entry[name + '_beta'] = {
                'median_block_over_wf': float((beta_blk / beta_wf).median()),
                'median_cal_over_block': float((beta_cal / beta_blk).median()),
                'p10_p90_cal_over_block': [float((beta_cal / beta_blk).quantile(q)) for q in (0.1, 0.9)]}
            fmt = '{:.4f}' if name == 'sum_rate' else '{:.3e}'
            print(f'b={b} {name:9s}: ' + '  '.join(f'{k} ' + fmt.format(v) for k, v in entry[name].items())
                  + f'  | beta block/WF {entry[name + "_beta"]["median_block_over_wf"]:.2f}, '
                  f'cal/block {entry[name + "_beta"]["median_cal_over_block"]:.3f}  ({time.perf_counter() - t0:.0f}s)',
                  flush=True)
        results['results'][f'b{b}'] = entry
        json.dump(results, open(args.out, 'w'), indent=1)
    print('Completed:', args.out)


if __name__ == '__main__':
    main()
