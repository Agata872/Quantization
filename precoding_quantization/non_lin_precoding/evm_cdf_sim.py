"""Simulated EVM CDFs for the paper's EVM figure (fig:tp_evm_cdf): GNN, WMMSE + DAC and IDE with a calibrated beta
for b = 1, 2, 3 DAC bits and unquantized WMMSE as a reference, M = 40, K = 2, including the start-up
transient of the receiver.

Link: the one of the 16-QAM constellation figure, i.e. the functions of constellation_16qam.py (Gray 16-QAM,
transmit SNR 20 dB, per-block power normalization, r = H^T x + AWGN, oracle block gain). Unquantized WMMSE
transmits x = V s (the WMMSE precoder of wmmse_baseline.best_wmmse, no DAC) through the same chain. IDE uses a
calibrated fixed beta as in ber_16qam_20dB.py: one beta per channel realization from ide_baseline.calibrated_beta
on an independent training block of 125 16-QAM symbol vectors (seed --seed-cal, never the data), shared by all
blocks of the capture; every symbol vector is then processed independently.
Channels: the first 20 evaluation channels Htest[4096:4116], fixed before simulation. Each channel is one
capture of 320 blocks of 125 symbols per UE (40,000 symbols), as if the receiver were started for every
measurement. Symbols and noise are shared by all methods and resolutions.
Receiver start-up: after it is started, a real receiver needs time until its gain (AGC, CMA equalizer) and
carrier phase (Costas loop) have converged. At the beginning of every capture, the equalized samples z[t] of
the link above are therefore multiplied by
    g[t] = 10^(G0 d[t] / 20) exp(j phi0 d[t]),   d[t] = exp(-t / tau),
with an initial gain error G0 uniform in +-6 dB and an initial phase error phi0 uniform in +-45 deg (a
decision-directed loop locks to the nearest multiple of 90 deg), drawn once per channel and UE and shared by
all methods and resolutions, and tau = 500 symbols: longer than the EVM window, as for the slow gain loops of
the USRP receiver (AGC rate 1e-4 per sample, CMA gain 1e-4). After a few tau, the samples equal those of the
link above; the transient covers about 4 tau = 2000 of the 40,000 symbols of a capture.
EVM: as defined for the over-the-air measurement (paper, Sec. VI-C): per symbol, the magnitude of the error
vector with respect to the nearest 16-QAM point, normalized by the RMS amplitude of the constellation (1),
averaged over a sliding window of 200 symbols (decision-directed).
UE selection (one CDF per channel realization, then mean and +- one standard deviation over the 20 channels):
    both   - the windowed values of both UEs pooled;
    strong - only the UE with the larger channel gain ||h_k||^2 of the channel realization;
    weak   - only the other UE.
The selection depends on the channel only, so all methods are evaluated on the same UE. Also stored: the
steady-state EVM without the transient, decision-directed and data-aided.

  python evm_cdf_sim.py        # -> exp_results/evm_cdf_sim_20dB_startup_ue/, Figure/evm_cdf_sim/*.dat (paper repo)
  python evm_cdf_sim.py --blocks 32 --startup-tau 0 --output-dir exp_results/evm_cdf_sim_20dB_steady
                               # steady state only (the earlier figure, 32 blocks per channel)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for directory in (HERE.parent, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from constellation_16qam import (checkpoint_path, fingerprint, ide_dac, infer_gnn, load_model, qam16, receive,
                                 save_npz, unique_path, write_json)
import ide_baseline
from exp_cellfree_ablation import normalize_power
from train_sweep import PT, load_levels, quantize_to_levels
from wmmse_baseline import best_wmmse

K = 2
METHODS = ('gnn', 'wmmse', 'ide_cal')
SELECTIONS = ('both', 'strong', 'weak')
PAM = np.array([-3., -1., 1., 3.]) / math.sqrt(10)
FIGURE_DIR = HERE.parents[2] / 'Quantized-aware-training-and-deploy-study' / 'Figure' / 'evm_cdf_sim'
GRID = np.round(np.arange(0, 250.0001, 0.1), 4)                                             # EVM in %


def nearest_16qam(z):
    """Decision per I/Q dimension (nearest PAM-4 level); outer decisions are unbounded."""
    def dim(x):
        return PAM[np.abs(x[..., None] - PAM).argmin(-1)]
    return dim(z.real) + 1j * dim(z.imag)


def windowed(error, window):
    """Mean of the per-symbol error magnitudes over all windows of `window` consecutive symbols."""
    c = np.concatenate(([0.], np.cumsum(error)))
    return (c[window:] - c[:-window]) / window


def empirical_cdf(values):
    return np.searchsorted(np.sort(values), GRID, side='right') / len(values)


def crossing(cdf, level):
    """Smallest EVM where a nondecreasing CDF on GRID reaches `level` (linear interpolation)."""
    i = int(np.searchsorted(cdf, level))
    if i == 0:
        return float(GRID[0])
    if i >= len(GRID):
        return float('nan')
    x0, x1, y0, y1 = GRID[i - 1], GRID[i], cdf[i - 1], cdf[i]
    return float(x0 + (level - y0) / (y1 - y0) * (x1 - x0)) if y1 > y0 else float(x1)


def startup_gain(n_channels, n_symbols, tau, gain_db, phase_deg, seed):
    """Per channel and UE: g[t] = 10^(G0 d/20) exp(j phi0 d), d = exp(-t/tau); ones if tau == 0."""
    rng = np.random.default_rng(seed)
    g0 = rng.uniform(-gain_db, gain_db, (n_channels, K))
    phi0 = np.deg2rad(rng.uniform(-phase_deg, phase_deg, (n_channels, K)))
    decay = np.exp(-np.arange(n_symbols) / tau) if tau > 0 else np.zeros(n_symbols)
    return 10 ** (g0[..., None] * decay / 20) * np.exp(1j * phi0[..., None] * decay), g0, phi0


def receive_unquantized(x, H, symbols, noise):
    """receive() of constellation_16qam.py for a precoder output that is not on a DAC grid."""
    tx = normalize_power(x, PT)
    r0 = H.transpose(1, 2) @ tx
    gain = (r0 * symbols.conj()).sum(-1) / symbols.abs().square().sum(-1)
    return (r0 + noise) / gain[..., None]


def analyze(z, s_np, gain, strong, kinds, args):
    """z: C*B x K x Ns equalized samples -> per selection and kind: per-channel CDFs; start-up statistics."""
    cdfs = {sel: {kind: [] for kind in kinds} for sel in SELECTIONS}
    counts = {sel: {'above_50': 0, 'above_100': 0, 'raised_1pp': 0, 'windows': 0, 'max': 0.} for sel in SELECTIONS}
    for c in range(args.n_channels):
        rows = slice(c * args.blocks, (c + 1) * args.blocks)
        per_ue = []
        for k in range(K):
            stream = z[rows, k].reshape(-1)                         # the blocks of a UE, in order: one capture
            sent = s_np[rows, k].reshape(-1)
            values = {'dd': 100 * windowed(np.abs(stream - nearest_16qam(stream)), args.window),
                      'da': 100 * windowed(np.abs(stream - sent), args.window)}
            if 'dd_startup' in kinds:
                observed = stream * gain[c, k]
                values['dd_startup'] = 100 * windowed(np.abs(observed - nearest_16qam(observed)), args.window)
            per_ue.append(values)
        ues = {'both': range(K), 'strong': [strong[c]], 'weak': [k for k in range(K) if k != strong[c]]}
        for sel in SELECTIONS:
            for kind in kinds:
                cdfs[sel][kind].append(empirical_cdf(np.concatenate([per_ue[k][kind] for k in ues[sel]])))
            if 'dd_startup' in kinds:
                for k in ues[sel]:
                    transient, steady = per_ue[k]['dd_startup'], per_ue[k]['dd']
                    counts[sel]['above_50'] += int((transient > 50).sum())
                    counts[sel]['above_100'] += int((transient > 100).sum())
                    counts[sel]['raised_1pp'] += int((transient > steady + 1).sum())
                    counts[sel]['windows'] += len(transient)
                    counts[sel]['max'] = max(counts[sel]['max'], float(transient.max()))
    return cdfs, counts


def summarize(cdfs, counts, kinds):
    summary, curves = {}, {}
    for sel in SELECTIONS:
        summary[sel] = {}
        for kind in kinds:
            stack = np.stack(cdfs[sel][kind])
            mean, std = stack.mean(0), stack.std(0, ddof=1)
            medians = [crossing(cdf, 0.5) for cdf in stack]
            summary[sel][kind] = {
                'median_of_mean_cdf': crossing(mean, 0.5),
                'p10_of_mean_cdf': crossing(mean, 0.1), 'p90_of_mean_cdf': crossing(mean, 0.9),
                'p99_of_mean_cdf': crossing(mean, 0.99),
                'per_channel_median_mean': float(np.mean(medians)),
                'per_channel_median_std': float(np.std(medians, ddof=1)),
                'per_channel_median_min': float(np.min(medians)), 'per_channel_median_max': float(np.max(medians)),
                'max_band_halfwidth': float(std.max()),
            }
            curves[sel, kind] = (stack, mean, std)
        if 'dd_startup' in kinds:
            n = counts[sel]['windows']
            summary[sel]['startup'] = {'fraction_windows_above_50pct': counts[sel]['above_50'] / n,
                                       'fraction_windows_above_100pct': counts[sel]['above_100'] / n,
                                       'fraction_windows_raised_by_1pp': counts[sel]['raised_1pp'] / n,
                                       'max_windowed_evm': counts[sel]['max']}
    return summary, curves


def write_figure_data(figure_dir, name, mean, std):
    """Mean CDF with its band, trimmed to the EVM range where anything happens (+ 0.5 % margin)."""
    active = np.flatnonzero((mean + std > 0) & (mean - std < 1))
    lo_i, hi_i = max(active[0] - 5, 0), min(active[-1] + 5, len(GRID) - 1)
    sel = slice(lo_i, hi_i + 1)
    table = np.column_stack((GRID[sel], 100 * mean[sel], 100 * np.clip(mean - std, 0, 1)[sel],
                             100 * np.clip(mean + std, 0, 1)[sel]))
    np.savetxt(figure_dir / f'{name}.dat', table, fmt='%.4f', header='evm mean lo hi', comments='')
    band = np.vstack((table[:, [0, 3]], table[::-1][:, [0, 2]]))                           # closed polygon
    np.savetxt(figure_dir / f'{name}_band.dat', band, fmt='%.4f', header='evm cdf', comments='')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=HERE / 'exp_results' / 'evm_cdf_sim_20dB_startup_ue_cal')
    p.add_argument('--figure-dir', type=Path, default=FIGURE_DIR)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--snr-db', type=float, default=20.)
    p.add_argument('--start', type=int, default=4096)
    p.add_argument('--n-channels', type=int, default=20)
    p.add_argument('--blocks', type=int, default=320, help='blocks of 125 symbols per channel (one capture)')
    p.add_argument('--symbols', type=int, default=125)
    p.add_argument('--window', type=int, default=200)
    p.add_argument('--startup-tau', type=float, default=500., help='time constant in symbols; 0: no transient')
    p.add_argument('--startup-gain-db', type=float, default=6., help='initial gain error uniform in +-this')
    p.add_argument('--startup-phase-deg', type=float, default=45., help='initial phase error uniform in +-this')
    p.add_argument('--seed-symbols', type=int, default=1234)
    p.add_argument('--seed-noise', type=int, default=20260930)
    p.add_argument('--seed-startup', type=int, default=2026)
    p.add_argument('--seed-cal', type=int, default=777, help='training symbols of the calibrated IDE beta')
    return p


def main():
    args = parser().parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to replace an existing experiment: {output}; use a new --output-dir')
    source = unique_path(f'datasets/cellfree/M_40_K_{K}_Ntr_200000_*/Htest.npy')
    models = {b: checkpoint_path(K, b) for b in (1, 2, 3)}
    output.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    noise_var = PT / 10 ** (args.snr_db / 10)
    channel_ids = np.arange(args.start, args.start + args.n_channels)
    dataset = np.load(source, mmap_mode='r')
    H_channels = torch.from_numpy(np.array(dataset[channel_ids], dtype=np.complex64))        # C x M x K
    H = H_channels.repeat_interleave(args.blocks, dim=0)                                    # C*B x M x K
    channel_gain = torch.linalg.vector_norm(H_channels, dim=1).square().numpy()             # C x K, ||h_k||^2
    strong = channel_gain.argmax(-1)
    symbols, tx_bits = qam16(len(H), K, args.symbols, args.seed_symbols)
    rng = torch.Generator().manual_seed(args.seed_noise + K)
    normal = torch.randn((*symbols.shape, 2), generator=rng) * math.sqrt(noise_var / 2)
    noise = torch.complex(normal[..., 0], normal[..., 1])
    n_capture = args.blocks * args.symbols
    gain, g0, phi0 = startup_gain(args.n_channels, n_capture, args.startup_tau, args.startup_gain_db,
                                  args.startup_phase_deg, args.seed_startup)
    save_npz(output / 'shared.npz', H_channels=H_channels, channel_indices=channel_ids, tx_symbols=symbols,
             tx_bits=tx_bits, noise=noise, blocks_per_channel=np.int64(args.blocks),
             startup_g0_db=g0, startup_phi0_rad=phi0, channel_gain=channel_gain, strong_ue=strong)
    with torch.inference_mode():
        V = best_wmmse(H_channels, noise_var).repeat_interleave(args.blocks, dim=0)
        unquantized = V @ symbols                                                           # C*B x M x Ns
        linear_input = unquantized / torch.linalg.vector_norm(V, dim=2, keepdim=True).clamp_min(1e-12)
    kinds = ('dd_startup', 'dd', 'da') if args.startup_tau > 0 else ('dd', 'da')
    figure_kind = kinds[0]
    s_np = symbols.numpy().astype(np.complex128)
    results, cdf_arrays, started = {}, {}, time.perf_counter()

    def finish(key, z, clock):
        cdfs, counts = analyze(z, s_np, gain, strong, kinds, args)
        summary, curves = summarize(cdfs, counts, kinds)
        results[key] = summary
        for (sel, kind), (stack, mean, std) in curves.items():
            cdf_arrays[f'{key}_{sel}_{kind}'] = stack.astype(np.float32)
            if kind == figure_kind:
                write_figure_data(args.figure_dir, f'{key}_{sel}', mean, std)
        line = f'{key:10s}: ' + ' | '.join(
            f'{sel} median {summary[sel][figure_kind]["median_of_mean_cdf"]:.2f}%' for sel in SELECTIONS)
        if 'startup' in summary['both']:
            st = summary['both']['startup']
            line += (f' | start-up: {100 * st["fraction_windows_raised_by_1pp"]:.1f}% windows raised, '
                     f'max {st["max_windowed_evm"]:.0f}%')
        print(line + f'  ({time.perf_counter() - clock:.0f}s)', flush=True)

    clock = time.perf_counter()
    with torch.inference_mode():
        z = receive_unquantized(unquantized, H, symbols, noise).numpy().astype(np.complex128)
    finish('wmmse_unq', z, clock)
    for b in (1, 2, 3):
        levels = load_levels(b, torch.device('cpu'))
        model, _ = load_model(models[b], K, b, levels.to(device), device)
        for method in METHODS:
            clock = time.perf_counter()
            if method == 'gnn':
                dac = infer_gnn(model, H, symbols, model.levels, device)
            elif method == 'ide_cal':
                train, _ = qam16(args.n_channels, K, args.symbols, args.seed_cal)
                beta = ide_baseline.calibrated_beta(H_channels, train, levels, args.snr_db)
                dac = ide_dac(H, symbols, levels, args.snr_db, 'fixed',
                              beta_fixed=beta.repeat_interleave(args.blocks))
            else:
                dac = quantize_to_levels(linear_input, levels)
            case = receive(dac, levels, H, symbols, tx_bits, noise, noise_var)
            z = case['rx_equalized'].numpy().astype(np.complex128)                             # C*B x K x Ns
            del case, dac
            finish(f'{method}_b{b}', z, clock)
        del model
    np.savez_compressed(output / 'cdfs.npz', grid=GRID, **cdf_arrays)
    write_json(output / 'summary.json', {
        'definition': 'EVM % = 100 * mean over a sliding window of |z - ref|; ref = nearest 16-QAM point (dd) or '
                      'transmitted symbol (da); constellation RMS amplitude 1. dd_startup: dd of z*g[t] with the '
                      'receiver start-up transient g[t]. Per channel one CDF of the windowed values of the selected '
                      'UEs (both, strong = larger ||h_k||^2, weak); mean over channels and +- one std (ddof=1). '
                      f'Figure data ({figure_kind}): Figure/evm_cdf_sim/<method>_<selection>.dat',
        'figure_kind': figure_kind, 'selections': list(SELECTIONS),
        'strong_ue_per_channel': strong.tolist(), 'channel_gain_per_channel': channel_gain.tolist(),
        'results': results})
    write_json(output / 'metadata.json', {
        'k': K, 'm': 40, 'pt': PT, 'snr_db': args.snr_db, 'noise_variance_complex': noise_var,
        'channel_indices': channel_ids.tolist(), 'blocks_per_channel': args.blocks, 'symbols_per_block': args.symbols,
        'window': args.window, 'seed_symbols': args.seed_symbols, 'seed_noise': args.seed_noise + K,
        'seed_cal': args.seed_cal,
        'startup': {'tau_symbols': args.startup_tau, 'initial_gain_db_uniform_pm': args.startup_gain_db,
                    'initial_phase_deg_uniform_pm': args.startup_phase_deg, 'seed': args.seed_startup,
                    'model': 'g[t] = 10^(G0 exp(-t/tau)/20) exp(j phi0 exp(-t/tau)) applied to z[t] from the start of '
                             'each capture; same draw for all methods and resolutions'},
        'methods': ['wmmse_unq'] + list(METHODS),
        'wmmse_unq': 'x = V s with V = best_wmmse(H, noise_var) (total power Pt), no DAC; per-block power normalization',
        'link': 'constellation_16qam.py receive(): per-block power normalization, H^T x + AWGN, oracle block gain',
        'dataset': fingerprint(source), 'checkpoints': {str(b): fingerprint(p) for b, p in models.items()},
        'sources': {name: fingerprint(HERE / name) for name in
                    ('evm_cdf_sim.py', 'constellation_16qam.py', 'ide_baseline.py', 'wmmse_baseline.py', 'train_sweep.py')},
        'figure_dir': str(args.figure_dir.resolve()), 'seconds': time.perf_counter() - started})
    print(f'Completed: {output}')


if __name__ == '__main__':
    main()
