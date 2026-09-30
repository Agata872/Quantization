"""Discard the windows with the largest EVM of every capture for the paper's EVM figure (fig:tp_evm_cdf).

evm_cdf_sim.py stores, per channel realization (one capture) and UE selection, the empirical CDF F_c of the windowed
EVM on a 0.1 % grid (cdfs.npz). Removing the fraction q of the windows with the largest EVM and renormalizing gives
F_c'(x) = min(F_c(x) / (1 - q), 1), exact up to the rounding of q times the number of windows. The figure shows the
mean of F_c' over the channels and +- one standard deviation (ddof=1) across them, as for the untrimmed curves.

  python trim_evm_cdf.py [--trim 0.03]
      -> Figure/evm_cdf_sim/<method>_<selection>_t97.dat (+ _band.dat) in the paper repo, and
         <run>/trimmed_t97_summary.json with the medians of the mean CDFs (the dots of the figure)
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / 'exp_results' / 'evm_cdf_sim_20dB_startup_ue_cal'
FIGURE_DIR = HERE.parents[2] / 'Quantized-aware-training-and-deploy-study' / 'Figure' / 'evm_cdf_sim'


def crossing(grid, cdf, level):
    """Smallest EVM where a nondecreasing CDF reaches `level` (linear interpolation)."""
    i = int(np.searchsorted(cdf, level - 1e-12))
    if i == 0:
        return float(grid[0])
    x0, x1, y0, y1 = grid[i - 1], grid[i], cdf[i - 1], cdf[i]
    return float(x0 + (level - y0) / (y1 - y0) * (x1 - x0)) if y1 > y0 else float(x1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=RUN)
    ap.add_argument('--figure-dir', type=Path, default=FIGURE_DIR)
    ap.add_argument('--trim', type=float, default=0.03, help='fraction of the worst windows removed per capture')
    args = ap.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    kind = summary['figure_kind']
    cdfs = np.load(args.run / 'cdfs.npz')
    grid = cdfs['grid']
    tag = f'_t{round(100 * (1 - args.trim))}'
    medians = {}
    for key in sorted(k for k in cdfs.files if k.endswith('_' + kind)):
        name = key[:-len('_' + kind)]                                   # <method>_<selection>
        trimmed = np.minimum(cdfs[key].astype(np.float64) / (1 - args.trim), 1.0)
        mean, std = trimmed.mean(0), trimmed.std(0, ddof=1)
        active = np.flatnonzero((mean + std > 0) & (mean - std < 1))
        sel = slice(max(active[0] - 5, 0), min(active[-1] + 5, len(grid) - 1) + 1)
        table = np.column_stack((grid[sel], 100 * mean[sel], 100 * np.clip(mean - std, 0, 1)[sel],
                                 100 * np.clip(mean + std, 0, 1)[sel]))
        np.savetxt(args.figure_dir / f'{name}{tag}.dat', table, fmt='%.4f', header='evm mean lo hi', comments='')
        band = np.vstack((table[:, [0, 3]], table[::-1][:, [0, 2]]))    # closed polygon
        np.savetxt(args.figure_dir / f'{name}{tag}_band.dat', band, fmt='%.4f', header='evm cdf', comments='')
        medians[name] = {'median_of_mean_cdf': crossing(grid, mean, 0.5), 'evm_at_100pct': crossing(grid, mean, 1.0)}
    (args.run / f'trimmed{tag}_summary.json').write_text(json.dumps(
        {'trim_per_capture': args.trim, 'kind': kind, 'definition': "F_c' = min(F_c / (1 - trim), 1) per channel",
         'results': medians}, indent=1) + '\n')
    for name, v in medians.items():
        print(f'{name:22s} median {v["median_of_mean_cdf"]:6.2f} %, mean CDF reaches 100 % at {v["evm_at_100pct"]:5.1f} %')


if __name__ == '__main__':
    main()
