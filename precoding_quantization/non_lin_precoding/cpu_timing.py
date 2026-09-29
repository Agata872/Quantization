"""CPU inference time per channel realization (one block of N_s = 125 symbol vectors), M = 40.

Times the code the paper's results come from (imported, not re-implemented):
  ZF/MRT + DAC   train_sweep.linear_quantized + normalize_power
  WMMSE + DAC    wmmse_baseline.best_wmmse (3 starts x 150 iterations, float64) + wmmse_baseline.quantized
  GNN            train_sweep.run_model_warm (warm start, features, message passing, argmax) + normalize_power
  GNN-GD         GNN + gd_baseline.gd_optimize with the recipe of gd_refine_sweep (200 steps, LS objective)
Before timing, the loaded GNN is checked against the stored 20-dB rate on the 2048 held-out channels.

  python cpu_timing.py   (CPU-only PyTorch is enough; results in exp_results/cpu_timing.json)
"""
import glob
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch

ND = os.path.dirname(os.path.abspath(__file__))           # portable (was a hard-coded server path)
for p in (ND, os.path.dirname(ND)):
    sys.path.insert(0, p)
os.chdir(ND)

from exp_cellfree_ablation import GNNv2, sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import load_levels, run_model_warm, linear_quantized, PT, M_FULL  # noqa: E402
from wmmse_baseline import best_wmmse, quantized as wmmse_quantized  # noqa: E402
from gd_baseline import gd_optimize  # noqa: E402
from gd_refine_sweep import recipe  # noqa: E402

DEV = torch.device('cpu')
NV20 = PT / 10 ** (20 / 10)
NS = 125
N_CH = int(os.environ.get('N_CH', 10))       # channels timed per method
REPS = int(os.environ.get('REPS', 5))
THREADS = [int(t) for t in os.environ.get('THREADS', '1,16').split(',')]
CONFIGS = [(K, 2) for K in (2, 6)]


def data(K):
    d = glob.glob(os.path.join(ND, 'datasets', 'cellfree', f'M_40_K_{K}_Ntr_200000_*'))[0]
    H = np.load(os.path.join(d, 'Htest.npy'))
    s = np.load(os.path.join(d, 'stest.npy')).reshape(K, H.shape[0], NS).transpose(1, 0, 2)
    return (torch.from_numpy(H[4096:6144].astype(np.complex64)),
            torch.from_numpy(np.ascontiguousarray(s[4096:6144]).astype(np.complex64)))


def load_model(K, b, levels):
    pat = ('stored_models_cellfree_k1_continue/*/model_best.pt' if (K, b) == (1, 1)
           else f'stored_models_cellfree_sweep_continue/M40_K{K}_b{b}_*/model_best.pt')
    ck = torch.load(glob.glob(os.path.join(ND, pat))[0], map_location=DEV, weights_only=False)
    m = GNNv2(M_FULL, K, 128, 4, b, 0.25, levels, input_mode='polar4', feat_stats=ck['feat_stats'])
    m.load_state_dict(ck['model'])
    m.output_type = 'argmax'
    m.eval()
    return m


def stored_rate20(K, b):
    c = json.load(open(glob.glob(os.path.join(ND, f'stored_models_cellfree_sweep_continue/M40_K{K}_b{b}_*/snr_curves.json'))[0]))
    return c['gnn'][c['snr_db'].index(20.0)]


def median_ms(fn, H, s):
    ts = []
    for i in range(N_CH):
        Hc, sc = H[i:i + 1], s[i:i + 1]
        fn(Hc, sc)                                   # warm-up for this channel
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn(Hc, sc)
            ts.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(ts)


def main():
    print(f'{platform.processor() or platform.machine()} | torch {torch.__version__} | '
          f'N_CH={N_CH} REPS={REPS}', flush=True)
    cfg = recipe()
    out = {'torch': torch.__version__, 'n_channels': N_CH, 'reps': REPS, 'gd_recipe': cfg, 'results': []}
    for K, b in CONFIGS:
        H, s = data(K)
        levels = load_levels(b, DEV)
        model = load_model(K, b, levels)

        torch.set_num_threads(32)
        with torch.no_grad():
            r = torch.cat([sumrate_bussgang_ls(normalize_power(run_model_warm(model, H[i:i + 256], s[i:i + 256],
                                                                              levels, PT, 16), PT),
                                               H[i:i + 256], s[i:i + 256], NV20) for i in range(0, 2048, 256)]).mean().item()
        ref = stored_rate20(K, b)
        print(f'K={K} b={b}: check GNN @20dB on 2048 channels {r:.4f} vs stored {ref:.4f}', flush=True)
        assert abs(r - ref) < 2e-3, (r, ref)

        @torch.no_grad()
        def zf(Hc, sc):
            return normalize_power(linear_quantized(Hc, sc, levels, PT), PT)

        @torch.no_grad()
        def wmmse(Hc, sc):
            return wmmse_quantized(best_wmmse(Hc, NV20), sc, levels, False)

        @torch.no_grad()
        def gnn(Hc, sc):
            return normalize_power(run_model_warm(model, Hc, sc, levels, PT, 16), PT)

        @torch.no_grad()
        def gnn_one_symbol(Hc, sc):
            return run_model_warm(model, Hc, sc[:, :, :1], levels, PT, 16)

        def gnn_gd(Hc, sc):
            y0 = gnn(Hc, sc)
            return gd_optimize(Hc, sc, levels, NV20, y0, rate='ls', **cfg)[0]

        for th in THREADS:
            torch.set_num_threads(th)
            row = {'K': K, 'bits': b, 'threads': th}
            for name, fn in (('zf_dac', zf), ('wmmse_dac', wmmse), ('gnn', gnn),
                             ('gnn_one_symbol', gnn_one_symbol), ('gnn_gd', gnn_gd)):
                row[name] = median_ms(fn, H, s)
            out['results'].append(row)
            print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(ND, 'exp_results', 'cpu_timing.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
