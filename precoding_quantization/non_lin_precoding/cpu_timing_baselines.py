"""CPU inference time of the symbol-level baselines of the BER comparison, with the protocol of cpu_timing.py:
per channel realization (one block of N_s = 125 symbol vectors), M = 40, SNR = 20 dB, K in {2, 6}, 1 and 16
threads, median over N_CH = 10 channels x REPS = 5 runs after one warm-up per channel.

Timed code = the code the BER/rate results come from (imported, not re-implemented):
  IDE, beta_WF     ide_baseline.precode(..., 'ide', 'wf')      b = 2, float64 (as used)
  IDE, block beta  ide_baseline.precode(..., 'ide', 'block')   b = 2, float64 (as used)
  SQUID            squid_baseline.run(...)                      b = 1 (1-bit only), float64 (as used)
  GNN, ZF + DAC    as in cpu_timing.py (b = 2; GNN also b = 1), re-timed in the same session as a consistency check
                   against exp_results/cpu_timing.json
Also recorded:
  *_one_symbol     latency of a single symbol vector (IDE beta_WF, SQUID, GNN); IDE with block beta needs the block
  *_fp32           IDE / SQUID with every float64/complex128 replaced by float32/complex64 (timing-only copies made
                   from the same source); the fraction of identical output samples vs. float64 is stored as well

  python cpu_timing_baselines.py      # -> exp_results/cpu_timing_baselines.json
"""
import inspect
import json
import os
import platform
import statistics
import sys
import time

import torch

ND = os.path.dirname(os.path.abspath(__file__))
for p in (ND, os.path.dirname(ND)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpu_timing as C  # noqa: E402  (data, load_model, median_ms, N_CH, REPS, THREADS, protocol)
import ide_baseline  # noqa: E402
import squid_baseline  # noqa: E402
from exp_cellfree_ablation import normalize_power  # noqa: E402
from train_sweep import load_levels, run_model_warm, linear_quantized, PT  # noqa: E402

SNR_DB = 20.0
SQUID_GAIN = {2: 0.01, 6: 0.01}          # BER-tuned gains at 20 dB (ber_16qam_20dB.json); run time does not depend on it


def _fp32_copy(module, names):
    """Timing-only single-precision copies of module functions (same source, dtypes replaced)."""
    ns = dict(vars(module))
    for n in names:
        src = inspect.getsource(getattr(module, n))
        src = src.replace('torch.complex128', 'torch.complex64').replace('torch.float64', 'torch.float32')
        src = src.replace('.double()', '.float()')
        exec(compile(src, f'<{module.__name__}.{n} fp32>', 'exec'), ns)
    return ns


IDE32 = _fp32_copy(ide_baseline, ['beta_wf', 'ide', 'precode'])
SQUID32 = _fp32_copy(squid_baseline, ['squid', 'run'])


def agree(y1, y2):
    """Fraction of transmit samples on which two power-normalized outputs coincide."""
    return ((y1 - y2).abs() < 1e-3 * y1.abs().mean()).float().mean().item()


def main():
    print(f'{platform.processor() or platform.machine()} | torch {torch.__version__} | N_CH={C.N_CH} REPS={C.REPS}',
          flush=True)
    ref = {(r['K'], r['threads']): r for r in json.load(open(os.path.join(ND, 'exp_results', 'cpu_timing.json')))['results']}
    out = {'torch': torch.__version__, 'n_channels': C.N_CH, 'reps': C.REPS, 'snr_db': SNR_DB, 'results': [],
           'fp32_agreement': []}
    for K in (2, 6):
        H, s = C.data(K)
        lv1, lv2 = load_levels(1, C.DEV), load_levels(2, C.DEV)
        gnn1, gnn2 = C.load_model(K, 1, lv1), C.load_model(K, 2, lv2)

        @torch.no_grad()
        def zf(Hc, sc):
            return normalize_power(linear_quantized(Hc, sc, lv2, PT), PT)

        def gnn_fn(model, lv, one=False):
            @torch.no_grad()
            def f(Hc, sc):
                if one:
                    return run_model_warm(model, Hc, sc[:, :, :1], lv, PT, 16)
                return normalize_power(run_model_warm(model, Hc, sc, lv, PT, 16), PT)
            return f

        def ide_fn(mode, fp32=False, one=False):
            pre = IDE32['precode'] if fp32 else ide_baseline.precode
            return lambda Hc, sc: pre(Hc, sc[:, :, :1] if one else sc, lv2, SNR_DB, 'ide', mode)

        def squid_fn(fp32=False, one=False):
            run = SQUID32['run'] if fp32 else squid_baseline.run
            return lambda Hc, sc: run(Hc, sc[:, :, :1] if one else sc, SNR_DB, SQUID_GAIN[K])

        methods = {'zf_dac': zf, 'gnn': gnn_fn(gnn2, lv2), 'gnn_b1': gnn_fn(gnn1, lv1),
                   'ide_wf': ide_fn('wf'), 'ide_block': ide_fn('block'), 'squid_b1': squid_fn(),
                   'gnn_one_symbol': gnn_fn(gnn2, lv2, one=True), 'ide_wf_one_symbol': ide_fn('wf', one=True),
                   'squid_b1_one_symbol': squid_fn(one=True),
                   'ide_wf_fp32': ide_fn('wf', fp32=True), 'ide_block_fp32': ide_fn('block', fp32=True),
                   'squid_b1_fp32': squid_fn(fp32=True)}

        torch.set_num_threads(16)
        Hc, sc = H[:C.N_CH], s[:C.N_CH]
        out['fp32_agreement'].append({'K': K, **{m: agree(methods[m](Hc, sc), methods[m + '_fp32'](Hc, sc))
                                                 for m in ('ide_wf', 'ide_block', 'squid_b1')}})
        print(json.dumps(out['fp32_agreement'][-1]), flush=True)

        for th in C.THREADS:
            torch.set_num_threads(th)
            row = {'K': K, 'threads': th}
            for name, fn in methods.items():
                row[name] = C.median_ms(fn, H, s)
            r0 = ref[(K, th)]
            row['check_vs_cpu_timing'] = {m: row[m] / r0[m] for m in ('zf_dac', 'gnn', 'gnn_one_symbol')}
            out['results'].append(row)
            print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(ND, 'exp_results', 'cpu_timing_baselines.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
