"""CPU inference time of every method of Table I (tab:tp_complexity) in ONE session, interleaved per channel.

Supersedes exp_results/cpu_timing.json (2026-09-28) and cpu_timing_baselines.json: separate sessions of the same
code differed by up to ~20% for the single-thread GNN (thread placement on the two-socket machine), which made
ratios between rows of different sessions unreliable. Here
  * the process is pinned to the 16 physical cores of socket 0 (logical CPUs 0-15, NUMA node 0) before torch is
    imported, so that 1 and 16 threads always run on the same socket and memory node;
  * for every thread count and channel realization, all methods are run back to back (one warm-up + REPS timed
    runs each), so that slow drifts of the machine state affect all methods alike.
Protocol otherwise as in cpu_timing.py: per channel realization (one block of N_s = 125 symbol vectors), M = 40,
SNR = 20 dB, K in {2, 6}, N_CH = 10 channels x REPS = 5 runs, median (quartiles stored as well).

Methods (the code the results come from, imported): ZF + DAC, WMMSE + DAC, GNN, GNN-GD (b = 2, cpu_timing.py),
IDE with beta_WF / block beta (b = 2, ide_baseline.precode, float64), SQUID (b = 1, squid_baseline.run, float64,
BER-tuned gain), single-symbol latencies, and single-precision copies of IDE / SQUID (cpu_timing_baselines.py).

Environment variables:
  PIN_CPUS   logical CPUs to pin the process to, e.g. '0-15' or '0,1' (Linux: os.sched_setaffinity, otherwise
             psutil); applied before torch is imported so that every torch thread inherits it; unset: no pinning.
             The server run of 2026-09-29 used PIN_CPUS=0-15 (socket 0 of 2x AMD EPYC 7302).
  THREADS    comma-separated torch thread counts (default '1,16', from cpu_timing.py)
  N_CH, REPS channels and runs per channel (default 10, 5, from cpu_timing.py)
  OUT        output file name in exp_results/ (default cpu_timing_all.json)

  PIN_CPUS=0-15 python cpu_timing_all.py      # -> exp_results/cpu_timing_all.json
"""
import os


def _pin_from_env():
    spec = os.environ.get('PIN_CPUS', '').strip()
    if not spec:
        return None
    cpus = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        cpus.update(range(int(a), int(b or a) + 1))
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, cpus)
    else:                                          # Windows / macOS
        import psutil
        psutil.Process().cpu_affinity(sorted(cpus))
    return sorted(cpus)


PINNED = _pin_from_env()                           # before torch: every thread inherits the mask

import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

ND = os.path.dirname(os.path.abspath(__file__))
for p in (ND, os.path.dirname(ND)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpu_timing as C  # noqa: E402
import cpu_timing_baselines as B  # noqa: E402
import ide_baseline  # noqa: E402
import squid_baseline  # noqa: E402
from exp_cellfree_ablation import sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import load_levels, run_model_warm, linear_quantized, PT  # noqa: E402
from wmmse_baseline import best_wmmse, quantized as wmmse_quantized  # noqa: E402
from gd_baseline import gd_optimize  # noqa: E402
from gd_refine_sweep import recipe  # noqa: E402

SNR_DB = 20.0
NV20 = PT / 10 ** (SNR_DB / 10)
OUT = os.path.join(ND, 'exp_results', os.environ.get('OUT', 'cpu_timing_all.json'))


def cpu_name():
    try:
        with open('/proc/cpuinfo') as f:
            return next(line.split(':', 1)[1].strip() for line in f if line.startswith('model name'))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def methods_for(K, H, s):
    lv1, lv2 = load_levels(1, C.DEV), load_levels(2, C.DEV)
    gnn1, gnn2 = C.load_model(K, 1, lv1), C.load_model(K, 2, lv2)
    cfg = recipe()

    torch.set_num_threads(16)
    with torch.no_grad():
        r = torch.cat([sumrate_bussgang_ls(normalize_power(run_model_warm(gnn2, H[i:i + 256], s[i:i + 256], lv2, PT, 16),
                                                           PT), H[i:i + 256], s[i:i + 256], NV20)
                       for i in range(0, 2048, 256)]).mean().item()
    ref = C.stored_rate20(K, 2)
    print(f'K={K} b=2: check GNN @20dB on 2048 channels {r:.4f} vs stored {ref:.4f}', flush=True)
    assert abs(r - ref) < 2e-3, (r, ref)

    @torch.no_grad()
    def zf(Hc, sc):
        return normalize_power(linear_quantized(Hc, sc, lv2, PT), PT)

    @torch.no_grad()
    def wmmse(Hc, sc):
        return wmmse_quantized(best_wmmse(Hc, NV20), sc, lv2, False)

    def gnn_fn(model, lv, one=False):
        @torch.no_grad()
        def f(Hc, sc):
            if one:
                return run_model_warm(model, Hc, sc[:, :, :1], lv, PT, 16)
            return normalize_power(run_model_warm(model, Hc, sc, lv, PT, 16), PT)
        return f

    gnn = gnn_fn(gnn2, lv2)

    def gnn_gd(Hc, sc):
        return gd_optimize(Hc, sc, lv2, NV20, gnn(Hc, sc), rate='ls', **cfg)[0]

    def ide_fn(mode, fp32=False, one=False):
        pre = B.IDE32['precode'] if fp32 else ide_baseline.precode
        return lambda Hc, sc: pre(Hc, sc[:, :, :1] if one else sc, lv2, SNR_DB, 'ide', mode)

    def squid_fn(fp32=False, one=False):
        run = B.SQUID32['run'] if fp32 else squid_baseline.run
        return lambda Hc, sc: run(Hc, sc[:, :, :1] if one else sc, SNR_DB, B.SQUID_GAIN[K])

    return {'zf_dac': zf, 'wmmse_dac': wmmse, 'gnn': gnn, 'gnn_b1': gnn_fn(gnn1, lv1), 'gnn_gd': gnn_gd,
            'ide_wf': ide_fn('wf'), 'ide_block': ide_fn('block'), 'squid_b1': squid_fn(),
            'gnn_one_symbol': gnn_fn(gnn2, lv2, one=True), 'ide_wf_one_symbol': ide_fn('wf', one=True),
            'squid_b1_one_symbol': squid_fn(one=True),
            'ide_wf_fp32': ide_fn('wf', fp32=True), 'ide_block_fp32': ide_fn('block', fp32=True),
            'squid_b1_fp32': squid_fn(fp32=True)}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f'{cpu_name()} | {platform.system()} | torch {torch.__version__} | pinned {PINNED} | threads {C.THREADS} | '
          f'N_CH={C.N_CH} REPS={C.REPS} -> {OUT}', flush=True)
    out = {'cpu': cpu_name(), 'os': platform.platform(), 'torch': torch.__version__, 'pinned_cpus': PINNED,
           'threads': C.THREADS, 'n_channels': C.N_CH, 'reps': C.REPS, 'snr_db': SNR_DB, 'gd_recipe': recipe(),
           'squid_gain': B.SQUID_GAIN, 'results': []}
    for K in (2, 6):
        H, s = C.data(K)
        meth = methods_for(K, H, s)
        for th in C.THREADS:
            torch.set_num_threads(th)
            ts = {m: [] for m in meth}
            for i in range(C.N_CH):
                Hc, sc = H[i:i + 1], s[i:i + 1]
                for m, fn in meth.items():                      # all methods back to back on this channel
                    fn(Hc, sc)
                    for _ in range(C.REPS):
                        t0 = time.perf_counter()
                        fn(Hc, sc)
                        ts[m].append(time.perf_counter() - t0)
            row = {'K': K, 'threads': th}
            for m, v in ts.items():
                q = statistics.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
                row[m] = 1e3 * statistics.median(v)
                row[m + '_iqr'] = [1e3 * q[0], 1e3 * q[2]]
            out['results'].append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                              if not k.endswith('_iqr')}), flush=True)
            json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
