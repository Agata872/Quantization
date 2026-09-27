"""GNN-GD for every finished run in stored_models_cellfree_sweep_K_bits: the K/b sweep runs and the
K=1, 1-bit run.

GNN-GD = gd_baseline.gd_optimize initialized at the GNN's own argmax output: every DAC level of
every (symbol, AP, I/Q branch) gets a logit vector, peaked at the GNN's choice, and all of them are
optimized per channel by Adam on the LS Bussgang sum rate; the best hard iterate is kept, so the
result is on the DAC grid and never below the GNN. The recipe is the one of the K=1, 1-bit run,
read from its gd_baseline_20dB.json ('gd_from_gnn'), with ONE change: ITERS = 200 gradient steps
instead of 1000, the iteration budget of the Opt-GNN baseline of arXiv:2606.23141 (GNN output
refined by gradient ascent, I = 200). lr, scale, tau1 and the 'soft' estimator are copied as they
are, tau0 = 1.0 (so tau is annealed 1 -> 0.1 over the 200 steps), and the objective and the score
are both the LS estimator (rate='ls'). Nothing is re-tuned per (K, b).

Unlike gd_baseline_20dB.json, which refined at 20 dB only, each run is refined at every SNR of the
grid (-30:30:5 dB): the objective depends on the noise variance, so every SNR is its own
optimization, always started from the same GNN output (the GNN has no SNR input). Channels are the
2048 held-out ones (test set [4096:6144]) every curve uses. Before refining, the GNN start rates are
checked against the stored snr_curves.json 'gnn' curve (same weights and channels) at every SNR.

Per run folder it writes gnn_gd_curves.json, updated after every SNR so an interrupted job resumes
where it stopped (gd_baseline_20dB.json of the K=1 run is left as it is). --assemble adds
'gnn_gd' (per SNR) and 'gnn_gd_ref' (20 dB) to the matching entries of
figures_ls/results_for_figures.json.

  CUDA_VISIBLE_DEVICES=0 python gd_refine_sweep.py --runs K1b1,K2b1        # refine some runs
  CUDA_VISIBLE_DEVICES=1 python gd_refine_sweep.py --claim-dir /tmp/claims  # one of several workers
  python gd_refine_sweep.py --assemble                                     # merge into the figure json
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp_cellfree_ablation as k1  # noqa: E402  (data and forward of the K=1 run)
from exp_cellfree_ablation import GNNv2, sumrate_bussgang_ls, normalize_power  # noqa: E402
from gd_baseline import gd_optimize  # noqa: E402
from train_sweep import load_data, load_levels, run_model_warm, EVAL, PT, M_FULL  # noqa: E402
from reeval_sweep import SNR, REF_SNR, LS_NOTE, nv, run_kind  # noqa: E402

MARK = 'gd_refine_sweep.py'
SWEEP_DIR = os.path.join(CURRENT_DIR, 'stored_models_cellfree_sweep_K_bits')
RECIPE_RUN = 'M_40_K_1_1bit_polar4_mrtinit_refine1_tau1to0.25in3_lr0.001_dl128_L4'
ITERS = 200
OUT = 'gnn_gd_curves.json'
TRACE_EVERY = 10


def recipe():
    ref = json.load(open(os.path.join(SWEEP_DIR, RECIPE_RUN, 'gd_baseline_20dB.json')))
    assert ref['estimator'] == 'ls', ref['estimator']
    return dict(ref['config'], iters=ITERS)


def k_bits(d, kind):
    if kind == 'k1':
        return 1, 1
    h = json.load(open(os.path.join(d, 'history.json')))['config']
    return h['K'], h['bits']


def tag(d, kind):
    return 'K{}b{}'.format(*k_bits(d, kind))


def all_runs():
    return sorted((d, run_kind(d)) for d in glob.glob(os.path.join(SWEEP_DIR, 'M*')) if run_kind(d))


def eval_data(K, kind, dev, cache):
    if cache.get('key') != (K, kind):
        if kind == 'k1':
            _, _, Hte, Ste = k1.load_data(dev, 1000)
            ev = k1.EVAL
        else:
            _, _, Hte, Ste = load_data(K, 1000, dev)
            ev = EVAL
        assert (ev.start, ev.stop) == (4096, 6144)
        cache.update(key=(K, kind), data=(Hte[ev], Ste[ev]))
    return cache['data']


def gnn_raw_output(d, kind, K, bits, dev, Hev, sev):
    """Argmax output of model_best.pt on the eval slice, on the DAC grid (not power-normalized)."""
    ck = torch.load(os.path.join(d, 'model_best.pt'), map_location=dev, weights_only=False)
    c = ck['cfg']
    if kind == 'k1':
        levels = (np.sqrt(PT / (2 * M_FULL)) * torch.tensor([-1.0, 1.0])).to(dev)
        m = GNNv2(M_FULL, 1, c['dl'], c['layers'], 1, c['tau_final'], levels, input_mode='polar4',
                  feat_stats=ck['feat_stats']).to(dev)
        forward = lambda H, s: k1.run_model(m, H, s, 16, c['refine'], c['init'])  # noqa: E731
    else:
        levels = load_levels(bits, dev)
        m = GNNv2(M_FULL, K, c['dl'], c['layers'], bits, c['tau_final'], levels, input_mode='polar4',
                  feat_stats=ck['feat_stats'], logit_norm=c.get('logit_norm')).to(dev)
        warm = not c.get('no_warm_start', False)
        forward = lambda H, s: run_model_warm(m, H, s, levels, PT, 16, warm)  # noqa: E731
    m.load_state_dict(ck['model'])
    m.eval()
    m.output_type = 'argmax'
    with torch.no_grad():
        y = torch.cat([forward(Hev[b:b + 256], sev[b:b + 256]) for b in range(0, Hev.shape[0], 256)])
    del m
    return y, levels


def refine_run(d, kind, dev, cfg, chunk, cache):
    K, bits = k_bits(d, kind)
    f = os.path.join(d, 'snr_curves.json')
    if not os.path.exists(f):
        f = os.path.join(d, 'snr_curves_ls.json')      # K=1 b=1 continued by continue_k1_training.py
    stored = json.load(open(f))
    # LS curves on the common grid: a re-scored sweep run, or a continued one (continue_sweep_training.py,
    # continue_k1_training.py)
    assert (stored.get('corrected_by') in ('reeval_sweep.py', 'continue_sweep_training.py')
            or stored.get('estimator') == 'ls') and stored['snr_db'] == SNR, d
    Hev, sev = eval_data(K, kind, dev, cache)
    y0, levels = gnn_raw_output(d, kind, K, bits, dev, Hev, sev)

    f = os.path.join(d, OUT)
    out = json.load(open(f)) if os.path.exists(f) else {}
    if out.get('config') not in (None, cfg):
        raise RuntimeError(f'{f} was written with another recipe: {out["config"]}')
    out.update({'method': 'GNN-GD: gd_baseline.gd_optimize started at the GNN argmax output, '
                          f'recipe of {RECIPE_RUN}/gd_baseline_20dB.json with iters={ITERS} '
                          '(Opt-GNN budget of arXiv:2606.23141), not re-tuned',
                'config': cfg, 'tau0': 1.0, 'objective_and_score': 'ls', 'estimator': LS_NOTE,
                'n_channels': int(Hev.shape[0]), 'channels': 'test set [4096:6144]',
                'K': K, 'bits': bits, 'run': os.path.basename(d), 'written_by': MARK})
    out.setdefault('snr_db', SNR)
    for key in ('gnn', 'gnn_gd', 'improved_frac', 'seconds', 'trace_hard_mean'):
        out.setdefault(key, [None] * len(SNR))
    out['trace_every'] = TRACE_EVERY

    y0n = normalize_power(y0, PT)
    for i, x in enumerate(SNR):
        with torch.no_grad():
            start = torch.cat([sumrate_bussgang_ls(y0n[b:b + 512], Hev[b:b + 512], sev[b:b + 512], nv(x))
                               for b in range(0, Hev.shape[0], 512)])
        if abs(start.mean().item() - stored['gnn'][i]) >= 1e-3:
            print(f'  K={K} b={bits} {x:+.0f} dB: MISMATCH, GNN start {start.mean().item():.4f} vs stored '
                  f'{stored["gnn"][i]:.4f}; run left untouched', flush=True)
            return 'MISMATCH'
        if out['gnn_gd'][i] is not None:
            continue
        t0 = time.time()
        best, traces = [], []
        for b in range(0, Hev.shape[0], chunk):
            Hb, sb = Hev[b:b + chunk], sev[b:b + chunk]
            _, rb, tr = gd_optimize(Hb, sb, levels, nv(x), y0[b:b + chunk], rate='ls', **cfg)
            best.append(rb)
            traces.append((Hb.shape[0], tr[::TRACE_EVERY] + ([tr[-1]] if (len(tr) - 1) % TRACE_EVERY else [])))
        best = torch.cat(best)
        n = sum(w for w, _ in traces)
        out['gnn'][i] = start.mean().item()
        out['gnn_gd'][i] = best.mean().item()
        out['improved_frac'][i] = (best > start + 1e-6).float().mean().item()
        out['seconds'][i] = round(time.time() - t0, 1)
        out['trace_hard_mean'][i] = [sum(w * t[j] for w, t in traces) / n for j in range(len(traces[0][1]))]
        out['updated_at'] = f'{datetime.now():%Y-%m-%d %H:%M:%S}'
        json.dump(out, open(f, 'w'), indent=1)
        print(f'  K={K} b={bits} {x:+5.0f} dB: GNN {out["gnn"][i]:.4f} -> GNN-GD {out["gnn_gd"][i]:.4f} '
              f'(+{out["gnn_gd"][i] - out["gnn"][i]:.4f}, improved {100 * out["improved_frac"][i]:.1f}% '
              f'of channels)  {out["seconds"][i]:.0f}s', flush=True)
    out['gnn_gd_at_20dB'] = out['gnn_gd'][SNR.index(REF_SNR)]
    json.dump(out, open(f, 'w'), indent=1)
    torch.cuda.empty_cache()
    return 'OK'


def assemble():
    fj = os.path.join(SWEEP_DIR, 'figures_ls', 'results_for_figures.json')
    data = json.load(open(fj))
    cfg = recipe()
    done = {}
    for d, kind in all_runs():
        f = os.path.join(d, OUT)
        if os.path.exists(f):
            g = json.load(open(f))
            if None not in g['gnn_gd'] and g['config'] == cfg:
                done[(g['K'], g['bits'])] = g
    missing = []
    for c in data['configs']:
        g = done.get((c['K'], c['bits']))
        if g is None:
            missing.append((c['K'], c['bits']))
            continue
        assert max(abs(a - b) for a, b in zip(g['gnn'], c['gnn'])) < 1e-3, (c['K'], c['bits'])
        c['gnn_gd'] = g['gnn_gd']
        c['gnn_gd_ref'] = g['gnn_gd'][data['snr_db'].index(data['snr_ref_db'])]
    if missing:
        raise SystemExit(f'not all runs refined yet, missing (K, b) = {missing}; json left untouched')
    data['gnn_gd_method'] = {
        'what': "'gnn_gd': GNN output refined per channel by gd_baseline.gd_optimize, at every SNR "
                'of snr_db; start point = the GNN argmax output, so gnn_gd >= gnn channel by channel',
        'recipe': f'{RECIPE_RUN}/gd_baseline_20dB.json (gd_from_gnn) with iters={ITERS} '
                  '(Opt-GNN budget of arXiv:2606.23141), not re-tuned per (K, b)',
        'config': cfg, 'tau0': 1.0, 'objective_and_score': 'ls',
        'per_run_details': f'<run folder>/{OUT}', 'written_by': MARK}
    json.dump(data, open(fj, 'w'), indent=1)
    print(f'added gnn_gd to {len(data["configs"])} configurations of {fj}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='', help='comma-separated tags like K1b1,K6b3; default: all runs')
    ap.add_argument('--chunk', type=int, default=1024, help='channels optimized at once (results do not depend on it)')
    ap.add_argument('--assemble', action='store_true', help='merge finished runs into results_for_figures.json')
    ap.add_argument('--claim-dir', default='', help='several workers share the runs: each run is taken by the '
                    'first worker that creates <claim-dir>/<tag>; runs are visited most expensive (bits, K) first')
    a = ap.parse_args()
    if a.assemble:
        return assemble()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = recipe()
    want = {t for t in a.runs.split(',') if t}
    runs = [(d, kind) for d, kind in all_runs() if not want or tag(d, kind) in want]
    if a.claim_dir:
        os.makedirs(a.claim_dir, exist_ok=True)
        runs.sort(key=lambda r: tuple(-v for v in k_bits(*r))[::-1])
    print(f'{len(runs)} runs, recipe {cfg} (tau0 1.0, LS), SNR {SNR[0]:g}:{SNR[1] - SNR[0]:g}:{SNR[-1]:g} dB, '
          f'device {dev}', flush=True)
    cache, flags = {}, {}
    for d, kind in runs:
        t = tag(d, kind)
        if a.claim_dir:
            try:
                os.mkdir(os.path.join(a.claim_dir, t))
            except FileExistsError:
                continue
        t0 = time.time()
        flags[t] = refine_run(d, kind, dev, cfg, a.chunk, cache)
        print(f'{t}: {flags[t]}  [{time.time() - t0:.0f}s]', flush=True)
    print('ALL DONE', flags, flush=True)


if __name__ == '__main__':
    main()
