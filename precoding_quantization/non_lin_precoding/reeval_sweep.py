"""Re-score the finished runs in stored_models_cellfree_sweep_K_bits with the unbiased
least-squares Bussgang estimator, correcting every file of each run folder in place.

train_sweep.py (and train_cellfree_best.py, which produced the K=1 run) scored everything with
G = (1/Ns) sum y s^H, which assumes the sample covariance of the 125 symbols is exactly I and so
caps every SINDR near Ns (lower for K>1); see exp_cellfree_ablation.sumrate_bussgang_ls. The
trained weights are unaffected -- only the numbers are -- so this re-scores the saved
checkpoints; nothing is retrained.

Per finished run folder it rewrites, keeping every number it replaces under 'biased_estimator':
  snr_curves.json   GNN, quantized and unquantized linear precoder (ZF, or MRT for K=1) and
                    coordinate descent on -30:30:5 dB. A CD iterate does not depend on the SNR,
                    only the choice among them does, so every target scale kappa is run once and
                    its iterates after 0, 1, 2, 4, 8 (K=1: also 15) sweeps are scored; (kappa,
                    sweeps) is then re-selected per SNR on the disjoint tuning slice. 0 sweeps is
                    the start point, the quantized linear precoder, so the tuned search can no
                    longer end below it -- with a fixed 8 sweeps it did, by up to 1.7 bit at
                    K=4, b=3 below 10 dB, because a target of kappa times the ZF receive signal is
                    a poor objective there. The kappa grid runs to 100, where the search is in
                    effect maximizing correlation with the target (the kappa -> inf limit); the
                    original 0.5..0.9 grid was hit at its edge by 3 of 7 sweep runs at 20 dB and
                    at every SNR of the K=1 run.
  references.json   the same numbers at the training SNR (20 dB).
  history.json      'references' and 'best_eval_rate' replaced, plus the LS rate of the last
                    epoch. The per-epoch values in 'history' stay as logged: only the best and
                    the last checkpoints are stored, so the epochs in between cannot be re-scored.
  rate_vs_snr.pdf, rate_vs_snr.tex   redrawn from the corrected curves.
  train.log, DONE   one line appended; earlier lines are left as they were written.
  The binary checkpoints are not touched (their 'eval_rate'/'best' fields remain biased).
A folder counts as corrected once snr_curves.json carries corrected_by == this script; --force
redoes it, reading the biased numbers back from the 'biased_estimator' copy.

Consistency check: the GNN re-scored with the BIASED estimator at 20 dB must reproduce the value
stored at training time, which proves the same weights and channels were used.

Finally it assembles figures_ls/results_for_figures.json and redraws the two sweep figures.

  CUDA_VISIBLE_DEVICES=0 python reeval_sweep.py            # skips folders already corrected
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

import exp_cellfree_ablation as k1  # noqa: E402  (model, forward and CD of the K=1 run)
from exp_cellfree_ablation import GNNv2, sumrate_bussgang, sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import (load_data, load_levels, linear_quantized, linear_unquantized,  # noqa: E402
                         coord_descent, run_model_warm, EVAL, CD_TUNE, PT, M_FULL)
from plot_sweep_figures import fig_single_run, fig_rate_vs_snr, fig_rate_vs_bits, preview  # noqa: E402

MARK = 'reeval_sweep.py'
LS_NOTE = 'unbiased least-squares Bussgang estimator, G = (sum y s^H)(sum s s^H)^-1 (see eq:tp_G_hat)'
BIASED_NOTE = ('G = (1/Ns) sum y s^H, as computed at training time; caps the SINDR near Ns because '
               'the sample power of s is not exactly 1. Kept only for traceability, do not plot.')
SNR = [float(x) for x in range(-30, 31, 5)]
REF_SNR = 20.0
KAPPAS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0,
          3.0, 5.0, 10.0, 100.0)
FLIPS_K1 = (0.0, 0.1, 0.2, 0.3)
SNAPS, SNAPS_K1 = (0, 1, 2, 4, 8), (0, 1, 2, 4, 8, 15)


def nv(snr_db):
    return PT / 10 ** (snr_db / 10)


def per_channel(y, H, s, fn=sumrate_bussgang_ls):
    """Rate of every channel at every SNR of the grid: len(SNR) x n (y power-normalized)."""
    return torch.stack([fn(y, H, s, nv(x)) for x in SNR])


def curve(y, H, s, fn=sumrate_bussgang_ls, chunk=512):
    """Mean rate at every SNR of the grid (y already power-normalized)."""
    return torch.cat([per_channel(y[b:b + chunk], H[b:b + chunk], s[b:b + chunk], fn)
                      for b in range(0, H.shape[0], chunk)], 1).mean(1).tolist()


def cd_curve(solve, run_cfgs, snaps, Ht, st, Hev, sev, chunk):
    """Coordinate descent with (config, sweeps) re-selected at every SNR on the tuning slice.

    solve(H, s, cfg) returns the iterates after each of `snaps` sweeps, so every config is run
    once. Returns the eval-slice rate per SNR and the chosen (cfg, sweeps) per SNR."""
    tune, ev = {}, {}
    for c in run_cfgs:
        for n, y in zip(snaps, solve(Ht, st, c)):
            tune[(c, n)] = per_channel(normalize_power(y, PT), Ht, st).mean(1)
        acc = [[] for _ in snaps]
        for b in range(0, Hev.shape[0], chunk):
            Hb, sb = Hev[b:b + chunk], sev[b:b + chunk]
            for j, y in enumerate(solve(Hb, sb, c)):
                acc[j].append(per_channel(normalize_power(y, PT), Hb, sb))
        for j, n in enumerate(snaps):
            ev[(c, n)] = torch.cat(acc[j], 1).mean(1)
    keys = list(tune)
    best = [max(keys, key=lambda k: tune[k][i].item()) for i in range(len(SNR))]
    return [ev[k][i].item() for i, k in enumerate(best)], best


def run_kind(d):
    b = os.path.basename(d)
    if b.startswith('M40_K') and os.path.exists(os.path.join(d, 'DONE')):
        return 'sweep'                               # train_sweep.py
    if b.startswith('M_40_K_1_') and os.path.exists(os.path.join(d, 'snr_curves.json')):
        return 'k1'                                  # train_cellfree_best.py, finished
    return None


def is_corrected(d):
    f = os.path.join(d, 'snr_curves.json')
    return os.path.exists(f) and json.load(open(f)).get('corrected_by') == MARK


def biased_copy(stored):
    """The biased numbers of a stored json, whether or not it was corrected before."""
    if 'biased_estimator' in stored:
        b = dict(stored['biased_estimator'])
        b.setdefault('snr_db', stored.get('snr_db'))   # K=1 file: biased grid == the old grid
        return b
    return dict(stored)


def correct_run(d, kind, dev, cache, native=False):
    """Evaluate one finished run folder with the LS estimator and (re)write its files.

    native=False: a run scored with the biased estimator at training time; its numbers are
    replaced and kept under 'biased_estimator', after checking the biased re-score reproduces
    what was stored. native=True: a run already evaluated with the LS estimator during training
    (train_sweep.py calls this as its final evaluation); the check is then that the LS re-score of
    model_best.pt reproduces the logged best eval rate, and there is nothing biased to keep."""
    t0 = time.time()
    torch.manual_seed(0)
    hist = json.load(open(os.path.join(d, 'history.json')))
    ck = torch.load(os.path.join(d, 'model_best.pt'), map_location=dev, weights_only=False)
    c = ck['cfg']
    if kind == 'sweep':
        K, bits = hist['config']['K'], hist['config']['bits']
        if cache.get('K') != (K, 'sweep'):
            _, _, Hte, Ste = load_data(K, 1000, dev)
            cache.update(K=(K, 'sweep'), data=(Hte[EVAL], Ste[EVAL], Hte[CD_TUNE], Ste[CD_TUNE]))
        levels = load_levels(bits, dev)
        warm = not c.get('no_warm_start', False)
        m = GNNv2(M_FULL, K, c['dl'], c['layers'], bits, c['tau_final'], levels, input_mode='polar4',
                  feat_stats=ck['feat_stats'], logit_norm=c.get('logit_norm')).to(dev)
        forward = lambda H, s: run_model_warm(m, H, s, levels, PT, 16, warm)  # noqa: E731
        lin_q = lambda H, s: linear_quantized(H, s, levels, PT)              # noqa: E731
        lin_u = lambda H, s: linear_unquantized(H, s, PT)                    # noqa: E731
        solve = lambda H, s, kap: coord_descent(H, s, levels, PT, kap, max(SNAPS), SNAPS)  # noqa: E731
        cfgs, snaps, cd_chunk = list(KAPPAS), SNAPS, 128
        lin, LQ, LU = ('ZF' if K > 1 else 'MRT'), 'linear_quantized', 'linear_unquantized'
    else:
        K, bits = 1, 1
        if cache.get('K') != (1, 'k1'):
            _, _, Hte, Ste = k1.load_data(dev, 1000)
            cache.update(K=(1, 'k1'), data=(Hte[EVAL], Ste[EVAL], Hte[k1.CD_TUNE], Ste[k1.CD_TUNE]))
        levels = (np.sqrt(PT / (2 * M_FULL)) * torch.tensor([-1.0, 1.0])).to(dev)
        m = GNNv2(M_FULL, 1, c['dl'], c['layers'], 1, c['tau_final'], levels, input_mode='polar4',
                  feat_stats=ck['feat_stats']).to(dev)
        forward = lambda H, s: k1.run_model(m, H, s, 16, c['refine'], c['init'])  # noqa: E731
        lin_q = lambda H, s: k1.mrt_1bit(H, s, PT)                                 # noqa: E731
        lin_u = lambda H, s: torch.conj(H) @ s                                     # noqa: E731
        solve = lambda H, s, cf: k1.coord_descent(H, s, PT, cf[0], max(SNAPS_K1), cf[1],  # noqa: E731
                                                  snapshots=SNAPS_K1)
        cfgs, snaps, cd_chunk = [(kap, f) for kap in KAPPAS for f in FLIPS_K1], SNAPS_K1, 512
        lin, LQ, LU = 'MRT', 'mrt_1bit', 'mrt_unquantized'
    Hev, sev, Ht, st = cache['data']
    i20 = SNR.index(REF_SNR)

    def gnn_output(state):
        m.load_state_dict(state)
        m.eval()
        m.output_type = 'argmax'
        with torch.no_grad():
            return normalize_power(torch.cat([forward(Hev[b:b + 256], sev[b:b + 256])
                                              for b in range(0, Hev.shape[0], 256)]), PT)

    with torch.no_grad():
        y_gnn = gnn_output(ck['model'])
        y_last = gnn_output(torch.load(os.path.join(d, 'checkpoint.pt'), map_location=dev,
                                       weights_only=False)['model'])
        y_lq = normalize_power(lin_q(Hev, sev), PT)
        y_lu = normalize_power(lin_u(Hev, sev), PT)

        if native:
            b_curves, stored_ref = None, hist['best_eval_rate']
            check_ref = curve(y_gnn, Hev, sev)[i20]
        else:
            b_curves = biased_copy(json.load(open(os.path.join(d, 'snr_curves.json'))))
            stored_ref = b_curves['gnn'][b_curves['snr_db'].index(REF_SNR)]
            check_ref = curve(y_gnn, Hev, sev, fn=sumrate_bussgang)[i20]

        new = {'snr_db': SNR, 'estimator': LS_NOTE, 'corrected_by': MARK,
               'corrected_at': f'{datetime.now():%Y-%m-%d %H:%M:%S}',
               'gnn': curve(y_gnn, Hev, sev), LQ: curve(y_lq, Hev, sev), LU: curve(y_lu, Hev, sev)}
        if kind == 'k1':
            new['mrt_unquantized_closed_form'] = [
                torch.log2(1 + PT * (Hev.abs() ** 2).sum((1, 2)) / nv(x)).mean().item() for x in SNR]
        if abs(check_ref - stored_ref) >= 1e-3:      # not the stored model/channels: write nothing
            print(f'  {os.path.basename(d)}: MISMATCH, re-score {check_ref:.4f} vs stored '
                  f'{stored_ref:.4f}; folder left untouched', flush=True)
            return 'MISMATCH'
        cd, cd_best = cd_curve(solve, cfgs, snaps, Ht, st, Hev, sev, cd_chunk)
        gnn_last = curve(y_last, Hev, sev)[i20]
    cd_cfg = [(list(c) if isinstance(c, tuple) else [c]) + [n] for c, n in cd_best]   # [kappa, (flip,) sweeps]
    # kappa=100 is the kappa->inf limit, not an edge; with 0 sweeps kappa is irrelevant (the start point)
    edge = [x for x, cf in zip(SNR, cd_cfg) if cf[0] == min(KAPPAS) and cf[-1] > 0]
    new.update({'coord_descent': cd, 'coord_descent_cfg': cd_cfg,
                'coord_descent_cfg_is': '[kappa, flip, sweeps]' if kind == 'k1' else '[kappa, sweeps]',
                'coord_descent_kappa_grid': list(KAPPAS), 'coord_descent_sweeps_grid': list(snaps),
                'coord_descent_at_grid_edge_db': edge,
                'coord_descent_at_20dB': cd[i20], 'gnn_last_epoch_at_20dB': gnn_last,
                'check': {'estimator': 'ls' if native else 'biased', 'gnn_recomputed_20dB': check_ref,
                          'gnn_stored_20dB': stored_ref, 'abs_diff': abs(check_ref - stored_ref)},
                'K': K, 'bits': bits, 'run': os.path.basename(d)})
    if b_curves is not None:
        b_curves['note'] = BIASED_NOTE
        new['biased_estimator'] = b_curves
    json.dump(new, open(os.path.join(d, 'snr_curves.json'), 'w'), indent=1)

    # references.json, same keys as at training time, values at 20 dB
    old_ref = json.load(open(os.path.join(d, 'references.json')))
    ref = {LQ: new[LQ][i20], LU: new[LU][i20], 'coord_descent': cd[i20]}
    ref.update(cd_kappa=cd_cfg[i20][0], cd_sweeps=cd_cfg[i20][-1])
    if kind == 'k1':
        ref.update(cd_flip=cd_cfg[i20][1])
    ref.update(snr_db=REF_SNR, estimator=LS_NOTE, corrected_by=MARK)
    if not native:
        ref.update(biased_estimator=old_ref.get('biased_estimator', old_ref))
    json.dump(ref, open(os.path.join(d, 'references.json'), 'w'), indent=1)

    # history.json: references and headline numbers; per-epoch entries stay as logged
    if native:
        note = ("All eval rates, per epoch included, use the LS estimator; 'train_rate' is the "
                f"training objective itself (loss = {hist['config'].get('loss', 'biased')} estimator).")
    else:
        hist['best_eval_rate_biased'] = hist.get('best_eval_rate_biased', hist['best_eval_rate'])
        note = ("'best_eval_rate' and 'last_epoch_eval_rate' are LS re-scores of "
                'model_best.pt and checkpoint.pt at 20 dB. The per-epoch entries of '
                "'history' are the BIASED values logged during training (train_rate "
                'is the training objective itself); only the best and the last '
                'checkpoints are stored, so the epochs in between cannot be re-scored.')
    hist.update(references=ref, best_eval_rate=new['gnn'][i20], last_epoch_eval_rate=gnn_last,
                estimator=LS_NOTE, corrected_by=MARK, estimator_note=note)
    json.dump(hist, open(os.path.join(d, 'history.json'), 'w'), indent=1)

    # per-run figure
    title = f'cell-free, $M={M_FULL}$, $K={K}$, $b={bits}$'
    fig_single_run(SNR, {'gnn': new['gnn'], 'cd': cd, 'lin_q': new[LQ], 'lin_unq': new[LU]}, title, lin, bits,
                   os.path.join(d, 'rate_vs_snr.tex'), os.path.join(d, 'rate_vs_snr.pdf'), [
                       f'Sum rate vs SNR, cell-free, M = {M_FULL} APs, K = {K} UEs, b = {bits} bit(s).',
                       f'Source: {os.path.basename(d)}/snr_curves.json (written by {MARK}).',
                       'All curves: the same 2048 held-out channels (test set [4096:6144]), 125 symbols each,',
                       'per-channel E{||y||^2} = P_t, rates with the ' + LS_NOTE + '.',
                       'GNN: best-held-out checkpoint, argmax decisions, trained at 20 dB.',
                       'Coordinate descent: local search implemented for this work (not a published baseline);',
                       'its target scale and number of sweeps (0 = its start point, the quantized linear',
                       'precoder) are re-selected at every SNR on a disjoint tuning slice.',
                       f'{lin} + DAC: the quantized precoder that warm-starts the GNN; {lin} unquantized: no DAC.'])

    what = 'final evaluation' if native else 're-scored'
    tail = '' if native else ' Numbers above this line are biased.'
    for f, line in (('train.log', f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {MARK}: {what} with the LS "
                                  f"Bussgang estimator; @20dB GNN {new['gnn'][i20]:.3f} (last epoch "
                                  f"{gnn_last:.3f}) | {lin}+{bits}b {new[LQ][i20]:.3f} | CD {cd[i20]:.3f} | "
                                  f"unquantized {new[LU][i20]:.3f}.{tail}"),
                    ('DONE', f"best_eval_ls {new['gnn'][i20]:.4f} ({MARK})")):
        if os.path.exists(os.path.join(d, f)) or f == 'train.log':
            open(os.path.join(d, f), 'a').write(line + '\n')
    old_ls = os.path.join(d, 'snr_curves_ls.json')        # superseded by the corrected snr_curves.json
    if os.path.exists(old_ls):
        os.remove(old_ls)

    flag = 'OK' if abs(check_ref - stored_ref) < 1e-3 else 'MISMATCH'
    print(f'  K={K} b={bits}: @20dB GNN {new["gnn"][i20]:.3f} (last ep {gnn_last:.3f}) | {lin}+{bits}b '
          f'{new[LQ][i20]:.3f} | CD {cd[i20]:.3f} (cfg {cd_cfg[i20]}) | unq {new[LU][i20]:.3f}   '
          f'[check {"LS" if native else "biased"} {check_ref:.4f} vs stored {stored_ref:.4f}: {flag}]'
          f'{"  CD cfg AT GRID EDGE at " + str(edge) + " dB" if edge else ""}  {time.time() - t0:.0f}s',
          flush=True)
    print('      CD [kappa, (flip,) sweeps] per SNR: ' + ' '.join(
        f'{x:g}:{"/".join(f"{v:g}" for v in cf)}' for x, cf in zip(SNR, cd_cfg)),
          flush=True)
    del m
    torch.cuda.empty_cache()
    return flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', default=os.path.join(CURRENT_DIR, 'stored_models_cellfree_sweep_K_bits'))
    ap.add_argument('--force', action='store_true', help='redo folders that are already corrected')
    ap.add_argument('--xmin', type=float, default=-10, help='left end of the sweep figure A')
    ap.add_argument('--no-sweep-figures', action='store_true',
                    help='only correct the run folders; for directories of short ablation runs, where '
                         'several runs share a (K, b) and the sweep figures make no sense')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs = sorted((d, run_kind(d)) for d in glob.glob(os.path.join(a.sweep_dir, 'M*')) if run_kind(d))
    print(f'{len(runs)} finished runs, SNR grid {SNR[0]:g}:{SNR[1] - SNR[0]:g}:{SNR[-1]:g} dB, device {dev}',
          flush=True)
    cache, flags = {}, []
    for d, kind in runs:
        if is_corrected(d) and not a.force:
            print(f'  {os.path.basename(d)}: already corrected, skipping', flush=True)
            continue
        flags.append(correct_run(d, kind, dev, cache))

    if a.no_sweep_figures:
        if not all(f == 'OK' for f in flags):
            print('   !! a consistency check FAILED, see above', flush=True)
        return
    # assemble the sweep figures from the corrected folders. K=1, b=1 is the train_cellfree_best.py
    # run (20 epochs, 200000 channels, constant lr), every other (K, b) a train_sweep.py run; 'recipe'
    # says which. For K=1, ZF (h*/||h||^2) is MRT after power normalization, so 'zf_*' is MRT there.
    cfgs = []
    for d, kind in runs:
        if is_corrected(d):
            c = json.load(open(os.path.join(d, 'snr_curves.json')))
            LQ, LU = ('mrt_1bit', 'mrt_unquantized') if kind == 'k1' else ('linear_quantized', 'linear_unquantized')
            cfgs.append({'K': c['K'], 'bits': c['bits'], 'gnn': c['gnn'], 'zf_q': c[LQ],
                         'zf_unq': c[LU], 'cd': c['coord_descent'],
                         'cd_ref': c['coord_descent_at_20dB'],
                         'recipe': 'train_cellfree_best.py' if kind == 'k1' else 'train_sweep.py',
                         'run': os.path.basename(d)})
            gd = os.path.join(d, 'gnn_gd_curves.json')        # GNN-GD, written by gd_refine_sweep.py
            if os.path.exists(gd) and None not in json.load(open(gd))['gnn_gd']:
                g = json.load(open(gd))['gnn_gd']
                cfgs[-1].update(gnn_gd=g, gnn_gd_ref=g[SNR.index(REF_SNR)])
    cfgs.sort(key=lambda c: (c['K'], c['bits']))
    out_dir = os.path.join(a.sweep_dir, 'figures_ls')
    os.makedirs(out_dir, exist_ok=True)
    data = {'snr_db': SNR, 'snr_ref_db': REF_SNR, 'panels_K': [1, 2, 4, 6], 'panels_bits': [1, 2, 3],
            'estimator': LS_NOTE, 'configs': cfgs}
    prev = os.path.join(out_dir, 'results_for_figures.json')
    if any('gnn_gd' in c for c in cfgs) and os.path.exists(prev):
        m = json.load(open(prev)).get('gnn_gd_method')
        if m:
            data['gnn_gd_method'] = m
    json.dump(data, open(os.path.join(out_dir, 'results_for_figures.json'), 'w'), indent=1)
    fig_rate_vs_snr(data, os.path.join(out_dir, 'Rsum_sweep_vs_snr.tex'), a.xmin, '')
    fig_rate_vs_bits(data, os.path.join(out_dir, 'Rsum_sweep_vs_bits.tex'), '')
    preview(data, os.path.join(out_dir, 'preview_both_figures.png'), a.xmin, '')
    print(f'sweep figures redrawn from {len(cfgs)} configurations -> {out_dir}'
          + ('' if all(f == 'OK' for f in flags) else '   !! a consistency check FAILED, see above'),
          flush=True)


if __name__ == '__main__':
    main()
