"""Continue training the finished K x bits sweep runs, the way the K=1, 1-bit run was continued
(continue_k1_training.py, cosine branch: 5.010 -> 5.187 at 20 dB (LS), converged; the constant-lr
branch only reached 5.067). In that run all of the gain came once the lr had fallen below ~4e-4:
epochs 21-31 at 1e-3 .. 5e-4 stayed on the 4.87-5.07 plateau.

Sources: stored_models_cellfree_sweep_K_bits/M40_K{K}_b{b}_ep15_* (train_sweep.py, 15 epochs, lr
1e-3 -> 5e-5 cosine); K=1 b=1 is excluded, it was trained by train_cellfree_best.py and has been
continued already. The source folders are only read. Each run continues from its checkpoint.pt
(weights + Adam state after epoch 15) with EXACTLY its own recipe:
  the same first n_train training channels, the same TRAINING loss (biased Bussgang form, as in
  epochs 1-15), the same logit L2 weight (0.1 for K>=2, 0 for K=1), applied as train_sweep.py did
  (to model.last_logits, i.e. the last chan_chunk of each micro-batch), Gumbel-softmax hard output
  with tau held at its final 0.25, the same quantized-linear warm start, 128 channels per step in
  4 accumulation steps, chan_chunk 16,
and ONE change, the learning rate: cosine --peak-lr 2e-4 -> --final-lr 1e-5 over the extra epochs.
Unlike the K=1 source, which had been trained at a constant 1e-3 (so restarting there continued its
own lr), these runs had already decayed 1e-3 -> 5e-5 and were still improving slowly at the end
(+0.002..+0.06 per epoch over the last 3). Jumping back to 1e-3 would undo that convergence -- a
first attempt did, the training objective of K=6 fell from 20.49 to ~19.75 within a third of an
epoch -- and would spend half the budget re-climbing. 2e-4 is inside the lr range where the K=1
continuation gained, and close to where these runs were around epoch 11-12. --extra-epochs 15
doubles each run's budget, as the K=1 continuation did (20 -> 40).

Every rate is computed with the LS Bussgang estimator (sumrate_bussgang_ls). Every epoch, and the
start point first, is scored at 20 dB on the held-out channels (test set [4096:6144]):
  eval_rate_ls          argmax output -- the deployed rule; picks model_best.pt
  eval_rate_sampled_ls  Gumbel-sampled output (diagnostic: the argmax-minus-sampled gap)
  eval_rate_biased      argmax output, biased estimator: comparable with the logged epochs 1-15 of
                        runs trained before the estimator fix
The start point must reproduce the source's logged last epoch (biased for runs trained before the
fix, LS for the K=1 b=2/3 runs, which were evaluated with LS natively). At the end, with the best
epoch (the start point included, so a continued run can never report less than it started with):
  snr_curves.json     GNN on -30:30:5 dB (LS); the source's quantized / unquantized linear curves;
                      the source's GNN and GNN-GD curves ('*_before_continuation') for reference
  gnn_gd_curves.json  GNN-GD recomputed from THIS model's output: gd_refine_sweep.refine_run, the
                      same recipe as the sweep's GNN-GD (200 Adam steps on relaxed logits, every SNR)
  rate_vs_snr.pdf/.tex  continued GNN, GNN before continuation, its GNN-GD, linear baselines
  DONE                written last
Runs are resumable (re-run the same command); workers share the runs through --claim-dir, taking
them longest first.

  CUDA_VISIBLE_DEVICES=0 python continue_sweep_training.py --claim-dir exp_results/continue_claims
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import GNNv2, RATE_FNS, sumrate_bussgang, sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import load_data, load_levels, run_model_warm, EVAL, PT, M_FULL, SNR_TRAIN_DB  # noqa: E402
from reeval_sweep import SNR, REF_SNR, LS_NOTE, curve  # noqa: E402
from plot_sweep_figures import fig_single_run  # noqa: E402

MARK = 'continue_sweep_training.py'
SRC_DIR = os.path.join(CURRENT_DIR, 'stored_models_cellfree_sweep_K_bits')
NV20 = PT / 10 ** (SNR_TRAIN_DB / 10)
ORDER = [(6, 3), (6, 2), (6, 1), (4, 3), (4, 2), (4, 1), (2, 3), (2, 2), (2, 1), (1, 3), (1, 2)]  # longest first


def source_run(K, bits):
    ds = [d for d in glob.glob(os.path.join(SRC_DIR, f'M40_K{K}_b{bits}_ep15_*'))
          if os.path.exists(os.path.join(d, 'DONE'))]
    assert len(ds) == 1, (K, bits, ds)
    return ds[0]


@torch.no_grad()
def score(model, Hev, sev, levels, warm, chunk):
    """(LS argmax, biased argmax, LS Gumbel-sampled) at 20 dB."""
    prev = model.output_type
    out = []
    for mode in ('argmax', 'gumbel_softmax_hard'):
        model.output_type = mode
        ls, bi = [], []
        for b in range(0, Hev.shape[0], 512):
            Hb, sb = Hev[b:b + 512], sev[b:b + 512]
            y = normalize_power(run_model_warm(model, Hb, sb, levels, PT, chunk, warm), PT)
            ls.append(sumrate_bussgang_ls(y, Hb, sb, NV20))
            if mode == 'argmax':
                bi.append(sumrate_bussgang(y, Hb, sb, NV20))
        out.append(torch.cat(ls).mean().item())
        if bi:
            out.append(torch.cat(bi).mean().item())
    model.output_type = prev
    return out[0], out[1], out[2]


def continue_run(src, extra, out_root, dev, peak_lr, final_lr, n_train_override=None):
    c = dict(json.load(open(os.path.join(src, 'history.json')))['config'])
    K, bits = c['K'], c['bits']
    src_ck = torch.load(os.path.join(src, 'checkpoint.pt'), map_location=dev, weights_only=False)
    src_best = torch.load(os.path.join(src, 'model_best.pt'), map_location=dev, weights_only=False)
    src_curves = json.load(open(os.path.join(src, 'snr_curves.json')))
    src_gd = json.load(open(os.path.join(src, 'gnn_gd_curves.json')))
    assert src_curves['snr_db'] == SNR and src_gd['snr_db'] == SNR
    native = 'biased_estimator' not in src_curves      # evaluated with LS during its own training
    first = src_ck['epoch'] + 1
    last = first + extra
    run_dir = os.path.join(out_root, f'{os.path.basename(src)}_ep{first}to{last}_cos{peak_lr:g}to{final_lr:g}')
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'DONE')):
        print(f'[skip] {run_dir} already finished', flush=True)
        return 'OK'
    ckpt_path, best_path = os.path.join(run_dir, 'checkpoint.pt'), os.path.join(run_dir, 'model_best.pt')
    logf = open(os.path.join(run_dir, 'train.log'), 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()

    i20 = SNR.index(REF_SNR)
    before, gd_before = src_curves['gnn'][i20], src_gd['gnn_gd'][i20]
    log(f"\n===== K={K} b={bits} {datetime.now():%Y-%m-%d %H:%M:%S} {torch.cuda.get_device_name(0)} =====")
    log(f'continuing {src}\n  from its checkpoint.pt (after epoch {first}), epochs {first + 1}..{last}, '
        f'lr cosine {peak_lr:g} -> {final_lr:g} (source ended at {c["lr_min"]:g}); source recipe {c}')
    log(f'targets @20dB (LS): source model_best {before:.4f} | its GNN-GD {gd_before:.4f}')

    Htr, Str, Hte, Ste = load_data(K, n_train_override or c['n_train'], dev)
    Hev, sev = Hte[EVAL], Ste[EVAL]
    levels = load_levels(bits, dev)
    warm = not c.get('no_warm_start', False)
    loss_fn = RATE_FNS[c.get('loss', 'biased')]
    model = GNNv2(M_FULL, K, c['dl'], c['layers'], bits, c['tau_final'], levels, input_mode='polar4',
                  feat_stats=src_best['feat_stats'], logit_norm=c.get('logit_norm')).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    bs, acc = c['batch_channels'], c['accum_steps']
    micro, nb = bs // acc, Htr.shape[0] // bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=extra * nb, eta_min=final_lr)
    cfg_out = dict(c, epochs=last, continued_from=src, lr_schedule=f'cosine {peak_lr:g} -> {final_lr:g}',
                   n_train=n_train_override or c['n_train'])

    def save_best(ep, ls, bi):
        torch.save({'model': model.state_dict(), 'epoch': ep, 'eval_rate': ls, 'eval_rate_biased': bi,
                    'estimator': 'ls', 'cfg': cfg_out, 'feat_stats': src_best['feat_stats']}, best_path)

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start, hist, best = ck['epoch'] + 1, ck['hist'], ck['best']
        log(f'resumed this continuation at epoch {start + 1} (best LS so far {best:.4f})')
    else:
        model.load_state_dict(src_ck['model'])
        opt.load_state_dict(src_ck['opt'])
        for g in opt.param_groups:                 # the source ended at lr_min; start the new cosine at peak_lr
            g['lr'] = peak_lr
            g['initial_lr'] = peak_lr
        ls, bi, smp = score(model, Hev, sev, levels, warm, c['chan_chunk'])
        logged = src_ck['hist'][-1]['eval_rate']
        chk = ls if native else bi
        log(f'start point (epoch {first}): eval LS {ls:.4f} | sampled {smp:.4f} | biased {bi:.4f}   '
            f'[source logged {logged:.4f} ({"LS" if native else "biased"}) for this epoch]')
        assert abs(chk - logged) < 1e-3, 'not the stored weights/channels'
        start, best = first, ls
        hist = [dict(h, source='source run, eval ' + ('LS' if native else 'biased')) for h in src_ck['hist']]
        hist.append({'epoch': first - 1, 'eval_rate_ls': ls, 'eval_rate_biased': bi,
                     'eval_rate_sampled_ls': smp, 'source': 'start point'})
        save_best(first - 1, ls, bi)

    torch.manual_seed(1000 + start)                # epochs 1-15 used seed 0; do not replay its order
    t0 = time.time()
    for ep in range(start, last):
        model.tau = c['tau_final']
        model.output_type = 'gumbel_softmax_hard'
        model.train()
        perm = torch.randperm(Htr.shape[0], device=dev)
        run = 0.0
        bar = tqdm(range(nb), unit='batch', dynamic_ncols=True, desc=f'K{K} b{bits} ep{ep + 1}/{last}')
        for i in bar:
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad()
            for a in range(acc):
                sub = idx[a * micro:(a + 1) * micro]
                Hb, sb = Htr[sub], Str[sub]
                y = normalize_power(run_model_warm(model, Hb, sb, levels, PT, c['chan_chunk'], warm), PT)
                rate_loss = -loss_fn(y, Hb, sb, NV20).mean() / acc
                loss = rate_loss
                if c['logit_l2']:
                    loss = loss + c['logit_l2'] * (model.last_logits ** 2).mean() / acc
                loss.backward()
                run += rate_loss.item()
            opt.step()
            sched.step()
            if (i + 1) % 20 == 0:
                bar.set_postfix(rate=f'{-run / (i + 1):.3f}', lr=f'{opt.param_groups[0]["lr"]:.1e}')
        ls, bi, smp = score(model, Hev, sev, levels, warm, c['chan_chunk'])
        lr_now = opt.param_groups[0]['lr']
        hist.append({'epoch': ep, 'tau': model.tau, 'lr_end': lr_now, 'train_rate': -run / nb,
                     'eval_rate_ls': ls, 'eval_rate_sampled_ls': smp, 'eval_rate_biased': bi,
                     'elapsed_s': time.time() - t0, 'source': 'continuation'})
        mark = ''
        if ls > best:
            best, mark = ls, '  <- best (LS), saved'
            save_best(ep, ls, bi)
        log(f'K{K} b{bits} ep{ep + 1}/{last}  lr {lr_now:.2e}  train {-run / nb:.4f}  eval LS {ls:.4f} | '
            f'sampled {smp:.4f} | biased {bi:.4f}  [before {before:.3f}, {ls - before:+.3f}; GNN-GD before '
            f'{gd_before:.3f}]  {(time.time() - t0) / 60:.0f} min{mark}')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': ep, 'hist': hist, 'best': best}, ckpt_path)
        json.dump({'history': hist, 'source': src, 'config': c, 'continuation': {
                       'epochs': [first + 1, last], 'lr_schedule': cfg_out['lr_schedule'],
                       'seed': 1000 + first, 'written_by': MARK},
                   'best_eval_rate': best, 'estimator': LS_NOTE,
                   'targets_20dB_ls': {'source_model_best': before, 'source_gnn_gd': gd_before}},
                  open(os.path.join(run_dir, 'history.json'), 'w'), indent=1)

    # ---- final evaluation of the best epoch ----
    del Htr, Str
    torch.cuda.empty_cache()
    bk = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(bk['model'])
    model.output_type = 'argmax'
    with torch.no_grad():
        y = normalize_power(torch.cat([run_model_warm(model, Hev[b:b + 256], sev[b:b + 256], levels, PT,
                                                      c['chan_chunk'], warm)
                                       for b in range(0, Hev.shape[0], 256)]), PT)
        gnn = curve(y, Hev, sev)
    assert abs(gnn[i20] - bk['eval_rate']) < 1e-3, (gnn[i20], bk['eval_rate'])
    curves = {'snr_db': SNR, 'estimator': LS_NOTE, 'corrected_by': MARK, 'K': K, 'bits': bits,
              'run': os.path.basename(run_dir), 'source': src, 'best_epoch': bk['epoch'] + 1,
              'gnn': gnn, 'linear_quantized': src_curves['linear_quantized'],
              'linear_unquantized': src_curves['linear_unquantized'],
              'gnn_before_continuation': src_curves['gnn'], 'gnn_gd_before_continuation': src_gd['gnn_gd'],
              'note': "linear curves and '*_before_continuation' are copied from the source folder "
                      '(same 2048 channels, same estimator)'}
    json.dump(curves, open(os.path.join(run_dir, 'snr_curves.json'), 'w'), indent=1)
    log(f'best epoch {bk["epoch"] + 1}: GNN (LS) ' + ' '.join(f'{x:g}:{v:.3f}' for x, v in zip(SNR, gnn)))

    log('GNN-GD from this model (gd_refine_sweep.refine_run) ...')
    from gd_refine_sweep import refine_run, recipe
    flag = refine_run(run_dir, 'sweep', dev, recipe(), 1024, {})
    if flag != 'OK':
        raise RuntimeError(f'GNN-GD check failed ({flag}); DONE not written')
    gd = json.load(open(os.path.join(run_dir, 'gnn_gd_curves.json')))['gnn_gd']
    curves.update(gnn_gd=gd, gnn_gd_at_20dB=gd[i20])
    json.dump(curves, open(os.path.join(run_dir, 'snr_curves.json'), 'w'), indent=1)

    lin = 'ZF' if K > 1 else 'MRT'
    fig_single_run(SNR, {'gnn': gnn, 'gnn_before': src_curves['gnn'], 'gnn_gd': gd,
                         'lin_q': src_curves['linear_quantized'], 'lin_unq': src_curves['linear_unquantized']},
                   f'cell-free, $M={M_FULL}$, $K={K}$, $b={bits}$', lin, bits,
                   os.path.join(run_dir, 'rate_vs_snr.tex'), os.path.join(run_dir, 'rate_vs_snr.pdf'), [
                       f'Sum rate vs SNR, cell-free, M = {M_FULL} APs, K = {K} UEs, b = {bits} bit(s).',
                       f'Source: {os.path.basename(run_dir)}/snr_curves.json (written by {MARK}).',
                       'All curves: the same 2048 held-out channels (test set [4096:6144]), 125 symbols each,',
                       'per-channel E{||y||^2} = P_t, rates with the ' + LS_NOTE + '.',
                       f'GNN: best epoch ({bk["epoch"] + 1}) of the continued run, argmax decisions, trained at 20 dB;',
                       f'dashed: the same run after its first {first} epochs.',
                       'GNN-GD: the continued GNN output refined per channel and per SNR by 200 Adam steps.'],
                   labels={'gnn': f'GNN, continued (best: epoch {bk["epoch"] + 1} of {last})',
                           'gnn_before': f'GNN, first {first} epochs'})
    gain, gap = gnn[i20] - before, gnn[i20] - gd[i20]
    log(f'done @20dB (LS): GNN {before:.4f} -> {gnn[i20]:.4f} ({gain:+.4f}, best epoch {bk["epoch"] + 1}) | '
        f'GNN-GD {gd_before:.4f} -> {gd[i20]:.4f} | GNN - GNN-GD {before - gd_before:+.4f} -> {gap:+.4f}')
    open(os.path.join(run_dir, 'DONE'), 'w').write(
        f'{datetime.now():%Y-%m-%d %H:%M:%S} best_eval_ls {gnn[i20]:.4f} gnn_gd {gd[i20]:.4f}\n')
    return 'OK'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='', help='comma-separated tags like K6b3,K1b2; default: all 11, longest first')
    ap.add_argument('--extra-epochs', type=int, default=15)
    ap.add_argument('--peak-lr', type=float, default=2e-4)
    ap.add_argument('--final-lr', type=float, default=1e-5)
    ap.add_argument('--claim-dir', default='', help='workers share the runs: each is taken by the first '
                                                    'worker that creates <claim-dir>/<tag>')
    ap.add_argument('--out-root', default=os.path.join(CURRENT_DIR, 'stored_models_cellfree_sweep_continue'))
    ap.add_argument('--n-train-override', type=int, default=None, help='smoke tests only')
    a = ap.parse_args()
    dev = torch.device('cuda')
    order = ORDER
    if a.runs:
        want = set(a.runs.split(','))
        order = [kb for kb in ORDER if f'K{kb[0]}b{kb[1]}' in want]
    for K, bits in order:
        tag = f'K{K}b{bits}'
        if a.claim_dir:
            os.makedirs(a.claim_dir, exist_ok=True)
            try:
                os.mkdir(os.path.join(a.claim_dir, tag))
            except FileExistsError:
                continue
        continue_run(source_run(K, bits), a.extra_epochs, a.out_root, dev, a.peak_lr, a.final_lr, a.n_train_override)
        torch.cuda.empty_cache()
    print('worker finished', flush=True)


if __name__ == '__main__':
    main()
