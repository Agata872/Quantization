"""Continue training the K=1, 1-bit cell-free run from its last checkpoint, to test whether its epoch
budget is what limits it. At 20 dB (LS estimator) the GNN scores 5.010, 0.30 below GNN-GD (5.307,
gd_refine_sweep.py) and 0.27 below coordinate descent (5.279); its per-epoch (biased) eval had
plateaued at 4.53-4.70 from epoch 6 to 20 at constant lr.

The source run folder is only read. Its checkpoint.pt (weights + Adam state after epoch 20) is the
start point, and training continues with exactly the original recipe of train_cellfree_best.py:
the same first 200000 training channels, biased Bussgang training loss at 20 dB, Gumbel-softmax
hard output with tau held at 0.25, 128 channels per step in 4 accumulation steps, chan_chunk 16,
MRT init, 1 refine pass. The one knob:
  --lr-schedule constant   lr stays at 1e-3, as in epochs 1-20: the direct 'more epochs' test
  --lr-schedule cosine     lr decays 1e-3 -> 5e-5 per step over the extra epochs (the schedule of
                           train_sweep.py), which separates 'too few epochs' from 'optimization
                           noise at a constant lr'

Every epoch, and the start point first, is scored on the held-out channels (test set [4096:6144]) at
20 dB with the LS estimator -- the one every reported number uses; it also picks model_best.pt --
and with the biased one, comparable with the logged epochs 1-20. At the end model_best.pt is scored
on -30:30:5 dB (LS). The run folder is resumable (re-run the same command).

  CUDA_VISIBLE_DEVICES=0 python continue_k1_training.py --extra-epochs 20 --lr-schedule constant
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from exp_cellfree_ablation import (  # noqa: E402
    GNNv2, PT, K, M_FULL, EVAL, sumrate_bussgang, normalize_power, run_model, evaluate, load_data,
)

SRC = os.path.join(CURRENT_DIR, 'stored_models_cellfree_sweep_K_bits',
                   'M_40_K_1_1bit_polar4_mrtinit_refine1_tau1to0.25in3_lr0.001_dl128_L4')
OUT_ROOT = os.path.join(CURRENT_DIR, 'stored_models_cellfree_k1_continue')
SNR = [float(x) for x in range(-30, 31, 5)]
NV20 = PT / 10 ** (20.0 / 10)
LR_MIN = 5e-5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extra-epochs', type=int, default=20)
    p.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    a = p.parse_args()

    dev = torch.device('cuda')
    src_ck = torch.load(os.path.join(SRC, 'checkpoint.pt'), map_location=dev, weights_only=False)
    src_best = torch.load(os.path.join(SRC, 'model_best.pt'), map_location=dev, weights_only=False)
    c = dict(src_ck['cfg'])
    first = src_ck['epoch'] + 1                    # 0-based index of the first extra epoch (20)
    last = first + a.extra_epochs
    run_dir = os.path.join(OUT_ROOT, f'{os.path.basename(SRC)}_ep{first}to{last}_{a.lr_schedule}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path, best_path = os.path.join(run_dir, 'checkpoint.pt'), os.path.join(run_dir, 'model_best.pt')
    logf = open(os.path.join(run_dir, 'train.log'), 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()

    log(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {torch.cuda.get_device_name(0)} =====")
    log(f'continuing {SRC}\n  from its checkpoint.pt (after epoch {first}), epochs {first + 1}..{last}, '
        f'lr schedule {a.lr_schedule}; original config {c}')

    Htr, Str, Hte, Ste = load_data(dev, c['n_train'])
    Hev, sev = Hte[EVAL], Ste[EVAL]
    levels = (np.sqrt(PT / (2 * M_FULL)) * torch.tensor([-1.0, 1.0])).to(dev)
    model = GNNv2(M_FULL, K, c['dl'], c['layers'], 1, c['tau_final'], levels, input_mode='polar4',
                  feat_stats=src_best['feat_stats']).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
    bs, acc = c['batch_channels'], c['accum_steps']
    micro, nb = bs // acc, Htr.shape[0] // bs
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.extra_epochs * nb, eta_min=LR_MIN)
             if a.lr_schedule == 'cosine' else None)

    def score():
        kw = dict(refine_iters=c['refine'], init=c['init'])
        return (evaluate(model, Hev, sev, None, c['chan_chunk'], NV20, estimator='ls', **kw),
                evaluate(model, Hev, sev, None, c['chan_chunk'], NV20, estimator='biased', **kw))

    def save_best(ep, ls, bi):
        torch.save({'model': model.state_dict(), 'epoch': ep, 'eval_rate': ls, 'eval_rate_biased': bi,
                    'estimator': 'ls', 'cfg': dict(c, epochs=last, lr_schedule=a.lr_schedule, source=SRC),
                    'feat_stats': src_best['feat_stats']}, best_path)

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if sched is not None:
            sched.load_state_dict(ck['sched'])
        start, hist, best = ck['epoch'] + 1, ck['hist'], ck['best']
        log(f'resumed this continuation at epoch {start + 1} (best LS so far {best:.4f})')
    else:
        model.load_state_dict(src_ck['model'])
        opt.load_state_dict(src_ck['opt'])
        for g in opt.param_groups:                  # the scheduler starts from the original lr
            g['lr'] = c['lr']
            g['initial_lr'] = c['lr']
        assert torch.allclose(model.fstats.cpu(), torch.tensor(src_best['feat_stats'], dtype=model.fstats.dtype))
        ls, bi = score()
        log(f'start point (epoch {first}): eval LS {ls:.4f} | biased {bi:.4f}   '
            f'[source run logged biased {src_ck["hist"][-1]["eval_rate"]:.4f} for this epoch]')
        assert abs(bi - src_ck['hist'][-1]['eval_rate']) < 1e-3, 'not the stored weights/channels'
        start, best = first, ls
        hist = [dict(h, source='original run, eval biased') for h in src_ck['hist']]
        hist.append({'epoch': first - 1, 'eval_rate_ls': ls, 'eval_rate_biased': bi, 'source': 'start point'})
        save_best(first - 1, ls, bi)
    ref = json.load(open(os.path.join(SRC, 'snr_curves.json')))
    i20 = SNR.index(20.0)
    gd = json.load(open(os.path.join(SRC, 'gnn_gd_curves.json')))['gnn_gd'][i20]
    log(f'targets @20dB (LS): original model_best {ref["gnn"][i20]:.4f} | GNN-GD {gd:.4f} | '
        f'coordinate descent {ref["coord_descent"][i20]:.4f}')

    torch.manual_seed(1000 + start)                 # epochs 1-20 used seed 0; do not replay its order
    t0 = time.time()
    for ep in range(start, last):
        model.tau = c['tau_final']
        model.output_type = 'gumbel_softmax_hard'
        model.train()
        perm = torch.randperm(Htr.shape[0], device=dev)
        run = 0.0
        bar = tqdm(range(nb), unit='batch', dynamic_ncols=True, desc=f'epoch {ep + 1}/{last} {a.lr_schedule}')
        for i in bar:
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad()
            for j in range(acc):
                sub = idx[j * micro:(j + 1) * micro]
                H, s = Htr[sub], Str[sub]
                y = normalize_power(run_model(model, H, s, c['chan_chunk'], c['refine'], c['init']), PT)
                loss = -sumrate_bussgang(y, H, s, NV20).mean() / acc
                loss.backward()
                run += loss.item()
            opt.step()
            if sched is not None:
                sched.step()
            if (i + 1) % 20 == 0:
                bar.set_postfix(train_rate=f'{-run / (i + 1):.3f}', lr=f'{opt.param_groups[0]["lr"]:.2e}')
        ls, bi = score()
        lr_now = opt.param_groups[0]['lr']
        hist.append({'epoch': ep, 'tau': model.tau, 'lr_end': lr_now, 'train_rate': -run / nb,
                     'eval_rate_ls': ls, 'eval_rate_biased': bi, 'elapsed_s': time.time() - t0,
                     'source': 'continuation'})
        mark = ''
        if ls > best:
            best, mark = ls, '  <- best (LS), saved'
            save_best(ep, ls, bi)
        log(f'epoch {ep + 1}/{last}  lr {lr_now:.2e}  train {-run / nb:.4f}  eval LS {ls:.4f} | biased {bi:.4f}  '
            f'[GNN-GD {gd:.3f}, gap {ls - gd:+.3f}]  {(time.time() - t0) / 60:.0f} min{mark}')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict() if sched is not None else None,
                    'epoch': ep, 'hist': hist, 'best': best}, ckpt_path)
        json.dump({'history': hist, 'source': SRC, 'lr_schedule': a.lr_schedule, 'config': c,
                   'best_eval_rate_ls': best, 'targets_20dB_ls': {
                       'original_model_best': ref['gnn'][i20], 'gnn_gd': gd,
                       'coordinate_descent': ref['coord_descent'][i20]}},
                  open(os.path.join(run_dir, 'history.json'), 'w'), indent=1)

    bk = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(bk['model'])
    curve = [evaluate(model, Hev, sev, None, c['chan_chunk'], PT / 10 ** (x / 10), estimator='ls',
                      refine_iters=c['refine'], init=c['init']) for x in SNR]
    json.dump({'snr_db': SNR, 'gnn': curve, 'best_epoch': bk['epoch'] + 1, 'estimator': 'ls',
               'original_gnn': ref['gnn'], 'gnn_gd': json.load(open(os.path.join(SRC, 'gnn_gd_curves.json')))['gnn_gd'],
               'coord_descent': ref['coord_descent']},
              open(os.path.join(run_dir, 'snr_curves_ls.json'), 'w'), indent=1)
    log(f'done: best epoch {bk["epoch"] + 1}, LS @20dB {bk["eval_rate"]:.4f}; SNR curve (LS) '
        + ' '.join(f'{x:g}:{v:.3f}' for x, v in zip(SNR, curve)))


if __name__ == '__main__':
    main()
