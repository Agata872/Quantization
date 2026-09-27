"""Long training run for cell-free 1-bit GNN precoding, M=40, K=1, no phase drift.

Uses the recipe selected by the exp_cellfree_ablation.py sweeps (rounds 1-4):

  input_mode   polar4      phase as cos/sin + log-magnitude channel. In a cell-free channel
                           |h| spans ~5 decades, so in the raw cartesian input the phase of a
                           weak AP is invisible to the network.   (+0.57 over raw)
  init         mrt         the first pass is seeded with the 1-bit MRT vector, so the network
                           learns a correction to a known-good point instead of the whole
                           assignment -- the same structure the coordinate-descent baseline
                           uses. Costs no extra forward pass.     (+0.89 over a zero-init pass)
  refine_iters             extra deep-unfolding passes on top of that: each re-run sees what the
                           antennas currently transmit and what the users currently receive.
                           Without the MRT seed, a second pass was worth +0.51.
  tau          1.0 -> 0.25 reached over the first 3 epochs, then held. While tau stays high,
                           the model improves the SAMPLED objective by flattening its logits --
                           Gumbel noise decorrelates the distortion across the 125 symbols -- while
                           the argmax readout we actually deploy gets worse. Round 6 measured the
                           argmax-minus-sampled gap closing from +0.34 to +0.02 as tau fell, with
                           the rate peaking at tau=0.25 (0.15 and 0.40 both scored lower).
                           Switching to a deterministic straight-through still collapses training.
                           NB: with the zero-init recipe a low tau was harmful -- this conclusion
                           does not transfer between recipes, it was re-measured under mrt init.
  lr           1e-3        5e-3 (the iid paper's value) diverges with the 4-feature input,
                           with or without gradient clipping.
  layers 4, dl 128, full M=40 (top-N AP selection lowers the achievable-rate ceiling).

Progress is reported with a tqdm bar per epoch. Every epoch writes a resumable checkpoint and
the best-eval weights, so the run can be stopped at any point and still leave a usable model.

  CUDA_VISIBLE_DEVICES=1 python train_cellfree_best.py --epochs 12
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from exp_cellfree_ablation import (  # noqa: E402
    GNNv2, NS, PT, K, M_FULL, EVAL, CD_TUNE, DATASET,
    sumrate_bussgang, normalize_power, run_model, evaluate,
    mrt_1bit, cd_reference, load_data, feature_stats,
)

SNR_POINTS = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
SNR_TRAIN_DB = 20.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--n-train', type=int, default=200000)
    p.add_argument('--refine', type=int, default=1)
    p.add_argument('--init', default='mrt', choices=['zeros', 'mrt'])
    p.add_argument('--tau0', type=float, default=1.0)
    p.add_argument('--tau-final', type=float, default=0.25)
    p.add_argument('--tau-epochs', type=int, default=3,
                   help='epochs to reach tau_final, then hold (NOT spread over --epochs)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dl', type=int, default=128)
    p.add_argument('--batch-channels', type=int, default=128)
    p.add_argument('--accum-steps', type=int, default=4)
    p.add_argument('--chan-chunk', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='stored_models_cellfree_polar4_mrtinit_nodrift')
    args = p.parse_args()

    noise_var = PT / 10 ** (SNR_TRAIN_DB / 10)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device('cuda')

    run_dir = os.path.join(CURRENT_DIR, args.out,
                           f'M_{M_FULL}_K_{K}_1bit_polar4_{args.init}init_refine{args.refine}_'
                           f'tau{args.tau0:g}to{args.tau_final:g}in{args.tau_epochs}_'
                           f'lr{args.lr:g}_dl{args.dl}_L{args.layers}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, 'checkpoint.pt')
    best_path = os.path.join(run_dir, 'model_best.pt')
    hist_path = os.path.join(run_dir, 'history.json')
    log_path = os.path.join(run_dir, 'train.log')
    logf = open(log_path, 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()

    log(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {torch.cuda.get_device_name(0)} =====")
    log(f"config: {vars(args)}")

    Htr, Str, Hte, Ste = load_data(dev, args.n_train)
    Hev, sev = Hte[EVAL], Ste[EVAL]
    log(f"train {tuple(Htr.shape)}  eval {Hev.shape[0]} channels  Pt={PT} "
        f"noise={noise_var} ({SNR_TRAIN_DB:.0f} dB)  no phase drift")

    # fixed reference points on the eval set
    ref_file = os.path.join(run_dir, 'references.json')
    if os.path.exists(ref_file):
        refs = json.load(open(ref_file))
    else:
        refs = {'mrt_1bit': sumrate_bussgang(mrt_1bit(Hev, sev, PT), Hev, sev,
                                             noise_var).mean().item()}
        cd, kap, fl = cd_reference(Hte[CD_TUNE], Ste[CD_TUNE], Hev, sev, PT, noise_var)
        refs.update(coord_descent=cd, cd_kappa=kap, cd_flip=fl)
        json.dump(refs, open(ref_file, 'w'), indent=1)
    log(f"references @20dB: 1-bit MRT {refs['mrt_1bit']:.3f} | "
        f"coord-descent {refs['coord_descent']:.3f} "
        f"(kappa={refs['cd_kappa']}, flip={refs['cd_flip']})")

    stats = feature_stats(Htr[:5000])
    levels = (np.sqrt(PT / (2 * M_FULL)) * torch.tensor([-1.0, 1.0])).to(dev)
    model = GNNv2(M_FULL, K, args.dl, args.layers, 1, args.tau0, levels,
                  input_mode='polar4', feat_stats=stats).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log(f"trainable params: {sum(q.numel() for q in model.parameters())}")

    start_ep, hist, best = 0, [], -1.0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_ep, hist, best = ck['epoch'] + 1, ck['hist'], ck['best']
        log(f"resumed from epoch {start_ep} (best eval so far {best:.3f})")

    bs, acc = args.batch_channels, args.accum_steps
    micro = bs // acc
    nb = Htr.shape[0] // bs
    t0 = time.time()

    for ep in range(start_ep, args.epochs):
        frac = min(1.0, ep / max(1, args.tau_epochs - 1))
        model.tau = args.tau0 + (args.tau_final - args.tau0) * frac
        model.output_type = 'gumbel_softmax_hard'
        model.train()
        perm = torch.randperm(Htr.shape[0], device=dev)
        run = 0.0
        bar = tqdm(range(nb), unit='batch', dynamic_ncols=True,
                   desc=f'epoch {ep + 1}/{args.epochs} tau={model.tau:.2f}')
        for i in bar:
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad()
            for a in range(acc):
                sub = idx[a * micro:(a + 1) * micro]
                H, s = Htr[sub], Str[sub]
                y = normalize_power(
                    run_model(model, H, s, args.chan_chunk, args.refine, args.init), PT)
                loss = -sumrate_bussgang(y, H, s, noise_var).mean() / acc
                loss.backward()
                run += loss.item()
            opt.step()
            if (i + 1) % 20 == 0:
                bar.set_postfix(train_rate=f'{-run / (i + 1):.3f}')

        ev = evaluate(model, Hev, sev, None, args.chan_chunk, noise_var,
                      refine_iters=args.refine, init=args.init)
        hist.append({'epoch': ep, 'tau': model.tau, 'train_rate': -run / nb, 'eval_rate': ev,
                     'elapsed_s': time.time() - t0})
        mark = ''
        if ev > best:
            best, mark = ev, '  <- best, saved'
            torch.save({'model': model.state_dict(), 'epoch': ep, 'eval_rate': ev,
                        'cfg': vars(args), 'feat_stats': stats}, best_path)
        log(f"epoch {ep + 1}/{args.epochs}  tau={model.tau:.2f}  train {-run / nb:.3f}  "
            f"eval(argmax) {ev:.3f}  [CD {refs['coord_descent']:.3f}, "
            f"gap {ev - refs['coord_descent']:+.3f}]  {(time.time() - t0) / 60:.0f} min{mark}")
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep,
                    'hist': hist, 'best': best, 'cfg': vars(args)}, ckpt_path)
        json.dump({'history': hist, 'references': refs, 'config': vars(args),
                   'best_eval_rate': best}, open(hist_path, 'w'), indent=1)

    # ---- final: SNR sweep with the best checkpoint, against both baselines ----
    log("training done, running SNR sweep with the best checkpoint")
    model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=False)['model'])
    curves = {'snr_db': SNR_POINTS.tolist(), 'gnn': [], 'mrt_1bit': [], 'coord_descent': []}
    for snr in SNR_POINTS:
        nv = PT / 10 ** (snr / 10)
        curves['gnn'].append(evaluate(model, Hev, sev, None, args.chan_chunk, nv,
                                      refine_iters=args.refine, init=args.init))
        curves['mrt_1bit'].append(
            sumrate_bussgang(mrt_1bit(Hev, sev, PT), Hev, sev, nv).mean().item())
        cd, _, _ = cd_reference(Hte[CD_TUNE], Ste[CD_TUNE], Hev, sev, PT, nv)
        curves['coord_descent'].append(cd)
        log(f"  {snr:+6.1f} dB: GNN {curves['gnn'][-1]:.3f} | "
            f"1-bit MRT {curves['mrt_1bit'][-1]:.3f} | coord-descent {cd:.3f}")
    json.dump(curves, open(os.path.join(run_dir, 'snr_curves.json'), 'w'), indent=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab in (('gnn', 'GNN (polar4 + refine)'), ('coord_descent', 'coordinate descent'),
                     ('mrt_1bit', '1-bit MRT')):
        ax.plot(SNR_POINTS, curves[key], marker='o', label=lab)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('rate [bits/channel use]')
    ax.set_title(f'cell-free, M={M_FULL}, K={K}, 1 bit, no drift')
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'rate_vs_snr.pdf'))
    log(f"saved everything to {run_dir}")


if __name__ == '__main__':
    main()
