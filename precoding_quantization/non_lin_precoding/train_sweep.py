"""Sweep the validated cell-free recipe over user count K and DAC resolution (bits).

Recipe carried over from the exp_cellfree_ablation.py rounds 1-6 (measured at M=40, K=1,
1 bit, no phase drift), with two generalisations and one addition:

  polar4 input      phase as cos/sin + standardized log-magnitude. In a cell-free channel |h|
                    spans ~5 decades, so in the raw cartesian input a weak AP's phase is
                    invisible to the network.                                (+0.57)
  linear warm start the first (and only) pass is seeded with the QUANTIZED linear precoder --
                    MRT for K=1, ZF for K>1 -- so the network learns a correction to a
                    known-good point instead of the whole assignment.        (+0.89)
                    Generalised here from the 1-bit sign() seed to nearest-level quantization
                    of W s / ||w_m||, i.e. the paper's per-antenna input normalization.
  tau 1.0 -> 0.25   reached over the first 3 epochs then held. While tau is high the model
                    improves the SAMPLED objective by flattening its logits (Gumbel noise
                    decorrelates the distortion across the symbol block) while the deployed
                    argmax readout degrades.                                 (+0.18)
  lr 1e-3           5e-3 diverges with the 4-feature input, with or without gradient clipping.
  cosine LR decay   NEW, not carried over: the K=1 run oscillated between 4.53 and 4.70 for
                    12 epochs at constant lr instead of converging, and its last epoch fell to
                    4.166. Diagnosis was optimisation noise near convergence, not overfitting
                    (train-minus-eval was +0.010) and not Gumbel-noise exploitation
                    (argmax-minus-sampled was -0.012). Decay is the untested fix; the
                    best-eval checkpoint is still kept, so a bad late epoch cannot lose work.

Baselines per config: quantized linear (MRT/ZF), unquantized linear, and a coordinate-descent
search reference.

Every reported rate -- references, per-epoch eval (hence the best-checkpoint choice) and the final
evaluation -- uses the unbiased least-squares Bussgang estimator (sumrate_bussgang_ls). The
TRAINING loss is selected by --loss and defaults to the biased form every earlier run was trained
with (for K=1 the two are per-channel monotone in each other). The final evaluation is
reeval_sweep.correct_run, the code that re-scored the earlier runs: -30:30:5 dB, CD with
(kappa, sweeps) re-selected per SNR on a disjoint slice, snr_curves.json / references.json /
history.json and rate_vs_snr.pdf/.tex. Runs up to K=2 b=3 were trained before this change and were
scored with the biased estimator, then corrected by reeval_sweep.py.

  CUDA_VISIBLE_DEVICES=1 python train_sweep.py --K 2 --bits 1
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in (PROJECT_ROOT, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import GNNv2, RATE_FNS, sumrate_bussgang_ls, normalize_power, feature_stats
from data_handling import getdata_nonlinprec

NS, M_FULL, PT = 125, 40, 40.0
SNR_TRAIN_DB = 20.0
QUANT_PARAMS = os.path.join(PROJECT_ROOT, 'non-uniform-quant-params', 'Gaussian_var_0.5', 'numerical')
DATA_ROOT = os.path.join(CURRENT_DIR, 'datasets', 'cellfree')
NTR_FILE, NVAL_FILE, NTE_FILE = 200000, 10000, 10000
EVAL = slice(4096, 6144)     # held-out channels every reported number comes from
CD_TUNE = slice(0, 128)      # coordinate-descent kappa is picked here, never on EVAL


# ---------------------------------------------------------------------------- quantization
def load_levels(bits, dev):
    """DAC output levels per real dimension.

    1 bit: +-sqrt(Pt/2M), pre-scaled so ||y||^2 = Pt exactly (same convention as training.py).
    >1 bit: Lloyd-Max levels for a Gaussian of variance 0.5 per real dimension, which already
    give E[|y_m|^2] ~ 1 and hence E[||y||^2] ~ M = Pt. normalize_power fixes any residual.
    """
    if bits == 1:
        return (np.sqrt(PT / (2 * M_FULL)) * torch.tensor([-1.0, 1.0])).to(dev)
    lv = np.load(os.path.join(QUANT_PARAMS, f'{bits}bits_outputlevels.npy'))
    return torch.from_numpy(np.sort(lv.ravel())).float().to(dev)


def quantize_to_levels(x, levels):
    """Nearest-level quantization of the real and imaginary parts separately."""
    def q(v):
        return levels[(v.unsqueeze(-1) - levels).abs().argmin(-1)]
    return torch.complex(q(x.real), q(x.imag))


# ---------------------------------------------------------------------------- linear precoders
def linear_precoder(H, Pt):
    """MRT for K=1, ZF for K>1, normalized to total power Pt. H: bs x M x K -> bs x M x K."""
    K = H.shape[2]
    Hc = torch.conj(H)
    if K == 1:
        W = Hc
    else:
        W = Hc @ torch.linalg.inv(H.transpose(1, 2) @ Hc)
    a = torch.sqrt(Pt / (W.abs() ** 2).sum(dim=(1, 2), keepdim=True).clamp_min(1e-20))
    return a * W


def linear_quantized(H, s, levels, Pt):
    """Quantized linear precoder: the warm start, and also the linear baseline. bs x M x Ns."""
    W = linear_precoder(H, Pt)
    x = W @ s
    rn = torch.linalg.vector_norm(W, dim=2, keepdim=True).clamp_min(1e-12)   # ||w_m||
    return quantize_to_levels(x / rn, levels)


def linear_unquantized(H, s, Pt):
    return linear_precoder(H, Pt) @ s


# ---------------------------------------------------------------------------- GNN forward
def run_model_warm(model, H, s, levels, Pt, chan_chunk, warm=True):
    """Forward with the quantized-linear warm start, symbols folded into the batch dim."""
    outs = []
    for b in range(0, H.shape[0], chan_chunk):
        Hb, sb = H[b:b + chan_chunk], s[b:b + chan_chunk]
        nb, ns, K = Hb.shape[0], sb.shape[-1], Hb.shape[2]
        Hr = Hb.repeat_interleave(ns, 0)
        sr = sb.permute(0, 2, 1).reshape(nb * ns, K)
        if warm:
            y0 = linear_quantized(Hb, sb, levels, Pt)             # nb x M x ns
            y0r = y0.permute(0, 2, 1).reshape(nb * ns, -1)
        else:
            y0r = None
        y = model(Hr, sr, y0r)
        outs.append(y.reshape(nb, ns, -1).permute(0, 2, 1))
    return torch.cat(outs)


@torch.no_grad()
def evaluate(model, H, s, levels, Pt, nv, chunk, out='argmax', warm=True):
    prev, model.output_type = model.output_type, out
    model.eval()
    vals = []
    for b in range(0, H.shape[0], 512):
        Hb, sb = H[b:b + 512], s[b:b + 512]
        y = normalize_power(run_model_warm(model, Hb, sb, levels, Pt, chunk, warm), Pt)
        vals.append(sumrate_bussgang_ls(y, Hb, sb, nv))
    model.output_type = prev
    model.train()
    return torch.cat(vals).mean().item()


# ---------------------------------------------------------------------------- CD reference
def coord_descent(H, s, levels, Pt, kappa, sweeps=8, snapshots=None):
    """Per-symbol coordinate descent over the |levels|^2 complex DAC outputs, any K.

    Aims each user's received signal at kappa times what the UNQUANTIZED linear precoder would
    deliver, then greedily sweeps antennas. Local search, so this is a lower bound on what a
    per-symbol optimizer can reach, not the optimum.
    snapshots: optional sorted sweep counts (0 = the quantized-linear start); the iterates after
    those sweeps are returned as a list instead of the final one, so the number of sweeps can be
    tuned at the cost of a single run.
    """
    bs, M, K = H.shape
    cand = torch.complex(levels[:, None].expand(-1, len(levels)).reshape(-1),
                         levels[None, :].expand(len(levels), -1).reshape(-1))
    W = linear_precoder(H, Pt)
    x_lin = W @ s
    target = kappa * torch.einsum('bmk,bmn->bkn', H, x_lin)
    rn = torch.linalg.vector_norm(W, dim=2, keepdim=True).clamp_min(1e-12)
    y = quantize_to_levels(x_lin / rn, levels)
    snaps = [y.clone()] if snapshots and 0 in snapshots else []
    r = torch.einsum('bmk,bmn->bkn', H, y)
    for it in range(sweeps):
        for m in range(M):
            hm = H[:, m, :]                                        # bs x K
            r0 = r - torch.einsum('bk,bn->bkn', hm, y[:, m, :])
            contrib = torch.einsum('bk,c->bkc', hm, cand)          # bs x K x C
            d = r0.unsqueeze(2) + contrib.unsqueeze(3) - target.unsqueeze(2)
            ci = (d.real ** 2 + d.imag ** 2).sum(1).argmin(1)      # bs x Ns
            y[:, m, :] = cand[ci]
            r = r0 + torch.einsum('bk,bn->bkn', hm, y[:, m, :])
        if snapshots and it + 1 in snapshots:
            snaps.append(y.clone())
    return snaps if snapshots else y


def cd_reference(Ht, st, Hev, sev, levels, Pt, nv, sweeps=8, chunk=128):
    """Tune kappa on Ht, report the rate that kappa reaches on Hev (LS estimator). Only the
    reference printed during training; the final one comes from reeval_sweep.correct_run."""
    best = (-1e9, None)
    for kappa in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        y = coord_descent(Ht, st, levels, Pt, kappa, sweeps)
        v = sumrate_bussgang_ls(normalize_power(y, Pt), Ht, st, nv).mean().item()
        if v > best[0]:
            best = (v, kappa)
    kappa = best[1]
    vals = []
    for b in range(0, Hev.shape[0], chunk):
        Hb, sb = Hev[b:b + chunk], sev[b:b + chunk]
        y = coord_descent(Hb, sb, levels, Pt, kappa, sweeps)
        vals.append(sumrate_bussgang_ls(normalize_power(y, Pt), Hb, sb, nv))
    return torch.cat(vals).mean().item(), kappa


# ---------------------------------------------------------------------------- data
def load_data(K, n_train, dev):
    Htr, _, Hte, str_, _, ste = getdata_nonlinprec(
        NS, DATA_ROOT, M_FULL, K, NTR_FILE, NVAL_FILE, NTE_FILE, 'cellfree')
    Htr = torch.from_numpy(np.ascontiguousarray(Htr[:n_train]).astype(np.complex64)).to(dev)
    Str = torch.from_numpy(np.ascontiguousarray(str_[:, :n_train * NS]).astype(np.complex64))
    Str = Str.reshape(K, n_train, NS).permute(1, 0, 2).contiguous().to(dev)
    Hte_t = torch.from_numpy(Hte.astype(np.complex64)).to(dev)
    Ste_t = torch.from_numpy(ste.astype(np.complex64))
    Ste_t = Ste_t.reshape(K, Hte.shape[0], NS).permute(1, 0, 2).contiguous().to(dev)
    return Htr, Str, Hte_t, Ste_t


# ---------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, required=True)
    p.add_argument('--bits', type=int, required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--n-train', type=int, default=100000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=5e-5)
    p.add_argument('--tau0', type=float, default=1.0)
    p.add_argument('--tau-final', type=float, default=0.25)
    p.add_argument('--tau-epochs', type=int, default=3)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dl', type=int, default=128)
    p.add_argument('--batch-channels', type=int, default=128)
    p.add_argument('--accum-steps', type=int, default=4)
    p.add_argument('--chan-chunk', type=int, default=16)
    p.add_argument('--cd-sweeps', type=int, default=8)
    p.add_argument('--logit-l2', type=float, default=0.0,
                   help='penalty on mean(logits^2); discourages the saturation that freezes K=6')
    p.add_argument('--no-warm-start', action='store_true',
                   help='start the pass from zeros instead of the quantized linear precoder')
    p.add_argument('--logit-norm', type=float, default=None,
                   help='rescale each decision logits to this spread before Gumbel (argmax unchanged)')
    p.add_argument('--loss', default='biased', choices=['biased', 'ls'],
                   help='Bussgang estimator in the TRAINING loss; evaluation always uses ls')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='stored_models_cellfree_sweep_K_bits')
    args = p.parse_args()

    K, bits = args.K, args.bits
    warm = not args.no_warm_start
    nv_train = PT / 10 ** (SNR_TRAIN_DB / 10)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device('cuda')

    run_dir = os.path.join(CURRENT_DIR, args.out, f'M{M_FULL}_K{K}_b{bits}_'
                           f'ep{args.epochs}_n{args.n_train}_lr{args.lr:g}to{args.lr_min:g}'
                           + f'_tau{args.tau0:g}to{args.tau_final:g}in{args.tau_epochs}'
                           + (f'_ln{args.logit_norm:g}' if args.logit_norm else '')
                           + (f'_l2{args.logit_l2:g}' if args.logit_l2 else '')
                           + ('_nowarm' if args.no_warm_start else '')
                           + ('_lossls' if args.loss == 'ls' else ''))
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, 'checkpoint.pt')
    best_path = os.path.join(run_dir, 'model_best.pt')
    done_flag = os.path.join(run_dir, 'DONE')
    logf = open(os.path.join(run_dir, 'train.log'), 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()

    if os.path.exists(done_flag):
        log(f"[skip] {run_dir} already finished")
        return

    log(f"\n===== K={K} bits={bits} @ {datetime.now():%Y-%m-%d %H:%M:%S} "
        f"{torch.cuda.get_device_name(0)} =====")
    log(f"config: {vars(args)}")

    Htr, Str, Hte, Ste = load_data(K, args.n_train, dev)
    Hev, sev = Hte[EVAL], Ste[EVAL]
    levels = load_levels(bits, dev)
    log(f"train {tuple(Htr.shape)}  eval {Hev.shape[0]} ch  Pt={PT} noise={nv_train} "
        f"({SNR_TRAIN_DB:.0f} dB)  levels({len(levels)})={np.round(levels.cpu().numpy(), 3).tolist()}")

    ref_file = os.path.join(run_dir, 'references.json')
    if os.path.exists(ref_file):
        refs = json.load(open(ref_file))
    else:
        lin = sumrate_bussgang_ls(normalize_power(linear_quantized(Hev, sev, levels, PT), PT),
                                  Hev, sev, nv_train).mean().item()
        unq = sumrate_bussgang_ls(normalize_power(linear_unquantized(Hev, sev, PT), PT),
                                  Hev, sev, nv_train).mean().item()
        t0 = time.time()
        cd, kap = cd_reference(Hte[CD_TUNE], Ste[CD_TUNE], Hev, sev, levels, PT, nv_train,
                               args.cd_sweeps)
        refs = {'linear_quantized': lin, 'linear_unquantized': unq,
                'coord_descent': cd, 'cd_kappa': kap, 'cd_seconds': time.time() - t0}
        json.dump(refs, open(ref_file, 'w'), indent=1)
    log(f"references @20dB: {'MRT' if K == 1 else 'ZF'}+{bits}bit {refs['linear_quantized']:.3f} | "
        f"unquantized {refs['linear_unquantized']:.3f} | "
        f"coord-descent {refs['coord_descent']:.3f} (kappa={refs['cd_kappa']})")

    stats = feature_stats(Htr[:5000])
    model = GNNv2(M_FULL, K, args.dl, args.layers, bits, args.tau0, levels,
                  input_mode='polar4', feat_stats=stats, logit_norm=args.logit_norm).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bs, acc = args.batch_channels, args.accum_steps
    micro, nb = bs // acc, Htr.shape[0] // bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * nb,
                                                       eta_min=args.lr_min)
    log(f"trainable params: {sum(q.numel() for q in model.parameters())}  batches/epoch {nb}  "
        f"(training loss: {args.loss} estimator; every eval: LS estimator)")

    start_ep, hist, best = 0, [], -1e9
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_ep, hist, best = ck['epoch'] + 1, ck['hist'], ck['best']
        log(f"resumed at epoch {start_ep} (best so far {best:.3f})")

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.tau = args.tau0 + (args.tau_final - args.tau0) * min(
            1.0, ep / max(1, args.tau_epochs - 1))
        model.output_type = 'gumbel_softmax_hard'
        model.train()
        perm = torch.randperm(Htr.shape[0], device=dev)
        run = 0.0
        bar = tqdm(range(nb), unit='batch', dynamic_ncols=True,
                   desc=f'K{K} b{bits} ep{ep + 1}/{args.epochs} tau={model.tau:.2f}')
        for i in bar:
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad()
            for a in range(acc):
                sub = idx[a * micro:(a + 1) * micro]
                Hb, sb = Htr[sub], Str[sub]
                y = normalize_power(run_model_warm(model, Hb, sb, levels, PT,
                                                   args.chan_chunk, warm), PT)
                rate_loss = -RATE_FNS[args.loss](y, Hb, sb, nv_train).mean() / acc
                loss = rate_loss
                if args.logit_l2:
                    loss = loss + args.logit_l2 * (model.last_logits ** 2).mean() / acc
                loss.backward()
                run += rate_loss.item()      # report the rate only, not the penalty
            opt.step()
            sched.step()
            if (i + 1) % 20 == 0:
                bar.set_postfix(rate=f'{-run / (i + 1):.3f}', lr=f'{sched.get_last_lr()[0]:.1e}')

        ev = evaluate(model, Hev, sev, levels, PT, nv_train, args.chan_chunk, warm=warm)
        ev_s = evaluate(model, Hev, sev, levels, PT, nv_train, args.chan_chunk,
                        out='gumbel_softmax_hard', warm=warm)
        hist.append({'epoch': ep, 'tau': model.tau, 'lr': sched.get_last_lr()[0],
                     'train_rate': -run / nb, 'eval_rate': ev, 'eval_rate_sampled': ev_s,
                     'elapsed_s': time.time() - t0})
        mark = ''
        if ev > best:
            best, mark = ev, '  <- best, saved'
            torch.save({'model': model.state_dict(), 'epoch': ep, 'eval_rate': ev,
                        'cfg': vars(args), 'feat_stats': stats}, best_path)
        log(f"K{K} b{bits} ep{ep + 1}/{args.epochs} tau={model.tau:.2f} "
            f"lr={sched.get_last_lr()[0]:.1e} train {-run / nb:.3f}  eval {ev:.3f}  "
            f"sampled {ev_s:.3f}  [lin {refs['linear_quantized']:.3f}, "
            f"CD {refs['coord_descent']:.3f}, gap {ev - refs['coord_descent']:+.3f}]  "
            f"{(time.time() - t0) / 60:.0f} min{mark}")
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'epoch': ep, 'hist': hist, 'best': best,
                    'cfg': vars(args)}, ckpt_path)
        json.dump({'history': hist, 'references': refs, 'config': vars(args),
                   'best_eval_rate': best}, open(os.path.join(run_dir, 'history.json'), 'w'),
                  indent=1)

    # ---- final evaluation: the same code path that re-scored the earlier runs ----
    # LS estimator, -30:30:5 dB, CD with (kappa, sweeps) re-selected per SNR, per-run figure.
    # DONE is written only after it succeeds, so re-running this command retries it.
    log("training done, final evaluation of the best checkpoint (reeval_sweep.correct_run)")
    from reeval_sweep import correct_run      # imported here: reeval_sweep imports this module
    flag = correct_run(run_dir, 'sweep', dev, {}, native=True)
    if flag != 'OK':
        raise RuntimeError(f'final evaluation check failed ({flag}); DONE not written')
    open(done_flag, 'w').write(f'{datetime.now():%Y-%m-%d %H:%M:%S} best_eval_ls {best:.4f}\n')
    log(f"saved everything to {run_dir}")


if __name__ == '__main__':
    main()
