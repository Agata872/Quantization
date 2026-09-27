"""Short-run ablation harness for cell-free 1-bit GNN precoding (K=1, no phase drift).

Trains on a subset of the existing cell-free dataset for a few epochs and scores the
result against 1-bit MRT and a coordinate-descent search reference on a held-out set of
channels. Meant for fast A/B of training changes (input representation, lr, top-N AP
selection, tau schedule) before committing to a full ~11h run of training.py.

Run on one GPU:  CUDA_VISIBLE_DEVICES=1 python exp_cellfree_ablation.py --sweep round1

Results are appended as JSON lines to exp_results/<sweep>.jsonl so a sweep can be
interrupted and resumed without losing finished runs.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in (PROJECT_ROOT, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import GNN_layer_fast  # noqa: E402

DATASET = os.path.join(CURRENT_DIR, 'datasets', 'cellfree',
                       'M_40_K_1_Ntr_200000_Nval_10000_Nte_10000_SperChannel125')
RESULT_DIR = os.path.join(CURRENT_DIR, 'exp_results')

M_FULL, K, NS = 40, 1, 125
PT = 40.0                      # total transmit power budget, independent of how many APs are active
SNR_TRAIN_DB = 20.0
NOISE_VAR = PT / 10 ** (SNR_TRAIN_DB / 10)

# held-out channel slices of Htest (disjoint)
CD_TUNE = slice(0, 256)        # coordinate-descent hyperparameters are picked here
EVAL = slice(4096, 6144)       # every number we report comes from here


# --------------------------------------------------------------------------------------
# rate / power
# --------------------------------------------------------------------------------------
def sumrate_bussgang(y, H, s, noise_var):
    """Per-channel sum rate, identical math to SumRateLoss_generalized_Bussgang.

    y: bs x M x Ns (complex)   H: bs x M x K   s: bs x K x Ns   -> bs
    """
    ns = y.shape[-1]
    G = (y @ torch.conj(s).transpose(1, 2)) / ns          # E[y s^H], bs x M x K
    q = y - G @ s
    HT = H.transpose(1, 2)
    HTG = HT @ G
    sig = torch.abs(torch.diagonal(HTG, dim1=1, dim2=2)) ** 2
    interf = torch.sum(torch.abs(HTG) ** 2, dim=-1) - sig
    Cq = (q @ torch.conj(q).transpose(1, 2)) / ns
    dist = torch.real(torch.diagonal(HT @ Cq @ torch.conj(H), dim1=1, dim2=2))
    return torch.sum(torch.log2(1 + sig / (interf + dist + noise_var)), dim=-1)


def sumrate_bussgang_ls(y, H, s, noise_var):
    """Per-channel sum rate with the in-sample LEAST-SQUARES Bussgang gain.

    sumrate_bussgang() above takes G = (1/Ns) sum y s^H, which equals E[y s^H] E[s s^H]^-1 only
    if the sample covariance of s is exactly I. Over Ns=125 symbols it is not: the sample power
    of each stream fluctuates by ~1/sqrt(Ns) ~ 9%, and the mismatch (1 - P_s) G s is booked as
    distortion. That fake distortion lies along the useful signal, so it is beamformed to the user
    and caps the SINDR near 1/(1-P_s)^2 ~ Ns (~24 dB median here). Symptom: unquantized MRT
    saturated at 8.4 bits at 30 dB instead of the 13.8 of log2(1+SNR||h||^2). Fitting G by least
    squares, G = (sum y s^H)(sum s s^H)^-1, removes it and matches the closed form to 0.012 bit.

    sumrate_bussgang() is deliberately left unchanged: it is the TRAINING loss of every model
    trained so far, including the running train_sweep.py sweep, which re-imports this module for
    each new configuration. Use this function for evaluation.
    """
    ns = y.shape[-1]
    Rys = (y @ torch.conj(s).transpose(1, 2)) / ns
    Rss = (s @ torch.conj(s).transpose(1, 2)) / ns
    G = Rys @ torch.linalg.inv(Rss)
    q = y - G @ s
    HT = H.transpose(1, 2)
    HTG = HT @ G
    sig = torch.abs(torch.diagonal(HTG, dim1=1, dim2=2)) ** 2
    interf = torch.sum(torch.abs(HTG) ** 2, dim=-1) - sig
    Cq = (q @ torch.conj(q).transpose(1, 2)) / ns
    dist = torch.real(torch.diagonal(HT @ Cq @ torch.conj(H), dim1=1, dim2=2))
    return torch.sum(torch.log2(1 + sig / (interf + dist + noise_var)), dim=-1)


RATE_FNS = {'biased': sumrate_bussgang, 'ls': sumrate_bussgang_ls}


def normalize_power(y, pt):
    l2 = torch.linalg.vector_norm(y, ord=2, dim=1)
    return torch.sqrt(pt / (torch.mean(l2 ** 2, dim=-1) + 1e-7))[:, None, None] * y


def select_top_n(H, n):
    """Keep the n strongest APs per channel (rows of H, ranked by row norm)."""
    if n is None or n >= H.shape[1]:
        return H
    order = torch.linalg.vector_norm(H, dim=2).argsort(dim=1, descending=True)[:, :n]
    return torch.gather(H, 1, order.unsqueeze(-1).expand(-1, -1, H.shape[2]))


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class GNNv2(nn.Module):
    """Same GNN as model.GNNmodel, with a selectable input representation.

    input_mode='raw'   : edge feature = [Re h, Im h]                      (current pipeline)
    input_mode='norm4' : edge feature = [Re h/||H||, Im h/||H||,
                                         std(log |h|/||H||), std(log ||H||)]
    input_mode='polar4': edge feature = [cos(arg h), sin(arg h),
                                         std(log |h|/||H||), std(log ||H||)]
        Magnitude and phase are decoupled: in a cell-free channel |h| spans ~5 decades, so
        in the cartesian forms the phase of a weak AP lives in the last bits of a tiny
        number and is effectively invisible to the network. Here phase is always O(1) and
        the dynamic range is carried by a separate log magnitude channel.
    The loss always sees the true (unnormalized) H; only what the network reads changes.
    """

    def __init__(self, M, K, dl, nr_hidden_layers, bits, tau, levels,
                 input_mode='raw', feat_stats=None, output_type='gumbel_softmax_hard',
                 logit_norm=None):
        super().__init__()
        # logit_norm rescales each decision's logits to a fixed spread before the Gumbel
        # softmax. Dividing by a positive per-decision scalar leaves argmax -- the deployed
        # rule -- untouched, so this only changes training dynamics. Without it the K>1 runs
        # saturated at |logit gap| ~31 against tau=0.25, which both kills Gumbel exploration
        # and drives the straight-through softmax gradient to zero, freezing the model on a
        # rate-equivalent copy of the warm start. The K=1 run that worked sat at ~1.8.
        # A tanh clamp was rejected: it saturates too, so it would stall the gradient as well.
        self.logit_norm = logit_norm
        self.M, self.K, self.bits = M, K, bits
        self.nr_out_levels = 2 ** bits
        self.tau = tau
        self.levels = levels
        self.input_mode = input_mode
        self.output_type = output_type
        self.register_buffer('fstats', torch.tensor(feat_stats if feat_stats is not None
                                                    else [0., 1., 0., 1.]))
        din = 2 if input_mode == 'raw' else 4
        self.din = din
        self.input_layer = GNN_layer_fast(din, dl, M, K)
        self.hidden_layers = nn.ModuleList(
            [GNN_layer_fast(dl, dl, M, K) for _ in range(nr_hidden_layers)])
        self.output_layer = GNN_layer_fast(dl, 2 * self.nr_out_levels, M, K, outputlayer=True)

    def features(self, H, s, y_prev=None):
        """y_prev (bs x M, complex) turns this into a refinement pass: the antenna nodes are
        seeded with what they currently transmit and the user nodes with what they currently
        receive (r = H^T y_prev), instead of the usual zero / symbol-only initialisation."""
        bs = H.shape[0]
        if self.input_mode == 'raw':
            z_mk = torch.stack((H.real, H.imag), -1).reshape(bs, self.M * self.K, 2)
            z_k = torch.stack((s.real, s.imag), -1)
        else:
            nrm = torch.linalg.vector_norm(H.reshape(bs, -1), dim=1).clamp_min(1e-12)
            hn = H / nrm[:, None, None]
            lg = torch.log(hn.abs().clamp_min(1e-12))
            lg = (lg - self.fstats[0]) / self.fstats[1]
            ln = ((torch.log(nrm) - self.fstats[2]) / self.fstats[3])[:, None, None].expand_as(lg)
            if self.input_mode == 'polar4':
                mag = hn.abs().clamp_min(1e-12)
                f0, f1 = hn.real / mag, hn.imag / mag      # unit-modulus phase
            else:
                f0, f1 = hn.real, hn.imag
            z_mk = torch.stack((f0, f1, lg, ln), -1).reshape(bs, self.M * self.K, 4)
            if y_prev is None:
                fb_k = torch.zeros_like(s.real), torch.zeros_like(s.real)
            else:
                # what each user currently receives, scaled to O(1)
                r = torch.einsum('bmk,bm->bk', H, y_prev) / (nrm * PT ** 0.5).unsqueeze(-1)
                fb_k = r.real, r.imag
            z_k = torch.stack((s.real, s.imag, fb_k[0], fb_k[1]), -1)
        if y_prev is None or self.input_mode == 'raw':
            z_m = torch.zeros(bs, self.M, self.din, device=H.device, dtype=z_mk.dtype)
        else:
            # what each antenna currently transmits, in units of the DAC level
            a = self.levels.abs().max()
            z_m = torch.stack((y_prev.real / a, y_prev.imag / a,
                               torch.zeros_like(y_prev.real), torch.zeros_like(y_prev.real)), -1)
        return z_mk, z_m, z_k

    def forward(self, H, s, y_prev=None):
        z_mk, z_m, z_k = self.features(H, s, y_prev)
        z_mk, z_m, z_k = self.input_layer(z_mk, z_m, z_k)
        for layer in self.hidden_layers:
            z_mk, z_m, z_k = layer(z_mk, z_m, z_k)

        _, z_m, _ = self.output_layer(z_mk, z_m, z_k)
        logits = z_m.reshape(-1, self.M, 2, self.nr_out_levels)
        # NOTE: dividing by the per-decision std was tried and destroyed training -- with 2
        # levels the std of two near-equal logits is ~0 at init, so the +1e-4 floor amplified
        # by ~1e4 and the gradient exploded (K=2 fell from 5.86 to 0.02). Kept only so the
        # failed experiment stays reproducible; prefer the logit L2 penalty in train_sweep.py.
        if self.logit_norm:
            logits = logits / (logits.std(dim=-1, keepdim=True) + 1e-4) * self.logit_norm
        self.last_logits = logits
        if self.output_type == 'gumbel_softmax_hard':
            oh = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        elif self.output_type == 'softmax_hard':
            ys = F.softmax(logits / self.tau, dim=-1)
            yh = torch.zeros_like(ys).scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
            oh = yh - ys.detach() + ys
        elif self.output_type == 'argmax':      # inference only
            oh = torch.zeros_like(logits).scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
        else:
            raise ValueError(self.output_type)
        lv = (oh * self.levels).sum(-1)
        return lv[:, :, 0] + 1j * lv[:, :, 1]


def mrt_init_folded(H, s, pt):
    """1-bit MRT transmit vector for the folded layout: H (b x M x K), s (b x K) -> b x M."""
    a = (pt / (2 * H.shape[1])) ** 0.5
    x = torch.einsum('bmk,bk->bm', torch.conj(H), s)
    return a * (torch.sign(x.real) + 1j * torch.sign(x.imag))


def run_model(model, H, s, chan_chunk, refine_iters=1, init='zeros'):
    """Forward all Ns symbols by folding them into the batch dim. -> bs x M x Ns

    refine_iters>1 re-runs the network on its own output (deep-unfolding style): each extra
    pass gets the current transmit vector on the antenna nodes and the current received
    signal on the user nodes, so the model can correct its own assignment.

    init='mrt' seeds the very first pass with the 1-bit MRT solution instead of zeros, so the
    network learns a correction to a known-good point rather than the whole assignment. This
    is what the coordinate-descent baseline does, and it costs no extra forward pass.
    """
    outs = []
    for b in range(0, H.shape[0], chan_chunk):
        Hb, sb = H[b:b + chan_chunk], s[b:b + chan_chunk]
        nb, ns = Hb.shape[0], sb.shape[-1]
        Hr = Hb.repeat_interleave(ns, 0)                      # (nb*ns) x M x K
        sr = sb.permute(0, 2, 1).reshape(nb * ns, Hb.shape[2])
        y = mrt_init_folded(Hr, sr, PT) if init == 'mrt' else None
        for _ in range(refine_iters):
            y = model(Hr, sr, y)
        outs.append(y.reshape(nb, ns, -1).permute(0, 2, 1))
    return torch.cat(outs)


# --------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------
def mrt_1bit(H, s, pt):
    n = H.shape[1]
    a = (pt / (2 * n)) ** 0.5
    x = torch.conj(H) @ s                                     # bs x M x Ns
    return a * (torch.sign(x.real) + 1j * torch.sign(x.imag))


def coord_descent(H, s, pt, kappa, sweeps=15, flip=0.0, gen=None, snapshots=None):
    """Per-symbol coordinate descent over the 4 QPSK levels (K=1 only). Local search.

    snapshots: optional sorted sweep counts (0 = the start point); the iterates after those
    sweeps are returned as a list instead of the final one."""
    dev = H.device
    n = H.shape[1]
    a = (pt / (2 * n)) ** 0.5
    lev = a * torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=torch.complex64, device=dev)
    h = H[:, :, 0]                                            # bs x M
    s1 = s[:, 0, :]                                           # bs x Ns
    target = (kappa * a * 2 ** 0.5 * h.abs().sum(-1))[:, None] * s1
    y = mrt_1bit(H, s, pt)
    if flip > 0:
        idx = torch.randint(0, 4, y.shape, device=dev, generator=gen)
        y = torch.where(torch.rand(y.shape, device=dev, generator=gen) < flip, lev[idx], y)
    snaps = [y.clone()] if snapshots and 0 in snapshots else []
    r = torch.einsum('bm,bmn->bn', h, y)
    for it in range(sweeps):
        for m in range(n):
            hm = h[:, m][:, None]
            r0 = r - hm * y[:, m, :]
            cost = (r0[:, None, :] + (hm * lev[None, :])[:, :, None] - target[:, None, :]).abs() ** 2
            y[:, m, :] = lev[cost.argmin(1)]
            r = r0 + hm * y[:, m, :]
        if snapshots and it + 1 in snapshots:
            snaps.append(y.clone())
    return snaps if snapshots else y


def cd_reference(Htune, stune, Hev, sev, pt, noise_var, sweeps=15, estimator='ls'):
    rate = RATE_FNS[estimator]
    """Tune (kappa, flip) on Htune, report the rate that config gets on Hev."""
    best = (-1, None, None)
    for kappa in (0.5, 0.6, 0.7, 0.8, 0.9):
        for flip in (0.0, 0.1, 0.2, 0.3):
            y = coord_descent(Htune, stune, pt, kappa, sweeps, flip)
            v = rate(y, Htune, stune, noise_var).mean().item()
            if v > best[0]:
                best = (v, kappa, flip)
    _, kappa, flip = best
    y = coord_descent(Hev, sev, pt, kappa, sweeps, flip)
    return rate(y, Hev, sev, noise_var).mean().item(), kappa, flip


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_data(dev, n_train):
    Htr = np.load(os.path.join(DATASET, 'Htrain.npy'), mmap_mode='r')[:n_train]
    str_ = np.load(os.path.join(DATASET, 'strain.npy'), mmap_mode='r')[:, :n_train * NS]
    Hte = np.load(os.path.join(DATASET, 'Htest.npy'))
    ste = np.load(os.path.join(DATASET, 'stest.npy'))
    Htr = torch.from_numpy(np.ascontiguousarray(Htr).astype(np.complex64)).to(dev)
    Str = torch.from_numpy(np.ascontiguousarray(str_).astype(np.complex64)).to(dev)
    Str = Str.reshape(K, n_train, NS).permute(1, 0, 2).contiguous()     # n x K x Ns
    Hte_t = torch.from_numpy(Hte.astype(np.complex64)).to(dev)
    Ste_t = torch.from_numpy(ste.astype(np.complex64)).to(dev)
    Ste_t = Ste_t.reshape(K, Hte.shape[0], NS).permute(1, 0, 2).contiguous()
    return Htr, Str, Hte_t, Ste_t


def feature_stats(H):
    """mean/std of log(|h|/||H||) and log||H|| over a sample, for input standardization."""
    nrm = torch.linalg.vector_norm(H.reshape(H.shape[0], -1), dim=1).clamp_min(1e-12)
    lg = torch.log((H / nrm[:, None, None]).abs().clamp_min(1e-12))
    ln = torch.log(nrm)
    return [lg.mean().item(), lg.std().item(), ln.mean().item(), ln.std().item()]


# --------------------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, H, s, top_n, chan_chunk, noise_var=NOISE_VAR, output_type='argmax',
             refine_iters=1, init='zeros', estimator='ls'):
    prev, model.output_type = model.output_type, output_type
    model.eval()
    Hs = select_top_n(H, top_n)
    vals = []
    for b in range(0, Hs.shape[0], 512):
        y = run_model(model, Hs[b:b + 512], s[b:b + 512], chan_chunk, refine_iters, init)
        y = normalize_power(y, PT)
        vals.append(RATE_FNS[estimator](y, Hs[b:b + 512], s[b:b + 512], noise_var))
    model.output_type = prev
    model.train()
    return torch.cat(vals).mean().item()


def train_one(cfg, data, dev, log):
    Htr, Str, Hte, Ste = data
    torch.manual_seed(cfg.get('seed', 0))
    np.random.seed(cfg.get('seed', 0))

    top_n = cfg.get('top_n')
    m_eff = top_n if top_n else M_FULL
    Htr_s = select_top_n(Htr, top_n)

    stats = feature_stats(Htr_s[:5000]) if cfg['input_mode'] != 'raw' else None
    levels = (np.sqrt(PT / (2 * m_eff)) * torch.tensor([-1.0, 1.0])).to(dev)
    model = GNNv2(m_eff, K, cfg.get('dl', 128), cfg.get('layers', 4), 1, cfg.get('tau0', 1.0),
                  levels, input_mode=cfg['input_mode'], feat_stats=stats).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    epochs = cfg['epochs']
    bs = cfg.get('batch_channels', 128)
    chunk = cfg.get('chan_chunk', 32)
    nb = Htr_s.shape[0] // bs

    sched = None
    if cfg.get('lr_decay'):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * nb,
                                                           eta_min=cfg['lr'] * 0.05)
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        # tau schedule + optional switch to deterministic straight-through at the end
        if cfg.get('tau_final') is not None and epochs > 1:
            model.tau = cfg['tau0'] + (cfg['tau_final'] - cfg['tau0']) * ep / (epochs - 1)
        else:
            model.tau = cfg['tau0']
        sh_last = cfg.get('softmax_hard_last_epochs', 0)
        model.output_type = 'softmax_hard' if ep >= epochs - sh_last and sh_last else 'gumbel_softmax_hard'

        perm = torch.randperm(Htr_s.shape[0], device=dev)
        run = 0.0
        # Gradient accumulation keeps the effective batch at batch_channels while capping
        # activation memory: chan_chunk alone cannot, since every chunk's activations stay
        # alive until backward. refine_iters>1 multiplies activations by the same factor.
        acc = cfg.get('accum_steps', 1)
        micro = bs // acc
        for i in range(nb):
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad()
            for a in range(acc):
                sub = idx[a * micro:(a + 1) * micro]
                H, s = Htr_s[sub], Str[sub]
                y = normalize_power(run_model(model, H, s, chunk, cfg.get('refine_iters', 1),
                                              cfg.get('init', 'zeros')), PT)
                loss = -RATE_FNS[cfg.get('loss', 'biased')](y, H, s, NOISE_VAR).mean() / acc
                loss.backward()
                run += loss.item()
            if cfg.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()
            if sched:
                sched.step()
        ekw = dict(refine_iters=cfg.get('refine_iters', 1), init=cfg.get('init', 'zeros'))
        ev = evaluate(model, Hte[EVAL], Ste[EVAL], top_n, chunk, **ekw)
        # Same model, same channels, scored with the sampling rule used during training. A
        # sampled score above the argmax score means the model is leaning on Gumbel noise to
        # decorrelate its distortion -- which is not available at inference.
        ev_s = evaluate(model, Hte[EVAL], Ste[EVAL], top_n, chunk,
                        output_type='gumbel_softmax_hard', **ekw)
        hist.append({'epoch': ep, 'train_rate': -run / nb, 'eval_rate': ev,
                     'eval_rate_sampled': ev_s, 'tau': model.tau, 'out': model.output_type})
        log(f"    ep{ep} tau={model.tau:.2f} train {-run / nb:.3f}  "
            f"eval(argmax) {ev:.3f}  eval(sampled) {ev_s:.3f}  "
            f"argmax-sampled {ev - ev_s:+.3f}  [{time.time() - t0:.0f}s]")
    return model, hist, time.time() - t0


# --------------------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------------------
BASE = dict(input_mode='raw', lr=5e-4, top_n=None, tau0=1.0, tau_final=None,
            epochs=3, batch_channels=128, chan_chunk=32, dl=128, layers=4, seed=0)


def cfgs(sweep, custom=None):
    if sweep == 'custom':
        # --configs '[["name", {"lr": 0.005}], ...]'
        return [(n, o) for n, o in json.loads(custom)]
    if sweep == 'smoke':
        return [('smoke_raw', dict(epochs=1)),
                ('smoke_norm4_top16', dict(epochs=1, input_mode='norm4', top_n=16, lr=5e-3))]
    if sweep == 'round1':
        return [
            ('A_baseline', {}),
            ('B_lr5e-3', dict(lr=5e-3)),
            ('C_norm4', dict(input_mode='norm4')),
            ('D_topN16', dict(top_n=16)),
            ('E_tau_anneal', dict(tau_final=0.3)),
            ('F_all', dict(lr=5e-3, input_mode='norm4', top_n=16, tau_final=0.3)),
        ]
    raise ValueError(sweep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='round1')
    ap.add_argument('--n-train', type=int, default=50000)
    ap.add_argument('--tag', default='')
    ap.add_argument('--configs', default=None, help="JSON list of [name, overrides] for --sweep custom")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f'{args.sweep}{args.tag}.jsonl')
    log_path = os.path.join(RESULT_DIR, f'{args.sweep}{args.tag}.log')
    logf = open(log_path, 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    dev = torch.device('cuda')
    log(f"\n===== {args.sweep}{args.tag} @ {datetime.now():%Y-%m-%d %H:%M:%S} "
        f"on {torch.cuda.get_device_name(0)} =====")
    data = load_data(dev, args.n_train)
    Htr, Str, Hte, Ste = data
    log(f"train {tuple(Htr.shape)}  eval {Hte[EVAL].shape[0]} channels  "
        f"Pt={PT} noise={NOISE_VAR} ({SNR_TRAIN_DB:.0f} dB)")

    # reference points on the eval set, recomputed for each top_n we will use
    refs = {}
    for n in (None, 12, 16, 20):
        He, se = select_top_n(Hte[EVAL], n), Ste[EVAL]
        Ht, st = select_top_n(Hte[CD_TUNE], n), Ste[CD_TUNE]
        mrt = sumrate_bussgang_ls(mrt_1bit(He, se, PT), He, se, NOISE_VAR).mean().item()
        cd, kap, fl = cd_reference(Ht, st, He, se, PT, NOISE_VAR)
        refs[str(n)] = {'mrt_1bit': mrt, 'coord_descent': cd, 'cd_kappa': kap, 'cd_flip': fl}
        log(f"  reference top_n={str(n):4s}: 1-bit MRT {mrt:.3f} | coord-descent {cd:.3f} "
            f"(kappa={kap}, flip={fl})")
    json.dump(refs, open(os.path.join(RESULT_DIR, f'refs{args.tag}.json'), 'w'), indent=1)

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            done.add(json.loads(line)['name'])

    for name, over in cfgs(args.sweep, args.configs):
        if name in done:
            log(f"-- skip {name} (already done)")
            continue
        cfg = dict(BASE, **over)
        log(f"-- {name}: {over}")
        model, hist, secs = train_one(cfg, data, dev, log)
        n = cfg.get('top_n')
        ref = refs[str(n)]
        best = max(h['eval_rate'] for h in hist)
        rec = {'name': name, 'cfg': cfg, 'hist': hist, 'seconds': secs,
               'eval_rate_final': hist[-1]['eval_rate'], 'eval_rate_best': best,
               'ref': ref, 'vs_cd': best - ref['coord_descent']}
        with open(out_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        log(f"   => final {hist[-1]['eval_rate']:.3f} best {best:.3f} | "
            f"CD {ref['coord_descent']:.3f} | gap {best - ref['coord_descent']:+.3f} | {secs:.0f}s")

    log("done")


if __name__ == '__main__':
    main()
