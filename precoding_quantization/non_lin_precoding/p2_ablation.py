"""Prompt-2 ablation: original GNN of Feys et al. (arXiv 2507.10634) vs. the design of this paper.

Every variant is the same P2GNN backbone (p2_model.py: input 4 -> 128, 4 hidden layers of width
128, output 128 -> 2L, LeakyReLU, mean aggregation, straight-through Gumbel-softmax, argmax at
inference), the same 100k training channels with 125 Gaussian symbol vectors each, 128 channels
per Adam step, the same number of epochs, the same moment-fit (biased) Bussgang training loss of
Feys eq. (5)/(9) and the paper, and the same per-epoch validation (LS estimator, argmax,
20 dB, Htest[4096:6144]) that selects the checkpoint. They differ in:

  variant      edge     ref   tau             logit L2   lr
  A_orig       raw      no    1 (constant)    0          5e-3 constant  (Feys Sec. VI-A)
  A_orig_lr1e-3 raw     no    1 (constant)    0          1e-3 constant  (tuning of A)
  B_orig_ts    raw      no    1 -> 0.25 / 3   0.1        cosine 1e-3 -> 5e-5   (paper recipe)
  C_F          polar4   no    paper           0.1        paper
  D_R          raw      yes   paper           0.1        paper
  E_FR_nol2    polar4   yes   paper           0          paper
  F_full       polar4   yes   paper           0.1        paper   (= train_sweep.py, K>=2)

  F = phase / log-magnitude edge features (eq:tp_edge_init), R = quantized ZF reference at the
  AP nodes and its predicted reception at the UE nodes (eq:tp_ap_init, eq:tp_ue_init).

Adaptations of the original to this paper's constraints (recorded in every config.json):
  * 4-dim zero-padded input (identical function, 0.8 % different Xavier bound, same #params);
  * cell-free channels, M = 40, 100k (not 200k) training channels and 15 (not 20) epochs, the
    common budget of all variants; Lloyd-Max levels for b >= 2, +-sqrt(Pt/2M) for b = 1 as in
    the original code; best-epoch selection on the validation set (Feys select
    hyperparameters on a validation set; the checkpoint rule here is the paper's).

Diagnostics stored per epoch (history.json): train rate, validation LS/biased/Gumbel-sampled
rate, total and per-block gradient norms (mean / median / max over the steps), the saturation
of the training logits (share of decisions whose softmax(logit / tau) max exceeds 0.99, mean
logit^2), and the logit-gap distribution (top-1 minus top-2 logit, argmax, first 256 validation
channels): quantiles, gap / tau quantiles, log-spaced histogram.

After the last epoch the best checkpoint is scored ONCE on the frozen test set Htest[6144:8192]
(never read by any script before 2026-10-14; see p1_independence.py): per-channel LS sum rate on
-30:30:5 dB and per-user rates at 20 and 30 dB -> test_rates.npz. Nothing reads these numbers
back for any choice made in this script.

  CUDA_VISIBLE_DEVICES=1 python p2_ablation.py --K 2 --bits 1 --variant F_full --seed 0
  python p2_ablation.py --worker --queue ablation_p2/queue.txt     # claims jobs in order
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch._dynamo

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import RATE_FNS, sumrate_bussgang_ls, normalize_power, feature_stats  # noqa: E402
from train_sweep import load_data, load_levels, run_model_warm, EVAL, PT, M_FULL, SNR_TRAIN_DB  # noqa: E402
from p2_model import P2GNN  # noqa: E402

TEST = slice(6144, 8192)            # frozen independent test set (Prompt 1 / Prompt 2)
SNR_GRID = [float(x) for x in range(-30, 31, 5)]
OUT_ROOT = os.path.join(CURRENT_DIR, 'ablation_p2')
NV20 = PT / 10 ** (SNR_TRAIN_DB / 10)
GAP_BINS = np.logspace(-4, 3, 57)   # logit-gap histogram edges

OURS = dict(tau0=1.0, tau_final=0.25, tau_epochs=3, logit_l2=0.1, lr=1e-3, lr_min=5e-5, sched='cosine')
ORIG = dict(tau0=1.0, tau_final=1.0, tau_epochs=1, logit_l2=0.0, lr=5e-3, lr_min=5e-3, sched='const')
VARIANTS = {
    'A_orig':        dict(ORIG, edge='raw', ref=False),
    'A_orig_lr1e-3': dict(ORIG, edge='raw', ref=False, lr=1e-3, lr_min=1e-3),
    'B_orig_ts':     dict(OURS, edge='raw', ref=False),
    'C_F':           dict(OURS, edge='polar4', ref=False),
    'D_R':           dict(OURS, edge='raw', ref=True),
    'E_FR_nol2':     dict(OURS, edge='polar4', ref=True, logit_l2=0.0),
    'F_full':        dict(OURS, edge='polar4', ref=True),
}


def run_dir_of(K, bits, variant, seed, root=OUT_ROOT):
    return os.path.join(root, 'runs', f'K{K}b{bits}', f'{variant}_s{seed}')


def git_rev():
    try:
        return subprocess.check_output(['git', '-C', CURRENT_DIR, 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:
        return 'unknown'


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ---------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def logit_gap_stats(model, H, s, levels, chunk, tau):
    """Top-1 minus top-2 logit of every decision (argmax, no Gumbel noise)."""
    model.eval()
    gaps, sat = [], []
    for b in range(0, H.shape[0], chunk):
        Hb, sb = H[b:b + chunk], s[b:b + chunk]
        nb, ns, K = Hb.shape[0], sb.shape[-1], Hb.shape[2]
        Hr = Hb.repeat_interleave(ns, 0)
        sr = sb.permute(0, 2, 1).reshape(nb * ns, K)
        yr = None
        if model.ref:
            from train_sweep import linear_quantized
            yr = linear_quantized(Hb, sb, levels, PT).permute(0, 2, 1).reshape(nb * ns, -1)
        lg = model.logits(Hr, sr, yr)                         # n x M x 2 x L
        top2 = lg.topk(2, dim=-1).values
        gaps.append((top2[..., 0] - top2[..., 1]).flatten().float().cpu())
        sat.append((torch.softmax(lg / tau, -1).amax(-1) > 0.99).flatten().float().cpu())
    model.train()
    g = torch.cat(gaps).numpy()
    q = [1, 5, 25, 50, 75, 95, 99]
    return {'n': int(g.size), 'quantiles_pct': q,
            'gap_quantiles': np.percentile(g, q).round(5).tolist(),
            'gap_over_tau_quantiles': (np.percentile(g, q) / tau).round(5).tolist(),
            'share_softmax_max_gt_0.99': float(torch.cat(sat).mean()),
            'hist_edges': GAP_BINS.tolist(),
            'hist_counts': np.histogram(np.clip(g, GAP_BINS[0], GAP_BINS[-1]), GAP_BINS)[0].tolist()}


@torch.no_grad()
def score(model, H, s, levels, chunk, warm, mode):
    """Validation rate at 20 dB: (LS, biased) for argmax, LS only for 'gumbel_softmax_hard'."""
    prev = model.output_type
    model.output_type = mode
    model.eval()
    ls, bi = [], []
    for b in range(0, H.shape[0], 256):
        Hb, sb = H[b:b + 256], s[b:b + 256]
        y = normalize_power(run_model_warm(model, Hb, sb, levels, PT, chunk, warm), PT)
        ls.append(sumrate_bussgang_ls(y, Hb, sb, NV20))
        bi.append(RATE_FNS['biased'](y, Hb, sb, NV20))
    model.output_type = prev
    model.train()
    return torch.cat(ls), torch.cat(bi)


def per_user_ls(y, H, s, noise_var):
    """Per-user LS rate (same math as sumrate_bussgang_ls, without the sum): n x K."""
    ns = y.shape[-1]
    G = (y @ torch.conj(s).transpose(1, 2)) / ns @ torch.linalg.inv((s @ torch.conj(s).transpose(1, 2)) / ns)
    q = y - G @ s
    HT = H.transpose(1, 2)
    HTG = HT @ G
    sig = torch.abs(torch.diagonal(HTG, dim1=1, dim2=2)) ** 2
    interf = torch.sum(torch.abs(HTG) ** 2, dim=-1) - sig
    Cq = (q @ torch.conj(q).transpose(1, 2)) / ns
    dist = torch.real(torch.diagonal(HT @ Cq @ torch.conj(H), dim1=1, dim2=2))
    return torch.log2(1 + sig / (interf + dist + noise_var))


# ---------------------------------------------------------------------------- one run
def train_run(K, bits, variant, seed, epochs, n_train, root, compile_model=True, chan_chunk=16,
              accum=4, batch_channels=128, layers=4, dl=128):
    cfg = dict(VARIANTS[variant])
    run_dir = run_dir_of(K, bits, variant, seed, root)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'DONE')):
        print(f'[skip] {run_dir} DONE', flush=True)
        return 'OK'
    logf = open(os.path.join(run_dir, 'train.log'), 'a')

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()

    dev = torch.device('cuda')
    full_cfg = dict(cfg, K=K, bits=bits, variant=variant, seed=seed, epochs=epochs, n_train=n_train,
                    M=M_FULL, Ns=125, Pt=PT, snr_train_db=SNR_TRAIN_DB, batch_channels=batch_channels,
                    accum_steps=accum, chan_chunk=chan_chunk, layers=layers, dl=dl,
                    train_loss='biased (moment) Bussgang fit, as Feys eq. (5) and the paper',
                    val='Htest[4096:6144], 20 dB, LS estimator, argmax; selects model_best.pt',
                    test='Htest[6144:8192] (frozen), scored once after training',
                    compiled=compile_model, git_rev=git_rev(),
                    script_sha256=sha256(os.path.abspath(__file__)),
                    model_sha256=sha256(os.path.join(CURRENT_DIR, 'p2_model.py')),
                    torch=torch.__version__, host=socket.gethostname(),
                    gpu=torch.cuda.get_device_name(0),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    command=' '.join([sys.executable] + sys.argv))
    json.dump(full_cfg, open(os.path.join(run_dir, 'config.json'), 'w'), indent=1)
    log(f"\n===== {variant} K={K} b={bits} seed={seed} {datetime.now():%Y-%m-%d %H:%M:%S} "
        f"{full_cfg['gpu']} (CUDA_VISIBLE_DEVICES={full_cfg['cuda_visible_devices']}) =====")
    log(f'config: {json.dumps(full_cfg)}')

    torch.manual_seed(seed)
    np.random.seed(seed)
    Htr, Str, Hte, Ste = load_data(K, n_train, dev)
    Hev, sev = Hte[EVAL], Ste[EVAL]
    levels = load_levels(bits, dev)
    stats = feature_stats(Htr[:5000])
    model = P2GNN(M_FULL, K, dl, layers, bits, cfg['tau0'], levels, edge=cfg['edge'], ref=cfg['ref'],
                  feat_stats=stats, pt=PT).to(dev)
    warm = cfg['ref']
    nparams = sum(q.numel() for q in model.parameters())
    if compile_model:
        torch._dynamo.config.cache_size_limit = 64
        model.logits = torch.compile(model.logits)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    micro, nb = batch_channels // accum, Htr.shape[0] // batch_channels
    if cfg['sched'] == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * nb, eta_min=cfg['lr_min'])
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    blocks = {'input': list(model.input_layer.parameters()),
              'hidden': [q for l in model.hidden_layers for q in l.parameters()],
              'output': list(model.output_layer.parameters())}
    log(f'params {nparams}  batches/epoch {nb}  feat_stats {np.round(stats, 4).tolist()}')

    ckpt_path, best_path = os.path.join(run_dir, 'checkpoint.pt'), os.path.join(run_dir, 'model_best.pt')
    start, hist, best = 0, [], -1e9
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start, hist, best = ck['epoch'] + 1, ck['hist'], ck['best']
        torch.set_rng_state(ck['rng_cpu'])
        torch.cuda.set_rng_state(ck['rng_cuda'])
        log(f'resumed after epoch {start} (best so far {best:.4f})')

    t0 = time.time() - (hist[-1]['elapsed_s'] if hist else 0.0)
    failed = None
    for ep in range(start, epochs):
        model.tau = cfg['tau0'] + (cfg['tau_final'] - cfg['tau0']) * min(1.0, ep / max(1, cfg['tau_epochs'] - 1))
        model.output_type = 'gumbel_softmax_hard'
        model.train()
        perm = torch.randperm(Htr.shape[0], device=dev)
        run_rate, gnorm, bnorm, satv, l2v = 0.0, [], {k: [] for k in blocks}, [], []
        te = time.time()
        for i in range(nb):
            idx = perm[i * batch_channels:(i + 1) * batch_channels]
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                sub = idx[a * micro:(a + 1) * micro]
                Hb, sb = Htr[sub], Str[sub]
                y = normalize_power(run_model_warm(model, Hb, sb, levels, PT, chan_chunk, warm), PT)
                rate_loss = -RATE_FNS['biased'](y, Hb, sb, NV20).mean() / accum
                loss = rate_loss
                if cfg['logit_l2']:
                    loss = loss + cfg['logit_l2'] * (model.last_logits ** 2).mean() / accum
                loss.backward()
                run_rate += rate_loss.item()
            if i % 10 == 0:
                with torch.no_grad():
                    lgt = model.last_logits.detach()
                    satv.append((torch.softmax(lgt / model.tau, -1).amax(-1) > 0.99).float().mean().item())
                    l2v.append((lgt ** 2).mean().item())
                    for k, ps in blocks.items():
                        bnorm[k].append(torch.sqrt(sum((q.grad ** 2).sum() for q in ps if q.grad is not None)).item())
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            gnorm.append(gn)
            if not np.isfinite(run_rate) or not np.isfinite(gn):
                failed = f'non-finite loss/gradient at epoch {ep + 1} step {i} (rate {run_rate}, |g| {gn})'
                break
            opt.step()
            sched.step()
        if failed:
            log('FAILED: ' + failed)
            break
        t_train = time.time() - te
        ls, bi = score(model, Hev, sev, levels, chan_chunk, warm, 'argmax')
        ls_s, _ = score(model, Hev, sev, levels, chan_chunk, warm, 'gumbel_softmax_hard')
        gaps = logit_gap_stats(model, Hev[:256], sev[:256], levels, chan_chunk, model.tau)
        ev = ls.mean().item()
        g = np.array(gnorm)
        rec = {'epoch': ep + 1, 'tau': model.tau, 'lr_end': opt.param_groups[0]['lr'],
               'train_rate_biased': -run_rate / nb, 'val_rate_ls': ev, 'val_rate_biased': bi.mean().item(),
               'val_rate_sampled_ls': ls_s.mean().item(), 'val_p5_ls': float(np.percentile(ls.cpu().numpy(), 5)),
               'grad_norm': {'mean': float(g.mean()), 'median': float(np.median(g)), 'max': float(g.max()),
                             'min': float(g.min())},
               'grad_norm_blocks_median': {k: float(np.median(v)) for k, v in bnorm.items()},
               'train_share_saturated': float(np.mean(satv)), 'train_mean_logit_sq': float(np.mean(l2v)),
               'val_logit_gap': gaps, 'train_seconds': t_train, 'elapsed_s': time.time() - t0}
        hist.append(rec)
        mark = ''
        if ev > best:
            best, mark = ev, '  <- best, saved'
            torch.save({'model': model.state_dict(), 'epoch': ep + 1, 'val_rate_ls': ev, 'cfg': full_cfg,
                        'feat_stats': stats}, best_path)
        gq = gaps['gap_quantiles']
        log(f"{variant} K{K}b{bits} s{seed} ep{ep + 1}/{epochs} tau={model.tau:.3f} lr={rec['lr_end']:.1e} "
            f"train {rec['train_rate_biased']:.3f} val LS {ev:.4f} (sampled {rec['val_rate_sampled_ls']:.4f}, "
            f"p5 {rec['val_p5_ls']:.3f}) |g| med {np.median(g):.3g} sat {rec['train_share_saturated']:.3f} "
            f"gap med {gq[3]:.3g} p95 {gq[5]:.3g} [{t_train / 60:.1f} min, total {rec['elapsed_s'] / 60:.0f}]{mark}")
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': ep, 'hist': hist, 'best': best, 'rng_cpu': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state()}, ckpt_path)
        json.dump({'config': full_cfg, 'params': nparams, 'history': hist, 'best_val_rate_ls': best},
                  open(os.path.join(run_dir, 'history.json'), 'w'), indent=1)

    if failed:
        json.dump({'config': full_cfg, 'params': nparams, 'history': hist, 'failed': failed},
                  open(os.path.join(run_dir, 'history.json'), 'w'), indent=1)
        open(os.path.join(run_dir, 'FAILED'), 'w').write(failed + '\n')
        if not os.path.exists(best_path):
            return 'FAILED'

    # ---- frozen test set, best checkpoint, scored once ----
    del Htr, Str
    torch.cuda.empty_cache()
    bk = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(bk['model'])
    Ht, st = Hte[TEST], Ste[TEST]
    model.output_type = 'argmax'
    model.eval()
    with torch.no_grad():
        y = normalize_power(torch.cat([run_model_warm(model, Ht[b:b + 256], st[b:b + 256], levels, PT,
                                                      chan_chunk, warm) for b in range(0, Ht.shape[0], 256)]), PT)
        rates = torch.stack([sumrate_bussgang_ls(y, Ht, st, PT / 10 ** (x / 10)) for x in SNR_GRID]).cpu().numpy()
        pu20 = per_user_ls(y, Ht, st, NV20).cpu().numpy()
        pu30 = per_user_ls(y, Ht, st, PT / 1e3).cpu().numpy()
        yv = normalize_power(torch.cat([run_model_warm(model, Hev[b:b + 256], sev[b:b + 256], levels, PT,
                                                       chan_chunk, warm) for b in range(0, Hev.shape[0], 256)]), PT)
        val20 = sumrate_bussgang_ls(yv, Hev, sev, NV20).cpu().numpy()
    assert abs(val20.mean() - bk['val_rate_ls']) < 2e-3, (val20.mean(), bk['val_rate_ls'])
    np.savez(os.path.join(run_dir, 'test_rates.npz'), snr_db=np.array(SNR_GRID), sum_rate=rates,
             per_user_20db=pu20, per_user_30db=pu30, val_sum_rate_20db=val20)
    i20 = SNR_GRID.index(20.0)
    summary = {'best_epoch': bk['epoch'], 'val_rate_ls': bk['val_rate_ls'],
               'test_mean_20db': float(rates[i20].mean()), 'test_p5_20db': float(np.percentile(rates[i20], 5)),
               'test_mean_30db': float(rates[SNR_GRID.index(30.0)].mean()),
               'train_hours': hist[-1]['elapsed_s'] / 3600 if hist else None, 'params': nparams,
               'failed': failed}
    json.dump(summary, open(os.path.join(run_dir, 'summary.json'), 'w'), indent=1)
    log(f"test (frozen, best epoch {bk['epoch']}): 20 dB mean {summary['test_mean_20db']:.4f} "
        f"p5 {summary['test_p5_20db']:.4f} | 30 dB mean {summary['test_mean_30db']:.4f}")
    open(os.path.join(run_dir, 'DONE'), 'w').write(f'{datetime.now():%Y-%m-%d %H:%M:%S}\n')
    return 'OK'


def worker(queue, claims, epochs, n_train, root, compile_model):
    os.makedirs(claims, exist_ok=True)
    while True:
        jobs = [l.split() for l in open(queue) if l.strip() and not l.startswith('#')]
        took = False
        for K, bits, variant, seed in jobs:
            tag = f'K{K}b{bits}_{variant}_s{seed}'
            if os.path.exists(os.path.join(run_dir_of(int(K), int(bits), variant, int(seed), root), 'DONE')):
                continue
            try:
                os.mkdir(os.path.join(claims, tag))
            except FileExistsError:
                continue
            open(os.path.join(claims, tag, 'owner'), 'w').write(
                f'{socket.gethostname()} pid {os.getpid()} CUDA_VISIBLE_DEVICES='
                f'{os.environ.get("CUDA_VISIBLE_DEVICES", "")} {datetime.now():%Y-%m-%d %H:%M:%S}\n')
            took = True
            try:
                flag = train_run(int(K), int(bits), variant, int(seed), epochs, n_train, root, compile_model)
            except torch.OutOfMemoryError:
                # GPU shared with other jobs: release the claim so the job is retried later, keep a record
                flag = 'OOM (claim released, retry later)'
                d = run_dir_of(int(K), int(bits), variant, int(seed), root)
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, f'OOM_{datetime.now():%Y%m%d_%H%M%S}'), 'w').write(traceback.format_exc())
                torch.cuda.empty_cache()
                import shutil
                shutil.rmtree(os.path.join(claims, tag), ignore_errors=True)
                time.sleep(600)
            except Exception:
                flag = 'EXCEPTION'
                d = run_dir_of(int(K), int(bits), variant, int(seed), root)
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, 'EXCEPTION'), 'a').write(traceback.format_exc())
                traceback.print_exc()
            print(f'[worker] {tag}: {flag}', flush=True)
            torch.cuda.empty_cache()
            break                       # re-read the queue: priorities may have been edited
        if not took:
            print('[worker] queue empty', flush=True)
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int)
    ap.add_argument('--bits', type=int)
    ap.add_argument('--variant', choices=list(VARIANTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--n-train', type=int, default=100000)
    ap.add_argument('--root', default=OUT_ROOT)
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--queue', default=os.path.join(OUT_ROOT, 'queue.txt'))
    ap.add_argument('--claims', default=os.path.join(OUT_ROOT, 'claims'))
    a = ap.parse_args()
    if a.worker:
        worker(a.queue, a.claims, a.epochs, a.n_train, a.root, not a.no_compile)
    else:
        train_run(a.K, a.bits, a.variant, a.seed, a.epochs, a.n_train, a.root, not a.no_compile)


if __name__ == '__main__':
    main()
