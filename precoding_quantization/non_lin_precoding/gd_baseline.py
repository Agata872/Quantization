"""Per-channel gradient-descent baseline for quantized non-linear precoding.

For one channel realization, the DAC level of every (symbol, AP, I/Q branch) is given its own
logit vector, and all of them are optimized jointly by Adam on exactly the objective the GNN is
trained with: the Bussgang sum rate estimated over the N_s-symbol block. The GNN is therefore
the amortized version of this procedure, and the gap between the two is the amortization gap.

Differences from the coordinate-descent reference (see exp_cellfree_ablation.coord_descent):
  * objective: the true sum rate here, versus distance to kappa * (linear received signal)
    with kappa grid-searched there;
  * variable: a softmax relaxation of the level choice here, versus exact search over the
    discrete alphabet there, so this method pays a rounding gap that CD does not;
  * scope: all N_s symbols of a block jointly here (the rate is a block statistic), versus
    each symbol independently there.

The straight-through estimator and the "keep the best hard iterate" rule mean the returned
solution is always on the DAC grid and never worse than the initialization.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(CURRENT_DIR), CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
from exp_cellfree_ablation import RATE_FNS, normalize_power  # noqa: E402

PT = 40.0


def _levels_to_y(p, levels):
    v = (p * levels).sum(-1)                       # ... x 2
    return torch.complex(v[..., 0], v[..., 1])


def gd_optimize(H, s, levels, noise_var, y_init, iters=300, lr=0.1, tau0=1.0, tau1=0.1,
                scale=2.0, estimator='st', pt=PT, rate='ls'):
    """H: B x M x K, s: B x K x Ns, y_init: B x M x Ns on the DAC grid.
    Returns (best hard y, its rate per channel, mean hard rate per iteration).
    `rate` selects the objective AND the score: 'ls' (default, unbiased) or 'biased'."""
    sumrate_bussgang = RATE_FNS[rate]
    L = len(levels)
    idx = torch.stack([(y_init.real.unsqueeze(-1) - levels).abs().argmin(-1),
                       (y_init.imag.unsqueeze(-1) - levels).abs().argmin(-1)], -1)
    logits = torch.zeros(*idx.shape, L, dtype=torch.float32, device=idx.device)
    logits.scatter_(-1, idx.unsqueeze(-1), scale)
    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)

    with torch.no_grad():
        best_y = y_init.clone()
        best_r = sumrate_bussgang(normalize_power(best_y, pt), H, s, noise_var)
    trace = [best_r.mean().item()]
    for it in range(iters):
        tau = tau0 * (tau1 / tau0) ** (it / max(1, iters - 1))
        p = F.softmax(logits / tau, -1)
        if estimator == 'st':
            hard = F.one_hot(p.argmax(-1), L).to(p.dtype)
            p = hard - p.detach() + p
        r = sumrate_bussgang(normalize_power(_levels_to_y(p, levels), pt), H, s, noise_var)
        opt.zero_grad()
        (-r.sum()).backward()
        opt.step()
        with torch.no_grad():
            yh = _levels_to_y(F.one_hot(logits.argmax(-1), L).to(logits.dtype), levels)
            rh = sumrate_bussgang(normalize_power(yh, pt), H, s, noise_var)
            better = rh > best_r
            best_r = torch.where(better, rh, best_r)
            best_y[better] = yh[better]
            trace.append(rh.mean().item())
    return best_y, best_r, trace
