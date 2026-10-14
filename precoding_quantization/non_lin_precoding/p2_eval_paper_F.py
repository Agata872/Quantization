"""Prompt 2, F (proposed GNN): the EXISTING paper checkpoints scored on the frozen test set -- no retraining
(user decision, 2026-10-14). Two checkpoints per (K, b), both seed 0 of the paper recipe (train_sweep.py, K >= 2):
  F_ep15   stored_models_cellfree_sweep_K_bits/M40_K{K}_b{b}_ep15_*/model_best.pt   15 epochs, the budget the
           original GNN (A) gets in p2_ablation.py: same 100k channels, 128 per step, same validation slice
  F_paper  stored_models_cellfree_sweep_continue/M40_K{K}_b{b}_*/model_best.pt      15 + 15 epochs, the model of
           the paper's figures and tables
Same protocol as the p2_ablation.py test stage (Htest[6144:8192], stored Gaussian symbols, argmax, per-block power
normalization, LS estimator, -30:30:5 dB, per-user rates at 20/30 dB, validation rate at 20 dB), written in the same
file layout (test_rates.npz, summary.json) under ablation_p2/paper_F/K{K}b{b}/{F_ep15,F_paper}/, so the same paired
comparisons apply. The validation rate must reproduce the value stored with the checkpoint (assert).

  python p2_eval_paper_F.py          # CPU; THREADS env sets torch threads
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_cellfree_ablation import GNNv2, sumrate_bussgang_ls, normalize_power  # noqa: E402
from train_sweep import load_levels, run_model_warm, EVAL, PT, M_FULL  # noqa: E402
from p1_independent_eval import load_slice, sha256_file, TEST, SNR_GRID  # noqa: E402
from p2_ablation import per_user_ls, NV20  # noqa: E402

DEV = torch.device('cpu')
OUT = os.path.join(HERE, 'ablation_p2', 'paper_F')
SRC = {'F_ep15': 'stored_models_cellfree_sweep_K_bits/M40_K{K}_b{b}_ep15_*/model_best.pt',
       'F_paper': 'stored_models_cellfree_sweep_continue/M40_K{K}_b{b}_*/model_best.pt'}


def one(pattern):
    ps = glob.glob(os.path.join(HERE, pattern))
    assert len(ps) == 1, (pattern, ps)
    return ps[0]


@torch.no_grad()
def outputs(model, H, s, lv):
    return torch.cat([normalize_power(run_model_warm(model, H[i:i + 256], s[i:i + 256], lv, PT, 16), PT)
                      for i in range(0, H.shape[0], 256)])


def main():
    torch.set_num_threads(int(os.environ.get('THREADS', 16)))
    for K, b in [(2, 1), (2, 3), (6, 1), (6, 3)]:
        Ht, st = load_slice(K, TEST)
        Hv, sv = load_slice(K, EVAL)
        lv = load_levels(b, DEV)
        for name, pat in SRC.items():
            d = os.path.join(OUT, f'K{K}b{b}', name)
            if os.path.exists(os.path.join(d, 'summary.json')):
                continue
            os.makedirs(d, exist_ok=True)
            path = one(pat.format(K=K, b=b))
            ck = torch.load(path, map_location=DEV, weights_only=False)
            m = GNNv2(M_FULL, K, 128, 4, b, 0.25, lv, input_mode='polar4', feat_stats=ck['feat_stats'])
            m.load_state_dict(ck['model'])
            m.output_type = 'argmax'
            m.eval()
            yv = outputs(m, Hv, sv, lv)
            val20 = sumrate_bussgang_ls(yv, Hv, sv, NV20).numpy()
            # stored validation value: LS for continued runs; the ep15 runs logged the biased one, so compare with
            # the LS 20 dB point of their re-scored snr_curves.json instead
            if name == 'F_paper':
                ref = ck['eval_rate']
            else:
                c = json.load(open(os.path.join(os.path.dirname(path), 'snr_curves.json')))
                ref = c['gnn'][c['snr_db'].index(20.0)]
            assert abs(val20.mean() - ref) < 2e-3, (K, b, name, val20.mean(), ref)
            y = outputs(m, Ht, st, lv)
            rates = torch.stack([sumrate_bussgang_ls(y, Ht, st, PT / 10 ** (x / 10)) for x in SNR_GRID]).numpy()
            np.savez(os.path.join(d, 'test_rates.npz'), snr_db=np.array(SNR_GRID), sum_rate=rates,
                     per_user_20db=per_user_ls(y, Ht, st, NV20).numpy(),
                     per_user_30db=per_user_ls(y, Ht, st, PT / 1e3).numpy(), val_sum_rate_20db=val20)
            i20, i30 = SNR_GRID.index(20.0), SNR_GRID.index(30.0)
            summ = {'checkpoint': os.path.relpath(path, HERE), 'checkpoint_sha256': sha256_file(path),
                    'epochs_trained': 15 if name == 'F_ep15' else 30, 'best_epoch': int(ck['epoch']) + 1,
                    'val_rate_ls': float(val20.mean()), 'test_mean_20db': float(rates[i20].mean()),
                    'test_p5_20db': float(np.percentile(rates[i20], 5)), 'test_mean_30db': float(rates[i30].mean()),
                    'retrained': False, 'seed': 0}
            json.dump(summ, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
            print(f"K{K}b{b} {name}: val {summ['val_rate_ls']:.4f} (stored {ref:.4f}) | test 20 dB mean "
                  f"{summ['test_mean_20db']:.4f} p5 {summ['test_p5_20db']:.4f} | 30 dB {summ['test_mean_30db']:.4f}",
                  flush=True)


if __name__ == '__main__':
    main()
