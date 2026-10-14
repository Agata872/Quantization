# Original GNN of Feys et al. and its reproduction in the Prompt-2 ablation

Source checked on 2026-10-14: T. Feys, L. Van der Perre, F. Rottenberg, "Learning to Quantize and Precode in
Massive MIMO Systems for Energy Reduction: a Graph Neural Network Approach", arXiv:2507.10634v1 (14 Jul 2025),
Sec. IV (training, Gumbel-softmax, architecture) and Sec. VI-A (training parameters). Code: the training
pipeline this repository started from (`model.py` `GNNmodel` / `GNN_layer_fast`, `training.py`, first commit
a4bdfe6, 2026-03-12).

| Item | Feys et al. (paper) | Original code (a4bdfe6) | A_orig in this ablation |
|---|---|---|---|
| Edge input | [Re h_mk, Im h_mk], eq. (27) | same | [Re h, Im h, 0, 0] (zero padding, same function) |
| AP node input | [0, 0], eq. (28) | zeros (`x_init`) | [0, 0, 0, 0] |
| UE node input | [Re s_k, Im s_k], eq. (29) | same | [Re s, Im s, 0, 0] |
| Layers | input + N_h = 4 hidden (d_h = 128) + output | same | same (`P2GNN`, identical math to `GNN_layer_fast`) |
| Aggregation / activation | mean, LeakyReLU(0.01) | same | same |
| Output | 2L logits per AP, straight-through Gumbel-softmax, argmax at inference | same | same |
| Temperature | τ = 1 | τ = 1 (sweep list), 4 in a default line | τ = 1 constant |
| Optimizer | Adam, lr 5e-3, batch 128 | Adam, lr 5e-4, batch 64 (defaults in `__main__`) | Adam, lr 5e-3 constant (`A_orig`) and 1e-3 constant (`A_orig_lr1e-3`), batch 128; the better one on validation is used for seeds 1–2 |
| Loss | −R_sum with G = E{y s^H} (moment fit), power normalized over N_s = 125 | same | same (biased/moment fit, as every variant) |
| Data | 200k channels, 20 epochs, i.i.d. Rayleigh / LoS | same | cell-free channels of this paper, 100k channels, 15 epochs (common budget of all variants) |
| Levels | Lloyd–Max (b ≥ 2), ±sqrt(Pt/2M) for b = 1 | same | same |
| Model selection | hyperparameters on a 1000-channel validation set | best validation loss | best validation LS rate per epoch on Htest[4096:6144] (same rule for all variants) |

Adaptations (all variants share them): cell-free channel model and M = 40 of this paper; the per-variant budget
is 15 epochs over 100k channels (the paper's E = 15 of Table I) instead of 20 epochs over 200k; the 4-dimensional
input layer (zero columns only change the Xavier bound of the used weights by 0.8 %); `torch.compile` of the
backbone (same decisions as the uncompiled model on a paper checkpoint; logits within 1e-4).

Design labels used in the ablation: F = phase / log-magnitude edge features (eq:tp_edge_init), R = quantized ZF
reference at the AP nodes and its predicted reception at the UE nodes (eq:tp_ap_init, eq:tp_ue_init),
TS = this paper's training strategy (τ 1 → 0.25 over 3 epochs, logit L2 μ = 0.1, cosine lr 1e-3 → 5e-5).
