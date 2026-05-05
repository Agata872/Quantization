# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on **quantization of GNN-based precoding for MIMO wireless communication systems**. The goal is to train neural networks that output quantized transmit signals while optimizing sum-rate (capacity). Two parallel implementations exist: one in PyTorch (primary, actively developed) and one in TensorFlow/Keras (older).

## Setup

Install dependencies from the requirements file:
```
pip install -r precoding_quantization/requirements.txt
```

Key packages: TensorFlow 2.13.0, PyTorch, NumPy 1.24.3, SciPy 1.10.1, SymPy, NetworkX, Matplotlib, Seaborn, tikzplotlib, tqdm, pytest 8.1.1.

## Running Training

**PyTorch GNN/MLP training (primary):**
```
cd precoding_quantization
python non_lin_precoding/training.py
```

Configure by editing the `__main__` block at the bottom of [precoding_quantization/non_lin_precoding/training.py](precoding_quantization/non_lin_precoding/training.py). Key parameters:
- `M` / `K`: number of antennas / users
- `bits`: quantization resolution (1–4 bits)
- `model_type`: `'GNN'` or `'MLP'`
- `output_type`: `'gumbel_softmax_hard'`, `'softmax_hard'`, `'softmax'`, `'gumbel_softmax'`
- `channel_model`: `'iid'`, `'los'`, or `'cellfree'`
- `quant`: `True`/`False` to enable or disable quantization

**TensorFlow GNN training (legacy):**
```
cd neuralnet_quantization
python gnn/training.py
```

## Running Tests

```
pytest
```

Most validation is done via standalone scripts rather than pytest — see `precoding_quantization/checks/` and `derivations/checks/` for numerical verification scripts.

## Architecture

### `precoding_quantization/` — PyTorch module (primary)

- **`non_lin_precoding/training.py`** — Main training loop. Trains GNN/MLP to learn a precoding map H, s → x̂ (quantized transmit vector). Applies power normalization, evaluates against ZF/MRT baselines. Saves model checkpoints and TikZ plots to `stored_models_*/`.
- **`non_lin_precoding/model.py`** — `GNNmodel`, `MLPmodel`, `MLPmodel_noquant`, `SumRateLoss`, `SumRateLoss_generalized_Bussgang`. The GNN model stacks `GNN_layer_fast` layers over a bipartite antenna–user graph.
- **`non_lin_precoding/data_handling.py`** — `ChannelSymbolsDataset` (PyTorch Dataset); `getdata_nonlinprec()` loads/generates channel matrices and symbols.
- **`MIMO_sims/Rsum_all.py`** — Bussgang-based sum-rate computation (`Rsum_Bussgang_Rx`), used for evaluation against baselines.
- **`utils/quantization.py`** — Uniform and non-uniform (Lloyd-Max) quantizers.
- **`utils/precoding.py`** — Zero-forcing (`ZF_precoding`) and MRT (`MRT_precoding`) baseline precoders.
- **`utils/utils.py`** — Channel generation (`rayleigh_channel_MU`, `los_channel_MU`), symbol generation, folder/logging helpers.
- **`GNN/`** — TF/Keras GNN layers, losses, callbacks (older, used by `neuralnet_quantization`).
- **`checks/`** — Standalone numerical validation scripts (quantizer design, polynomial fitting, etc.).

### `neuralnet_quantization/` — TensorFlow module (legacy)

- **`gnn/training.py`** — TF training entry point, builds Keras `Sequential` GNN model with `Efficient_GNN_layer` / `GNN_layer`.
- **`gnn/model.py`** — Keras custom layers for the GNN.
- **`gnn/losses.py`** — `polynomial_loss` and other Bussgang-based loss functions.
- **`utils/utils.py`** — Data utilities for TF pipeline.

### `PA_test_numerical/` — Power amplifier MLP fitting

Standalone scripts to fit MLPs to power amplifier (PA) models; independent from the main training pipeline.

### `derivations/`

LaTeX source and PDFs for the Bussgang decomposition paper. `checks/` contains Python scripts that numerically validate analytical results.

### `non-uniform-quant-params/`

Pre-computed Lloyd-Max quantizer parameters (`.npy` files) organized as `Gaussian_var_{varx}/numerical/{bits}bits_outputlevels.npy`. Referenced at runtime via `quant_params_path` in `sim_params`.

## Key Design Patterns

- **Graph structure**: The MIMO channel forms a bipartite graph — M antenna nodes, K user nodes, MK edge features (channel coefficients). GNN layers pass messages over this graph.
- **Quantization output**: The GNN outputs a probability distribution over discrete output levels using Gumbel-softmax or straight-through softmax, enabling differentiable discrete outputs during training.
- **Power constraint**: After the forward pass, outputs are normalized so `E[||x||²] = Pt` before computing loss.
- **Loss**: Generalized Bussgang sum-rate loss (`SumRateLoss_generalized_Bussgang`) decomposes the quantized output via the Bussgang theorem to estimate achievable rate.
- **Path handling**: Scripts set `PROJECT_ROOT` and `REPO_ROOT` dynamically via `__file__`; all dataset/model paths are relative to these roots, not hardcoded.
