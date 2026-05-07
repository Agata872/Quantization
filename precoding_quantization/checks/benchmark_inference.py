"""
Benchmark inference latency for all trained models under a given base directory.

Set BASE_DIR below, then run:
    python checks/benchmark_inference.py
"""
import os
import sys
import glob
import time
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
NON_LIN_DIR  = os.path.join(PROJECT_ROOT, 'non_lin_precoding')
for _p in (PROJECT_ROOT, NON_LIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
# Import via the bare 'model' module so pickle module paths match training.py,
# which also runs with non_lin_precoding/ on sys.path and uses 'from model import ...'.
from model import GNNmodel, GNNmodel_QAT, MLPmodel  # noqa: E402 (resolved via sys.path above)

# ── configure here ────────────────────────────────────────────────────────────
BASE_DIR        = r'stored_models_iid_generalized_bussgang_loss/M_32_K_4_bs_128_layers_4_dl_128_tau_1'
BATCH_SIZE      = 128
N_WARMUP        = 20
N_RUNS          = 200
# Force all models onto CPU so INT8 (CPU-only) and FP32 are measured on the same hardware.
# Set to False to let FP32 models run on GPU — faster but not comparable to INT8.
FORCE_CPU       = True
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CLS = {'GNN': GNNmodel, 'GNN_QAT': GNNmodel_QAT, 'MLP': MLPmodel}


def _load_config(model_dir):
    cfg = {}
    for fname in ('sim_params.json', 'train_params.json'):
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                cfg.update(json.load(f))
    return cfg


def _find_int8_checkpoint(model_dir):
    """Return the INT8 full-model file (model_*_int8.pt), or None."""
    matches = glob.glob(os.path.join(model_dir, 'model_*_int8.pt'))
    return matches[0] if matches else None


def _find_float_checkpoint(model_dir):
    """Return the float32 state-dict file, excluding INT8 and non-weight files."""
    matches = glob.glob(os.path.join(model_dir, 'model_*'))
    matches = [p for p in matches
               if not any(p.endswith(ext) for ext in ('.json', '.pdf', '.tex', '_int8.pt'))]
    return matches[0] if matches else None


def _make_inputs(M, K, device):
    H      = torch.randn(BATCH_SIZE, M, K, dtype=torch.complex64).to(device)
    s      = torch.randn(BATCH_SIZE, K, dtype=torch.complex64).to(device)
    x_init = torch.zeros(BATCH_SIZE, M, 2).to(device)
    return H, s, x_init


def _time_model(model, H, s, x_init):
    is_gnn = isinstance(model, (GNNmodel, GNNmodel_QAT))
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(H, s, x_init) if is_gnn else model(H, s)
        if H.is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(H, s, x_init) if is_gnn else model(H, s)
            if H.is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def benchmark_dir(model_dir, device):
    cfg = _load_config(model_dir)
    if not cfg:
        return None

    M                = cfg['M']
    K                = cfg['K']
    bits             = cfg['bits']
    nr_features      = cfg.get('nr_features', 128)
    nr_hidden_layers = cfg.get('nr_hidden_layers', 4)
    tau              = cfg.get('tau', 1)
    output_type      = cfg.get('output_type', 'gumbel_softmax_hard')
    model_type       = cfg.get('model_type', 'GNN')

    quant_params_path = os.path.join(PROJECT_ROOT, 'non-uniform-quant-params',
                                     'Gaussian_var_0.5', 'numerical')

    # GNN_QAT: load INT8 full-model if it exists (module paths are consistent because
    # both training.py and this script add non_lin_precoding/ to sys.path and import
    # from 'model', so pickle can resolve GNNmodel_QAT / GNN_layer_fast_qat correctly).
    int8_ckpt = _find_int8_checkpoint(model_dir) if model_type == 'GNN_QAT' else None
    if int8_ckpt:
        model = torch.load(int8_ckpt, map_location='cpu', weights_only=False)
        model.eval()
        inference_device = torch.device('cpu')
        weight_tag = 'INT8'
    else:
        inference_device = torch.device('cpu') if FORCE_CPU else device
        output_levels = torch.from_numpy(
            np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
        ).float()
        cls = MODEL_CLS[model_type]
        if model_type == 'MLP':
            model = cls(M, K, bits, tau, output_levels.to(inference_device)).to(inference_device)
        else:
            model = cls(M, K, nr_features, nr_hidden_layers, bits, tau,
                        output_levels.to(inference_device), quantize=True,
                        output_type=output_type).to(inference_device)
        float_ckpt = _find_float_checkpoint(model_dir)
        if float_ckpt:
            model.load_state_dict(torch.load(float_ckpt, map_location=inference_device, weights_only=True))
        model.eval()
        weight_tag = 'loaded' if float_ckpt else 'random'

    H, s, x_init = _make_inputs(M, K, inference_device)
    mean_ms, std_ms = _time_model(model, H, s, x_init)
    return mean_ms, std_ms, model_type, M, K, bits, weight_tag


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fp32_device = torch.device('cpu') if FORCE_CPU else device
    print(f'Available device: {device}')
    print(f'FP32 inference device: {fp32_device}  |  INT8 inference device: cpu (quantize_dynamic is CPU-only)')
    print(f'Base dir: {BASE_DIR}\n')

    subdirs = sorted([
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    if not subdirs:
        print('No subdirectories found.')
        sys.exit(1)

    print(f'{"Model folder":<50} {"Type":<10} {"Mean (ms)":>10}  {"Std (ms)":>9}  {"Weights"}')
    print('-' * 95)
    for d in subdirs:
        result = benchmark_dir(d, device)
        if result is None:
            print(f'{os.path.basename(d):<50}  (no config found, skipped)')
            continue
        mean_ms, std_ms, model_type, M, K, bits, weight_tag = result
        print(f'{os.path.basename(d):<50} {model_type:<10} {mean_ms:>10.3f}  {std_ms:>9.3f}  {weight_tag}')
