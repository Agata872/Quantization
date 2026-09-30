"""Simulate and archive the paper's 16-QAM link: GNN versus a baseline precoder.

The baseline (--baseline) is IDE (Wang et al., TWC 2018, Alg. 1, as in
ber_16qam_20dB.py) with a calibrated fixed beta (the default and the paper
figure: per-symbol processing, like the GNN), with the fixed factor beta_WF or
with block beta, or WMMSE + DAC.
Every plotted IQ sample is read back from these archives. The channel is the
stored cell-free flat-fading channel; AWGN is actually sampled. The receiver
uses the paper's noiseless, data-aided block gain (an oracle, not pilot CSI).
See CONSTELLATION_16QAM.md for assumptions, array layouts and reproduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for directory in (HERE.parent, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from exp_cellfree_ablation import GNNv2, normalize_power
from train_sweep import M_FULL, PT, linear_precoder, load_levels, quantize_to_levels, run_model_warm
from wmmse_baseline import best_wmmse
import ide_baseline

BASELINES = {
    'ide_cal': 'IDE (Wang et al., TWC 2018, Alg. 1) with a calibrated fixed beta, as in ber_16qam_20dB.py: per '
               'channel realization, beta = ide_baseline.calibrated_beta(...) on an independent training block of '
               '125 16-QAM symbol vectors (seed --seed-cal, never the data), then T=100, alpha=0.95 with this beta '
               'fixed, every symbol vector processed independently; the last hard iterate on the DAC grid',
    'ide_wf':'IDE (Wang et al., TWC 2018, Alg. 1) via ide_baseline.ide(..., "ide", beta_mode="wf"), as in '
              'ber_16qam_20dB.py: T=100, alpha=0.95, beta fixed to beta_WF of eq. (7) for each channel realization '
              '(never updated), every symbol vector processed independently; the last hard iterate on the DAC grid',
    'ide_block':'IDE (Wang et al., TWC 2018, Alg. 1) via ide_baseline.ide(..., "ide", beta_mode="block"), as in '
                 'ber_16qam_20dB.py: T=100, alpha=0.95, beta of (26) with numerator and denominator summed over '
                 'the 125 symbols of each block, updated every 10 iterations; the last hard iterate on the DAC grid',
    'wmmse': 'K=1 exact MRT; K=2 best RZF/MRT/ZF start, each 150 iterations and 80 power bisections; '
             'nearest DAC level after per-AP input normalization',
}
GRAY = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.uint8)
PAM = torch.tensor([-3., -1., 1., 3.]) / math.sqrt(10)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2 ** 20), b''):
            digest.update(block)
    return {'path': str(Path(path).resolve()), 'sha256': digest.hexdigest()}


def unique_path(pattern):
    matches = sorted(HERE.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one file for {pattern}, found {len(matches)}: {matches}')
    return matches[0]


def checkpoint_path(users, bits):
    pattern = ('stored_models_cellfree_k1_continue/*/model_best.pt' if (users, bits) == (1, 1)
               else f'stored_models_cellfree_sweep_continue/M40_K{users}_b{bits}_*/model_best.pt')
    return unique_path(pattern)


def load_model(path, users, bits, levels, device):
    # These are the trusted local training artifacts used by the paper scripts.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    if cfg.get('K', users) != users or cfg.get('bits', bits) != bits:
        raise ValueError(f'Checkpoint configuration disagrees with K={users}, b={bits}')
    if cfg.get('no_warm_start', False) or cfg.get('refine', 1) != 1:
        raise ValueError('This experiment requires the single-pass warm-start model')
    model = GNNv2(M_FULL, users, cfg['dl'], cfg['layers'], bits,
                  cfg['tau_final'], levels, input_mode='polar4',
                  feat_stats=checkpoint['feat_stats'], output_type='argmax',
                  logit_norm=cfg.get('logit_norm')).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, {**fingerprint(path), 'epoch': int(checkpoint['epoch']),
                   'configuration': cfg, 'feat_stats': checkpoint['feat_stats'],
                   'training_symbols': 'CN(0,1); no 16-QAM retraining',
                   'inference': 'deterministic argmax, one quantized MRT/ZF warm start'}


def qam16(count, users, symbols, seed):
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, 4, (count, users, symbols, 2), generator=rng)
    signal = torch.complex(PAM[indices[..., 0]], PAM[indices[..., 1]])
    bits = torch.cat((GRAY[indices[..., 0]], GRAY[indices[..., 1]]), dim=-1)
    return signal, bits


def demap(signal):
    thresholds = torch.tensor([-2., 0., 2.], device=signal.device) / math.sqrt(10)
    i = torch.bucketize(signal.real.contiguous(), thresholds)
    q = torch.bucketize(signal.imag.contiguous(), thresholds)
    gray = GRAY.to(signal.device)
    return torch.cat((gray[i], gray[q]), dim=-1)


def analytic_ber(noiseless_equalized, sent, gain, noise_var):
    """Conditional Gray 16-QAM BER, integrating only AWGN; per block/user."""
    sd = torch.sqrt(noise_var / (2 * gain.abs().square()))[..., None]
    boundary = 2 / math.sqrt(10)

    def q(value):
        return 0.5 * torch.erfc(value / math.sqrt(2))

    def dimension(received, reference):
        sign_error = q(torch.sign(reference) * received / sd)
        inner_error = q((boundary - received) / sd) + q((boundary + received) / sd)
        outer_error = q((-boundary - received) / sd) - q((boundary - received) / sd)
        amplitude_error = torch.where(reference.abs() < boundary, inner_error, outer_error)
        return (sign_error + amplitude_error) * 0.5

    return (0.5 * (dimension(noiseless_equalized.real, sent.real)
                   + dimension(noiseless_equalized.imag, sent.imag))).mean(-1).clamp(0, 1)


def save_npz(path, **arrays):
    arrays = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
              for key, value in arrays.items()}
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@torch.inference_mode()
def infer_gnn(model, H, symbols, levels, device, symbol_chunk=125, channel_chunk=16):
    """Bound activation memory even when the requested frame is long."""
    result = torch.empty(H.shape[0], M_FULL, symbols.shape[-1], dtype=torch.complex64)
    last_update = time.monotonic()
    for start in range(0, H.shape[0], channel_chunk):
        stop = min(start + channel_chunk, H.shape[0])
        channels = H[start:stop].to(device)
        for offset in range(0, symbols.shape[-1], symbol_chunk):
            data = symbols[start:stop, :, offset:offset + symbol_chunk].to(device)
            output = run_model_warm(model, channels, data, levels, PT, channel_chunk)
            result[start:stop, :, offset:offset + symbol_chunk] = output.cpu()
        if time.monotonic() - last_update > 20:
            print(f'  GNN inference {stop}/{len(H)} blocks', flush=True)
            last_update = time.monotonic()
    return result


@torch.inference_mode()
def ide_dac(H, symbols, levels, snr_db, beta_mode, chunk=128, beta_fixed=None):
    """IDE on the DAC alphabet (unscaled, for receive()); beta_mode 'wf', 'block' or
    'fixed' (beta_fixed: one value per block, e.g. from ide_baseline.calibrated_beta).

    Same scaling as ide_baseline.precode(), whose power-normalized output would
    hide the DAC levels: the grid is divided by sqrt(Pt), so sigma^2 = 1/SNR."""
    grid = levels.double() / PT ** 0.5
    sig2 = 10 ** (-snr_db / 10)
    result = []
    for start in range(0, H.shape[0], chunk):
        A = H[start:start + chunk].transpose(1, 2).to(torch.complex128)
        S = symbols[start:start + chunk].transpose(1, 2).to(torch.complex128)
        bf = None if beta_fixed is None else beta_fixed[start:start + chunk]
        x = ide_baseline.ide(A, S, grid, sig2, 'ide', beta_mode=beta_mode, beta_fixed=bf).transpose(1, 2)
        real = (x.real[..., None] - grid).abs().argmin(-1)
        imag = (x.imag[..., None] - grid).abs().argmin(-1)
        result.append(torch.complex(levels[real], levels[imag]))
    return torch.cat(result)


def summarize(case, sent, users, bits, method, n_eval, ns):
    summaries = []
    for scope, selection in [('ensemble', slice(0, n_eval)), ('snapshot', slice(n_eval, None))]:
        if scope == 'snapshot' and len(sent) == n_eval:
            continue
        for user in range(users):
            z = case['rx_equalized'][selection, user]
            s = sent[selection, user]
            n_symbols = s.numel()
            errors = int(case['bit_errors'][selection, user].sum())
            ser_errors = int(case['symbol_errors'][selection, user].sum())
            rms = torch.sqrt((z - s).abs().square().sum() / s.abs().square().sum())
            summaries.append({
                'scope': scope, 'K': users, 'user': user + 1, 'bits': bits, 'method': method,
                'n_blocks': n_symbols // ns, 'n_symbols': n_symbols, 'n_bits': 4 * n_symbols,
                'bit_errors': errors, 'ber_mc': errors / (4 * n_symbols),
                'ber_analytic': float(case['ber_analytic'][selection, user].mean()),
                'symbol_errors': ser_errors, 'ser_mc': ser_errors / n_symbols,
                'evm_rms_percent': 100 * float(rms),
                'mc_one_bit_resolution': 1 / (4 * n_symbols),
            })
    return summaries


@torch.inference_mode()
def receive(dac_output, levels, H, symbols, bits, noise, noise_var):
    """DAC -> common block power scale -> H^T -> AWGN -> block EQ -> bits."""
    tx = normalize_power(dac_output, PT)
    scale = torch.sqrt(PT / (dac_output.abs().square().sum(1).mean(-1) + 1e-7))
    dac_indices = torch.stack(((dac_output.real[..., None] - levels).abs().argmin(-1),
                               (dac_output.imag[..., None] - levels).abs().argmin(-1)), -1).to(torch.uint8)
    reconstructed = torch.complex(levels[dac_indices[..., 0].long()],
                                  levels[dac_indices[..., 1].long()])
    if not torch.equal(reconstructed, dac_output):
        raise RuntimeError('Precoder produced a value outside the declared DAC alphabet')
    r0 = H.transpose(1, 2) @ tx  # physical convention is transpose, not Hermitian transpose
    r = r0 + noise
    gain = (r0 * symbols.conj()).sum(-1) / symbols.abs().square().sum(-1)
    if torch.any(gain.abs() < 1e-12):
        raise RuntimeError('A user has zero effective gain; scalar equalization is undefined')
    z = r / gain[..., None]
    detected = demap(z)
    wrong = detected != bits
    case = {
        'tx_samples': tx, 'dac_indices': dac_indices, 'levels': levels, 'power_scale': scale,
        'rx_noiseless': r0, 'rx_raw': r, 'rx_equalized': z, 'gain': gain,
        'rx_bits': detected, 'bit_errors': wrong.sum((-1, -2)),
        'symbol_errors': wrong.any(-1).sum(-1),
        'evm_rms': torch.sqrt((z - symbols).abs().square().sum(-1) / symbols.abs().square().sum(-1)),
        'ber_analytic': analytic_ber(r0 / gain[..., None], symbols, gain, noise_var),
        'tx_power': tx.abs().square().sum(1).mean(-1),
    }
    if not all(torch.isfinite(value).all() for value in case.values()):
        raise RuntimeError('Nonfinite simulation result')
    return case


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='ide_cal', choices=list(BASELINES))
    p.add_argument('--output-dir', type=Path, default=None,
                   help='default: exp_results/constellation_16qam_20dB_<baseline> for IDE, '
                        'exp_results/constellation_16qam_20dB for WMMSE (the earliest run)')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--users', type=int, nargs='+', default=[1, 2], choices=[1, 2])
    p.add_argument('--bits', type=int, nargs='+', default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument('--snr-db', type=float, default=20.)
    p.add_argument('--n-channels', type=int, default=2048)
    p.add_argument('--start', type=int, default=4096)
    p.add_argument('--symbols', type=int, default=125)
    p.add_argument('--snapshot-blocks', type=int, default=32)
    p.add_argument('--seed-symbols', type=int, default=1234)
    p.add_argument('--seed-noise', type=int, default=20260930)
    p.add_argument('--seed-cal', type=int, default=777,
                   help='training symbols of the calibrated IDE beta (as SEED_CAL of ber_16qam_20dB.py)')
    p.add_argument('--no-plots', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if min(args.n_channels, args.symbols, args.threads) <= 0 or min(args.start, args.snapshot_blocks) < 0:
        raise ValueError('Counts/threads must be positive; start/snapshot-blocks must be nonnegative')
    if not math.isfinite(args.snr_db):
        raise ValueError('SNR must be finite')
    if args.output_dir is None:
        suffix = '' if args.baseline == 'wmmse' else f'_{args.baseline}'
        args.output_dir = HERE / 'exp_results' / f'constellation_16qam_{args.snr_db:g}dB{suffix}'
    methods = ('gnn', args.baseline)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to replace an existing experiment: {output}; use a new --output-dir')
    # Resolve every source before creating output; missing models must never become random networks.
    source_channels = {k: unique_path(f'datasets/cellfree/M_40_K_{k}_Ntr_200000_*/Htest.npy')
                       for k in args.users}
    source_models = {(k, b): checkpoint_path(k, b) for k in args.users for b in args.bits}
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    noise_var = PT / 10 ** (args.snr_db / 10)
    commit = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    metadata = {
        'status': 'running', 'm': M_FULL, 'pt': PT, 'snr_db': args.snr_db,
        'noise_variance_complex': noise_var, 'snr_definition': 'Pt / E[abs(n)^2], transmit reference',
        'users': args.users, 'bits': args.bits, 'n_eval': args.n_channels, 'ns': args.symbols,
        'start': args.start, 'snapshot_blocks': args.snapshot_blocks,
        'snapshot_channel_index': args.start, 'snapshot_selection': 'first evaluation channel, fixed before inference',
        'seed_symbols': args.seed_symbols, 'seed_noise': args.seed_noise, 'seed_cal': args.seed_cal,
        'modulation': 'unit-average-energy Gray 16-QAM; [-3,-1,1,3]/sqrt(10) -> [00,01,11,10]',
        'bit_axis': ['I_sign', 'I_inner_outer', 'Q_sign', 'Q_inner_outer'],
        'receiver': 'paper oracle per-block gain g=sum(rx_noiseless*conj(s))/sum(abs(s)^2); z=rx_raw/g',
        'channel': 'stored cell-free flat block fading: H^T x + n, perfect TX CSI and symbol timing',
        'channel_geometry': {'area_m': [100, 100], 'minimum_distance_m': 10,
                             'pathloss_db': '-30 - 37*log10(d)', 'shadowing_std_db': 8,
                             'small_scale': 'CN(0,1)', 'normalization': 'original full-dataset common scalar'},
        'scope': 'symbol-rate equivalent complex baseband; no pulse shaping, CFO, phase drift or timing acquisition',
        'power_normalization': 'one common scalar per 125-symbol block by default, no per-AP analog gain restoration',
        'comparison': 'shared channels, 16-QAM symbols and actual AWGN draws for all methods/bits at each K',
        'evaluation_caveat': 'Htest[4096:6144] was also used for checkpoint selection in the paper; not a new independent test set',
        'baseline': args.baseline, 'methods': list(methods), args.baseline: BASELINES[args.baseline],
        'software': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                     'device': str(device), 'threads': args.threads, 'git_commit': commit},
        'sources': {name: fingerprint(HERE / name) for name in
                    ['constellation_16qam.py', 'exp_cellfree_ablation.py', 'train_sweep.py',
                     'wmmse_baseline.py', 'ide_baseline.py', 'data_handling.py', 'model.py']},
        'datasets': {}, 'checkpoints': {}, 'case_files': [],
    }
    write_json(output / 'metadata.json', metadata)
    started = time.perf_counter()
    rows = []
    for users in args.users:
        path = source_channels[users]
        dataset = np.load(path, mmap_mode='r')
        stop = args.start + args.n_channels
        if stop > len(dataset):
            raise ValueError(f'Requested channels [{args.start}:{stop}] exceed {path} ({len(dataset)})')
        indices = np.concatenate((np.arange(args.start, stop), np.full(args.snapshot_blocks, args.start)))
        H = torch.from_numpy(np.array(dataset[indices], dtype=np.complex64))
        symbols, tx_bits = qam16(len(H), users, args.symbols, args.seed_symbols)
        rng = torch.Generator().manual_seed(args.seed_noise + users)
        normal = torch.randn((*symbols.shape, 2), generator=rng) * math.sqrt(noise_var / 2)
        noise = torch.complex(normal[..., 0], normal[..., 1])
        metadata['datasets'][str(users)] = {**fingerprint(path), 'shape': list(dataset.shape),
                                          'actual_noise_seed': args.seed_noise + users}
        save_npz(output / f'shared_K{users}.npz', H=H, tx_symbols=symbols, tx_bits=tx_bits,
                 noise=noise, channel_indices=indices, n_eval=np.int64(args.n_channels),
                 snapshot_channel_index=np.int64(args.start), noise_variance=np.float64(noise_var))
        print(f'K={users}: {args.n_channels} evaluation blocks + {args.snapshot_blocks} snapshot blocks', flush=True)
        if args.baseline == 'wmmse':
            # Small K makes the repository CPU solver faster than launching many small GPU kernels.
            with torch.inference_mode():
                V_eval = linear_precoder(H[:args.n_channels], PT) if users == 1 else best_wmmse(H[:args.n_channels], noise_var)
                V = torch.cat((V_eval, V_eval[:1].expand(args.snapshot_blocks, -1, -1)), dim=0)
                row_norm = torch.linalg.vector_norm(V, dim=2, keepdim=True).clamp_min(1e-12)
                linear_input = (V @ symbols) / row_norm
        for resolution in args.bits:
            levels_cpu = load_levels(resolution, torch.device('cpu'))
            model, info = load_model(source_models[users, resolution], users, resolution,
                                     levels_cpu.to(device), device)
            metadata['checkpoints'][f'K{users}_b{resolution}'] = info
            for method in methods:
                clock = time.perf_counter()
                if method == 'gnn':
                    dac = infer_gnn(model, H, symbols, model.levels, device)
                elif method in ('ide_wf', 'ide_block'):
                    # beta_WF depends on H only, so the snapshot blocks share it; with block
                    # beta, each snapshot block gets its own beta, like every evaluation block.
                    dac = ide_dac(H, symbols, levels_cpu, args.snr_db, method.removeprefix('ide_'))
                elif method == 'ide_cal':
                    # One calibrated beta per channel realization, from training symbols that are
                    # independent of the data; the snapshot blocks share the beta of their channel.
                    train, _ = qam16(args.n_channels, users, args.symbols, args.seed_cal)
                    beta_eval = ide_baseline.calibrated_beta(H[:args.n_channels], train, levels_cpu, args.snr_db)
                    beta = torch.cat((beta_eval, beta_eval[:1].expand(args.snapshot_blocks)))
                    dac = ide_dac(H, symbols, levels_cpu, args.snr_db, 'fixed', beta_fixed=beta)
                else:
                    dac = quantize_to_levels(linear_input, levels_cpu)
                case = receive(dac, levels_cpu, H, symbols, tx_bits, noise, noise_var)
                name = f'K{users}_b{resolution}_{method}.npz'
                extra = ({'precoder_matrix': V} if method == 'wmmse' else
                         {'ide_beta': beta} if method == 'ide_cal' else {})
                save_npz(output / name, **case, **extra)
                new_rows = summarize(case, symbols, users, resolution, method, args.n_channels, args.symbols)
                rows.extend(new_rows)
                metadata['case_files'].append(name)
                write_json(output / 'metadata.json', metadata)
                displayed = [r for r in new_rows if r['scope'] == 'ensemble']
                print(f'K={users} b={resolution} {method}: ' + ' | '.join(
                    f"UE{r['user']} EVM={r['evm_rms_percent']:.2f}% BER={r['ber_mc']:.4g}"
                    for r in displayed) + f' ({time.perf_counter() - clock:.1f}s)', flush=True)
                del case, dac
            del model
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'summary.json', {'results': rows,
               'definitions': {'evm_rms_percent': '100*sqrt(sum(abs(rx_equalized-tx_symbols)^2)/sum(abs(tx_symbols)^2))',
                               'ber_mc': 'actual hard-decoded Gray bit errors / transmitted bits',
                               'ber_analytic': 'conditional bit error probability integrated over AWGN',
                               'mc_one_bit_resolution': 'one observed bit error divided by total bits; zero observed errors is not zero true BER'}})
    metadata['simulation_seconds'] = time.perf_counter() - started
    metadata['status'] = 'simulated'
    write_json(output / 'metadata.json', metadata)
    if not args.no_plots:
        from plot_constellation_16qam import plot_all
        plot_all(output)
    metadata['status'] = 'complete'
    write_json(output / 'metadata.json', metadata)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
