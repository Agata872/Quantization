import glob
import os
import sys

# Allow running this file directly (python non_lin_precoding/training.py)
# by adding the parent folder (precoding_quantization) to sys.path.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import webcolors

# Ensure a CSS3 hex->name mapping exists regardless of webcolors version.
# Some webcolors versions expose different constant names; try common variants
# and fall back to an empty dict if none are available.
try:
    css3_names_to_hex = getattr(webcolors, 'CSS3_NAMES_TO_HEX', None)
    css3_hex_to_names = getattr(webcolors, 'CSS3_HEX_TO_NAMES', None)
    if css3_hex_to_names is None and css3_names_to_hex is not None:
        webcolors.CSS3_HEX_TO_NAMES = {v: k for k, v in css3_names_to_hex.items()}
    elif css3_hex_to_names is None and css3_names_to_hex is None:
        # Try alternate attribute names used in other versions
        alt_maps = [
            'CSS21_NAMES_TO_HEX', 'HTML4_NAMES_TO_HEX',
            'css3_names_to_hex', 'css21_names_to_hex', 'html4_names_to_hex'
        ]
        found = False
        for alt in alt_maps:
            if hasattr(webcolors, alt):
                cmap = getattr(webcolors, alt)
                webcolors.CSS3_HEX_TO_NAMES = {v: k for k, v in cmap.items()}
                found = True
                break
        if not found:
            webcolors.CSS3_HEX_TO_NAMES = {}
except Exception:
    webcolors.CSS3_HEX_TO_NAMES = {}

import torch
import torch.nn as nn
import numpy as np
from utils.utils import rayleigh_channel_MU, getSymbols, create_folder, logparams
from tqdm import tqdm
from model import MLPmodel, SumRateLoss, MLPmodel_noquant, GNNmodel, GNNmodel_QAT, SumRateLoss_generalized_Bussgang
import matplotlib.pyplot as plt
from torchsummary import summary
from datetime import datetime
from MIMO_sims.Rsum_all import Rsum_Bussgang_Rx
from data_handling import getdata_nonlinprec, ChannelSymbolsDataset
import tikzplotlib

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def apply_phase_drift(y, sigma_theta):
    """Apply i.i.d. per-antenna RF-chain phase drift, after the DAC/power-normalization stage.

    Models distributed-MIMO RF-chain (free-running LO) phase noise, same mechanism validated in
    Phase_impact/phase_impact.ipynb: dtheta_m ~ N(0, sigma_theta^2) i.i.d. per antenna, fixed across
    all symbols of a given channel realization (batch element), but resampled on every call so that
    training sees a fresh drift realization per batch/epoch.

    y: bs x M x nr_symbols (complex), the actual (quantized, power-normalized) transmit signal
    sigma_theta: std dev [rad] of the per-antenna phase error; sigma_theta <= 0 disables drift (no-op)
    """
    if sigma_theta <= 0:
        return y
    bs, M = y.shape[0], y.shape[1]
    dtheta = sigma_theta * torch.randn(bs, M, device=y.device)
    phase = torch.polar(torch.ones_like(dtheta), dtheta).to(y.dtype)  # exp(1j*dtheta)
    return y * phase.unsqueeze(-1)


def normalize_outputs(outputs, Pt, norm_block_size):
    """Apply power normalization with per-block alpha.

    norm_block_size=None or >= Ns uses a single alpha over all symbols (original behavior).
    Otherwise normalizes each block of norm_block_size symbols independently,
    matching causal deployment where the full symbol sequence is unavailable.
    """
    Ns = outputs.shape[-1]
    epsilon = 1e-7
    if norm_block_size is None or norm_block_size >= Ns:
        l2_norm = torch.linalg.vector_norm(outputs, ord=2, dim=1)  # bs x Ns
        expt_x2 = torch.mean(l2_norm ** 2, dim=-1)                 # bs
        alpha = torch.sqrt(Pt / (expt_x2 + epsilon))
        return alpha[:, None, None] * outputs
    else:
        normalized = torch.zeros_like(outputs)
        for blk_start in range(0, Ns, norm_block_size):
            blk = outputs[:, :, blk_start:blk_start + norm_block_size]
            l2_norm = torch.linalg.vector_norm(blk, ord=2, dim=1)
            expt_x2 = torch.mean(l2_norm ** 2, dim=-1)
            alpha = torch.sqrt(Pt / (expt_x2 + epsilon))
            normalized[:, :, blk_start:blk_start + norm_block_size] = alpha[:, None, None] * blk
        return normalized


def compute_base_path(model_dir, M, K, batch_size, nr_hidden_layers, nr_features, tau, sigma_theta_deg):
    return os.path.join(os.getcwd(), model_dir,
                         f'M_{M}_K_{K}_bs_{batch_size}_layers_{nr_hidden_layers}_dl_{nr_features}_'
                         f'tau_{tau}_sigmatheta_{sigma_theta_deg:g}deg')


def find_resume_candidate(base_path, bits, output_type, model_type):
    """Look for a previous run of this exact (bits, output_type, model_type) combo under base_path.

    Returns ('done', run_dir) if a run already finished (has the final Rsum_testeset.pdf),
    ('resume', checkpoint_path) if a run left a per-epoch checkpoint to continue from,
    or (None, None) if there's nothing to pick up -- caller should train from scratch.
    """
    if model_type == 'GNN_QAT':
        prefix = f'{bits}_bits_GNN_QAT_{output_type}_'
    elif model_type == 'GNN':
        prefix = f'{bits}_bits_GNN_{output_type}_'
    elif model_type == 'MLP':
        prefix = f'{bits}_bits_MLP_'
    else:
        return None, None

    if not os.path.isdir(base_path):
        return None, None

    candidates = sorted(
        d for d in os.listdir(base_path)
        if d.startswith(prefix) and os.path.isdir(os.path.join(base_path, d))
    )
    if not candidates:
        return None, None

    run_dir = os.path.join(base_path, candidates[-1])  # most recent timestamp
    if os.path.exists(os.path.join(run_dir, 'Rsum_testeset.pdf')):
        return 'done', run_dir

    checkpoints = sorted(glob.glob(os.path.join(run_dir, 'checkpoint_*.pt')))
    if checkpoints:
        return 'resume', checkpoints[-1]

    return None, None


def train(sim_params, train_params, resume_from=None):

    # unpack simulation parameters
    M = sim_params['M']
    K = sim_params['K']
    Pt = sim_params['Pt']
    bits = sim_params['bits']
    noise_var = sim_params['noise_var']
    quant_params_path = sim_params['quant_params_path']
    quant = sim_params['quant']
    varx = sim_params['varx']
    root_dir = sim_params['root_dir']
    sigma_theta_deg = sim_params.get('sigma_theta_deg', 0.0)
    sigma_theta_rad = np.deg2rad(sigma_theta_deg)

    # unpack training parameters
    channel_model = train_params['channel_model']
    model_type = train_params['model_type']
    output_type = train_params['output_type']
    tau = train_params['tau']
    Ntr = train_params['Nr_train']
    Nval = train_params['Nr_val']
    Nte = train_params['Nr_test']
    nr_symbols_per_channel = train_params['nr_symbols_per_channel']
    batch_size = train_params['batch_size']
    nr_epochs = train_params['epochs']
    lr = train_params['lr']
    nr_hidden_layers = train_params['nr_hidden_layers']
    nr_features = train_params['nr_features']
    model_dir = train_params['stored_model_dir']
    norm_block_size = train_params.get('norm_block_size', nr_symbols_per_channel)
    sigma_theta_warmup_epochs = train_params.get('sigma_theta_warmup_epochs', max(1, nr_epochs // 2))
    nr_drift_mc_samples = train_params.get('nr_drift_mc_samples', 4)

    # folder for storing model
    checkpoint = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        timestamp = checkpoint['timestamp']  # reuse the interrupted run's timestamp/folder
        print(f'resuming from checkpoint: {resume_from}')
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = compute_base_path(model_dir, M, K, batch_size, nr_hidden_layers, nr_features, tau, sigma_theta_deg)

    # quantizer params
    if bits == 1:
        output_levels = np.sqrt(Pt / (2 * M)) * torch.Tensor([-1, 1])  # only valid for 1 bit case
    else:
        output_levels = torch.from_numpy(np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))).type(
            torch.float32)
        print(f'{output_levels=}')

    # set GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if GPU is available
    print(f'device: {device}')

    # load/generate the data
    datapath = os.path.join(PROJECT_ROOT, 'non_lin_precoding', 'datasets', channel_model)

    Htrain, Hval, Htest, strain, sval, stest = getdata_nonlinprec(nr_symbols_per_channel, datapath, M, K, Ntr, Nval,
                                                                  Nte, channel_model)
    trainset = ChannelSymbolsDataset(Htrain.astype(np.complex64), strain.astype(np.complex64),
                                     nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    validation_set = ChannelSymbolsDataset(Hval.astype(np.complex64), sval.astype(np.complex64),
                                           nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    test_set = ChannelSymbolsDataset(Htest.astype(np.complex64), stest.astype(np.complex64),
                                     nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    training_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # test
    H, s = next(iter(training_dataloader)) #bs x MK x output_feature_size
    print(f'{H.shape} - {s.shape}')

    # create model
    if model_type == 'GNN':
        model = GNNmodel(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels.to(device),
                         quantize=quant, output_type=output_type).to(device)
        if quant:
            name = f'{bits}_bits_GNN_{output_type}_{timestamp}'
        else:
            name = f'GNN_no_quant_{timestamp}'

        model_path = os.path.join(base_path, name)
        create_folder(model_path)

    elif model_type == 'GNN_QAT':
        model = GNNmodel_QAT(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels.to(device),
                              quantize=quant, output_type=output_type).to(device)
        if quant:
            name = f'{bits}_bits_GNN_QAT_{output_type}_{timestamp}'
        else:
            name = f'GNN_QAT_no_quant_{timestamp}'

        model_path = os.path.join(base_path, name)
        create_folder(model_path)

    elif model_type == 'MLP':
        if quant:
            model = MLPmodel(M, K, bits, tau, output_levels.to(device)).to(device)
            model_path = os.path.join(base_path, f'{bits}_bits_MLP_{timestamp}')
            create_folder(model_path)
        else:  # train without quantization
            model = MLPmodel_noquant(M, K).to(device)  # santity check
            model_path = os.path.join(base_path, f'MLP_no_quant_{timestamp}')
            create_folder(model_path)

    print(model)
    print(f'nr trainable params: {sum([param.nelement() for param in model.parameters()])}')

    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # get loss function
    loss_fn = SumRateLoss_generalized_Bussgang() #SumRateLoss()

    # containers to store loss
    loss_history = []
    vloss_history = []
    best_vloss = 0
    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_history = checkpoint['loss_history']
        vloss_history = checkpoint['vloss_history']
        best_vloss = checkpoint['best_vloss']
        start_epoch = checkpoint['epoch'] + 1
        print(f'resumed at epoch {start_epoch}/{nr_epochs} (best_vloss so far: {best_vloss})')

    x_init = torch.zeros((batch_size, M, 2)).to(device)  # zeros as initial input for antennanode features
    # loop over batches
    for epoch in range(start_epoch, nr_epochs):
        # curriculum: ramp the *ceiling* of the training-time sigma_theta range linearly from 0
        # (epoch 0) up to the full target value (reached at epoch sigma_theta_warmup_epochs-1, held
        # there after). Training on the full, large drift from initialization destabilizes
        # optimization -- see the M8/K1/3bit/40deg run that ended up *worse* than the ZF/MRT baseline
        # even before drift was applied at test time, vs. the same config converging normally at
        # sigma_theta_deg=0.
        # Within that ceiling, every batch draws its own sigma_theta ~ U(0, ceiling) instead of
        # training at one fixed value per epoch: a single checkpoint then has to stay good across the
        # whole severity range instead of overfitting to one operating point, so it should generalize
        # better to deployment drift levels that don't exactly match sigma_theta_deg. Validation
        # always uses the full target value (below) so best-checkpoint selection tracks the real
        # deployment condition, not the training-time sampling range.
        ramp = min(1.0, epoch / max(1, sigma_theta_warmup_epochs - 1))
        sigma_theta_rad_ceiling = sigma_theta_rad * ramp
        print(f'epoch {epoch}: sigma_theta_deg (train ceiling) = {np.rad2deg(sigma_theta_rad_ceiling):.1f} '
              f'(target {sigma_theta_deg:.1f}), sampled per-batch from U(0, ceiling)')
        running_loss = 0
        with tqdm(training_dataloader, unit='batch') as tqdmbatch:
            for i, batch in enumerate(tqdmbatch):
                H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel

                # move input data to the GPU
                H, s = H, s

                # set accumulated grads to zero
                optimizer.zero_grad()

                # forward pass
                outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
                for sidx in range(s.shape[-1]):
                    if model_type in ('GNN', 'GNN_QAT'):
                        outputs[:, :, sidx] = model(H, s[:, :, sidx], x_init)  # NN takes 1 channel and 1 symbol as input
                    else:
                        outputs[:, :, sidx] = model(H, s[:, :, sidx])  # NN takes 1 channel and 1 symbol as input

                normalized_output = normalize_outputs(outputs, Pt, norm_block_size)
                # print(f'{outputs=}')

                # sample this batch's drift severity, then average the loss over several independent
                # phase-drift realizations at that severity (Monte-Carlo estimate of E_theta[loss]):
                # a single realization gives a high-variance gradient since dtheta is resampled fresh
                # on every apply_phase_drift call.
                sigma_theta_rad_train = sigma_theta_rad_ceiling * float(torch.rand(1))
                mc_samples = nr_drift_mc_samples if sigma_theta_rad_train > 0 else 1
                loss = 0.0
                for _ in range(mc_samples):
                    drifted_output = apply_phase_drift(normalized_output, sigma_theta_rad_train)
                    loss = loss + loss_fn(drifted_output.to(device), H.type(torch.complex64),
                                          s.type(torch.complex64), noise_var)
                loss = loss / mc_samples

                # backprop + gradient descent step
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tqdmbatch.set_postfix(loss=running_loss / (i + 1))

        # save training loss after each epoch
        loss_history.append(running_loss / (i + 1))

        # validation loss
        model.eval()
        with torch.no_grad():
            running_vloss = 0
            for i, batch in enumerate(validation_dataloader):
                H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel
                bs = H.shape[0]
                x_init = torch.zeros((bs, M, 2)).to(device)  # zeros as initial input for antenna node features

                # move input data to the GPU
                H, s = H.to(device), s.to(device)

                # forward pass
                outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
                for sidx in range(s.shape[-1]):
                    outputs[:, :, sidx] = model(H, s[:, :, sidx], x_init)  # NN takes 1 channel and 1 symbol as input

                normalized_output = normalize_outputs(outputs, Pt, norm_block_size)

                # average over multiple drift realizations for a lower-variance validation metric
                # (same MC averaging as training), so best-checkpoint selection isn't noisy
                mc_samples = nr_drift_mc_samples if sigma_theta_rad > 0 else 1
                vloss = 0.0
                for _ in range(mc_samples):
                    drifted_output = apply_phase_drift(normalized_output, sigma_theta_rad)
                    vloss = vloss + loss_fn(drifted_output.to(device), H.type(torch.complex64),
                                            s.type(torch.complex64), noise_var)
                vloss = vloss / mc_samples
                running_vloss += vloss.item()

        # log the validation loss
        avg_vloss = running_vloss / (i + 1)
        print(f'avg vallidation loss: {avg_vloss}')
        vloss_history.append(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            path = os.path.join(model_path, 'model_{}'.format(timestamp))
            torch.save(model.state_dict(), path)

        # Always persist a full checkpoint after every epoch (model + optimizer + epoch + loss
        # history), overwriting the previous one. Unlike the best-vloss-only weights file above,
        # this has everything needed to resume training exactly if the process is interrupted
        # (crash, reboot, preemption) -- see find_resume_candidate()/resume_from above.
        checkpoint_path = os.path.join(model_path, f'checkpoint_{timestamp}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_vloss': best_vloss,
            'loss_history': loss_history,
            'vloss_history': vloss_history,
            'timestamp': timestamp,
        }, checkpoint_path)

        # print epoch nr
        print(f'epoch: {epoch}')

    # plot training loss
    plt.plot(loss_history, label='training loss')
    plt.plot(vloss_history, label='validation loss')
    plt.legend()
    fig = plt.gcf()
    fig.savefig(os.path.join(model_path, 'loss_history.pdf'))
    plt.show()

    """post training"""

    # load the best model
    model_cls = GNNmodel_QAT if model_type == 'GNN_QAT' else GNNmodel
    saved_model = model_cls(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels.to(device),
                            quantize=True).to(device)
    saved_model.load_state_dict(torch.load(os.path.join(model_path, 'model_{}'.format(timestamp)), weights_only=True))
    saved_model.eval()
    # Evaluation always uses the float32 model on the training device (GPU if available).
    # INT8 conversion is done after evaluation and saved separately for deployment.
    # Reason: quantize_dynamic is CPU-only, running 125 symbols × N_batches on CPU
    # would be much slower than the GPU float32 model, without any quality difference
    # (QAT ensures the float32 weights already reflect INT8-level precision).
    inference_device = device

    # define snr points
    snr_points = np.array([-30, -20, -10, 0.1, 10, 20, 30])

    # test set
    nr_batches = int(Nte / batch_size)
    Rsum_batches = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf_agc = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf_noquant = np.zeros((nr_batches, len(snr_points)))
    if sigma_theta_rad > 0:
        # additional curves under distributed RF-chain phase drift, for comparison against the
        # drift-free curves above (case B vs. case C in Phase_impact/phase_impact.ipynb)
        Rsum_batches_drift = np.zeros((nr_batches, len(snr_points)))
        Rsum_batches_zf_drift = np.zeros((nr_batches, len(snr_points)))
        Rsum_batches_zf_agc_drift = np.zeros((nr_batches, len(snr_points)))

    with torch.no_grad():
        running_vloss = 0
        for i, batch in enumerate(test_dataloader):
            print(f'batch of test set {i} / {nr_batches}')
            H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel
            bs = H.shape[0]
            x_init = torch.zeros((bs, M, 2)).to(inference_device)

            H, s = H.to(inference_device), s.to(inference_device)

            # forward pass
            outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
            for sidx in range(s.shape[-1]):
                outputs[:, :, sidx] = saved_model(H, s[:, :, sidx], x_init)  # NN takes 1 channel and 1 symbol as input

            normalized_output = normalize_outputs(outputs, Pt, norm_block_size)

            # compute sumrate
            Rsum_batches[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                  Pt=M, automatic_gain_control=False, precoding='non-linear',
                                                  x_nonlin=normalized_output.numpy(),
                                                  quant_params_path=quant_params_path,
                                                  s_provided=s.cpu().numpy(), normalize_across_symbols=True)
            # zf/mrt benchmark
            Rsum_batches_zf[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                     Pt=M, automatic_gain_control=False, precoding='zf-mrt',
                                                     quant_params_path=quant_params_path, s_provided=s.cpu().numpy(),
                                                     normalize_across_symbols=True)

            # zf/mrt benchmark
            Rsum_batches_zf_agc[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                         Pt=M, automatic_gain_control=True, precoding='zf-mrt',
                                                         quant_params_path=quant_params_path,
                                                         s_provided=s.cpu().numpy(), normalize_across_symbols=True)
            # zf/mrt no quant
            Rsum_batches_zf_noquant[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='none',
                                                         Pt=M, precoding='zf-mrt', s_provided=s.cpu().numpy(),
                                                             normalize_across_symbols=True)

            if sigma_theta_rad > 0:
                # same three cases, but with distributed RF-chain phase drift applied after the DAC
                Rsum_batches_drift[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                            Pt=M, automatic_gain_control=False, precoding='non-linear',
                                                            x_nonlin=normalized_output.numpy(),
                                                            quant_params_path=quant_params_path,
                                                            s_provided=s.cpu().numpy(), normalize_across_symbols=True,
                                                            sigma_theta=sigma_theta_rad)
                Rsum_batches_zf_drift[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                               Pt=M, automatic_gain_control=False, precoding='zf-mrt',
                                                               quant_params_path=quant_params_path,
                                                               s_provided=s.cpu().numpy(), normalize_across_symbols=True,
                                                               sigma_theta=sigma_theta_rad)
                Rsum_batches_zf_agc_drift[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                                   Pt=M, automatic_gain_control=True, precoding='zf-mrt',
                                                                   quant_params_path=quant_params_path,
                                                                   s_provided=s.cpu().numpy(), normalize_across_symbols=True,
                                                                   sigma_theta=sigma_theta_rad)

    # avg across the batches
    Rsum_avg = np.mean(Rsum_batches, axis=0)
    Rsum_avg_zf = np.mean(Rsum_batches_zf, axis=0)
    Rsum_avg_zf_agc = np.mean(Rsum_batches_zf_agc, axis=0)
    Rsum_avg_zf_no_quant = np.mean(Rsum_batches_zf_noquant, axis=0)

    plt.plot(snr_points, Rsum_avg, label='non lin prec')
    plt.plot(snr_points, Rsum_avg_zf, label='ZF/MRT')
    plt.plot(snr_points, Rsum_avg_zf_agc, label='ZF/MRT - AGC')
    plt.plot(snr_points, Rsum_avg_zf_no_quant, label='ZF/MRT - no quant')
    if sigma_theta_rad > 0:
        Rsum_avg_drift = np.mean(Rsum_batches_drift, axis=0)
        Rsum_avg_zf_drift = np.mean(Rsum_batches_zf_drift, axis=0)
        Rsum_avg_zf_agc_drift = np.mean(Rsum_batches_zf_agc_drift, axis=0)
        # note: avoid linestyle='--' here -- tikzplotlib 0.10.1 crashes on dashed Line2D objects
        # under matplotlib >= 3.6 (accesses the since-renamed private attribute _us_dashSeq)
        plt.plot(snr_points, Rsum_avg_drift, label=f'non lin prec + drift ({sigma_theta_deg:.0f}deg)')
        plt.plot(snr_points, Rsum_avg_zf_drift, label=f'ZF/MRT + drift ({sigma_theta_deg:.0f}deg)')
        plt.plot(snr_points, Rsum_avg_zf_agc_drift, label=f'ZF/MRT - AGC + drift ({sigma_theta_deg:.0f}deg)')
    plt.xlabel('SNR [dB]')
    plt.ylabel('R sum')
    plt.legend()
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(os.path.join(model_path, 'Rsum_testeset.tex'))
    fig.savefig(os.path.join(model_path, 'Rsum_testeset.pdf'))
    plt.show()

    logparams(os.path.join(model_path, 'sim_params.json'), sim_params)
    logparams(os.path.join(model_path, 'train_params.json'), train_params)

    # Convert GNN_QAT to INT8 and save after evaluation (not before), so evaluation
    # runs on GPU with the float32 model. quantize_dynamic is CPU-only.
    if model_type == 'GNN_QAT':
        int8_model = saved_model.cpu()
        int8_model = torch.ao.quantization.quantize_dynamic(int8_model, {nn.Linear}, dtype=torch.qint8)
        torch.save(int8_model, os.path.join(model_path, f'model_{timestamp}_int8.pt'))



if __name__ == '__main__':
    # Use repository-relative Linux/Unix-safe paths.
    varx = 0.5
    root_dir = REPO_ROOT
    quant_params_path = os.path.join(PROJECT_ROOT, 'non-uniform-quant-params', f'Gaussian_var_{varx}', 'numerical')

    # sim params
    M = 16
    K = 2
    Pt = M
    bits = 2
    quant = True #train with or without quantization
    sigma_theta_deg = 20.0  # std dev [deg] of distributed RF-chain phase drift (post-DAC); 0 disables it
                            # -- see Phase_impact/phase_impact.ipynb; most physically relevant for 'cellfree'
                            # (each AP has its own free-running LO), but the mechanism is enabled for any channel_model

    # train paramsw
    channel_model = 'cellfree' #'los' #'cellfree'
    nr_hidden_layers, nr_features = 4, 128
    model_type = 'GNN' #MLP, 'GNN', 'GNN_QAT'
    output_type = 'gumbel_softmax_hard' #'softmax_hard', 'softmax', 'gumbel_softmax_hard', 'gumbel_softmax'
    batch_size = 128 #128, 64
    lr = 0.5*10**-3
    nr_epochs = 25 #20 #10
    snr_tx = 20  # in db
    noise_var = Pt / (10 ** (snr_tx / 10))
    tau = 4 # for gumbel softmax
    stored_model_dir = f'stored_models_{channel_model}_generalized_bussgang_loss_mcdrift_randsigma' # todo set to desired folder!
    norm_block_size = 14  # symbols per normalization block; set to nr_symbols_per_channel for original behavior
    sigma_theta_warmup_epochs = nr_epochs // 2  # epochs to linearly ramp the training sigma_theta *ceiling*
                                                 # 0 -> target; avoids destabilizing optimization by exposing the
                                                 # untrained network to the full (possibly large) drift from epoch 0.
                                                 # Validation/eval always use the full target sigma_theta_deg.
    nr_drift_mc_samples = 4  # nr of independent phase-drift realizations averaged into the loss per batch,
                              # to reduce the gradient/validation-metric variance from the single-realization draw

    # data set params
    Ntr = 200000 #should be multiple of batchsize 200000
    Nval = 10000  #1000
    Nte = 10000  #10000
    nr_symbols_per_channel = 125 #todo big enough?

    # put all the params in a dictionary to store it
    sim_params = {
        'M': M,
        'K': K,
        'Pt': Pt,
        'bits': bits,
        'snr_tx': snr_tx,
        'noise_var': noise_var,
        'quant_params_path': quant_params_path,
        'quant': quant,
        'varx': varx,
        'root_dir': root_dir,
        'sigma_theta_deg': sigma_theta_deg,
    }

    training_params = {
        'channel_model': channel_model,
        'model_type': model_type,
        'output_type': output_type,
        'tau': tau,
        'Nr_train': Ntr,
        'Nr_val': Nval,
        'Nr_test': Nte,
        'nr_symbols_per_channel': nr_symbols_per_channel,
        'batch_size': batch_size,
        'epochs': nr_epochs,
        'lr': lr,
        'nr_hidden_layers': nr_hidden_layers,
        'nr_features': nr_features,
        'stored_model_dir': stored_model_dir,
        'norm_block_size': norm_block_size,
        'sigma_theta_warmup_epochs': sigma_theta_warmup_epochs,
        'nr_drift_mc_samples': nr_drift_mc_samples,
    }



    M = [40]
    K = [1, 2, 4]
    bits = [1, 2]
    output = ['softmax_hard', 'gumbel_softmax_hard', 'softmax_hard', 'softmax', 'gumbel_softmax'] #todo later
    tau_range = [1] #todo later (+annealing during training)
    sigma_theta_deg_range = [15.0, 20.0]  # sweep over different RF-chain phase drift levels
    for m in M:
        for tau in tau_range:
            for b in bits:
                for k in K:
                    for sigma_theta_deg in sigma_theta_deg_range:
                        sim_params['K'] = k
                        sim_params['M'] = m
                        sim_params['Pt'] = m
                        sim_params['bits'] = b
                        sim_params['sigma_theta_deg'] = sigma_theta_deg
                        snr_tx = 20  # in db
                        noise_var = sim_params['Pt'] / (10 ** (snr_tx / 10))
                        sim_params['noise_var'] = noise_var
                        training_params['output_type'] = 'gumbel_softmax_hard'
                        training_params['tau'] = tau

                        # auto-resume: skip combos that already finished, pick up combos that were
                        # interrupted (crash/reboot) from their last per-epoch checkpoint, so a plain
                        # re-run of this script after an interruption doesn't waste the sweep progress
                        combo_base_path = compute_base_path(
                            training_params['stored_model_dir'], m, k, training_params['batch_size'],
                            training_params['nr_hidden_layers'], training_params['nr_features'], tau,
                            sigma_theta_deg)
                        status, info = find_resume_candidate(
                            combo_base_path, b, training_params['output_type'], training_params['model_type'])
                        if status == 'done':
                            print(f'skipping (already completed): {info}')
                            continue
                        resume_from = info if status == 'resume' else None

                        print(f'---------------starting training for-------------------')
                        print(f'{sim_params=}')
                        print(f'{training_params=}')
                        print(f'{resume_from=}')
                        train(sim_params, training_params, resume_from=resume_from)
                        print(f'--------------------Done training---------------')

    """ todo:
    - speed up GNN
    
    
    - reduce lr on plateau?
    - try output types
    - try different tau's for gumbel
    - anneal tau during training
    
    - how many layers? 
    - how many features?
    
    - also learn the output levels?
    """



