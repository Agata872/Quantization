import torch
import numpy as np
from utils.utils import rayleigh_channel_MU, getSymbols, create_folder, logparams, load_params
import os
from tqdm import tqdm
from model import MLPmodel, SumRateLoss, MLPmodel_noquant, GNNmodel
import matplotlib.pyplot as plt
from torchsummary import summary
from datetime import datetime
from MIMO_sims.Rsum_all import Rsum_Bussgang_Rx
from data_handling import getdata_nonlinprec, ChannelSymbolsDataset
import re

if __name__ == "__main__":
    # for local pc or server
    local = False
    varx = 0.5
    if local:
        root_dir = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub'
        quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
        quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')
    else:
        root_dir = r'D:\thomas.feys'
        quant_params_path = r'D:\thomas.feys\Quantization\precoding_quantization\non-uniform-quant-params'
        quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # load model and sim parameters
    model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\4_bits_GNN_gumbel_softmax_hard_2024-07-21_08-15-32'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\3_bits_GNN_gumbel_softmax_hard_2024-07-23_00-14-39'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\2_bits_GNN_gumbel_softmax_hard_2024-07-24_16-06-13'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-26_07-57-53'
    # r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_2_bs_128_layers_4_dl_128_tau_1\3_bits_GNN_gumbel_softmax_hard_2024-07-22_16-03-00'

    # extract timestamp at the end of model dir
    pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'  # Define the regular expression pattern to match the timestamp
    match = re.search(pattern, model_dir)  # Search for the pattern in the string
    if match:
        timestamp = match.group()
        print(f'Timestamp: {timestamp}')
    else:
        print('No timestamp found in the provided path.')
    model = f'model_{timestamp}'
    output_dir = os.path.join(model_dir, 'post_training_eval')
    create_folder(output_dir)
    train_params = load_params(os.path.join(model_dir, 'train_params.json'))
    sim_params = load_params(os.path.join(model_dir, 'sim_params.json'))
    print(f'{train_params=}')
    print(f'{sim_params=}')

    # unpack params
    M = sim_params['M']
    K = sim_params['K']
    Pt = sim_params['Pt']
    bits = sim_params['bits']
    nr_features = train_params['nr_features']
    nr_hidden_layers = train_params['nr_hidden_layers']
    tau = train_params['tau']
    nr_symbols_per_channel = train_params['nr_symbols_per_channel']
    Ntr = train_params['Nr_train']
    Nval = train_params['Nr_val']
    Nte = train_params['Nr_test']
    batch_size = train_params['batch_size']

    # load output levels  of the DAC
    if bits == 1:
        output_levels = np.sqrt(Pt / (2 * M)) * torch.Tensor([-1, 1])  # only valid for 1 bit case
    else:
        output_levels = torch.from_numpy(np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))).type(
            torch.float32)
        print(f'{output_levels=}')

    # set GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if GPU is available
    print(f'device: {device}')

    # load the best model
    saved_model = GNNmodel(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels.to(device), quantize=True,
                           output_type='softmax_hard').to(
        device)  # set output type to softmax hard so that discrete outputlevels are selected during inference!
    saved_model.load_state_dict(torch.load(os.path.join(model_dir, model)))

    # load the data
    datapath = os.path.join(root_dir, r'\Quantization\precoding_quantization\non_lin_precoding\datasets')
    Htrain, Hval, Htest, strain, sval, stest = getdata_nonlinprec(nr_symbols_per_channel, datapath, M, K, Ntr, Nval,
                                                                  Nte)
    # construct test set
    test_set = ChannelSymbolsDataset(Htest.astype(np.complex64), stest.astype(np.complex64),
                                     nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # define snr points
    snr_points = np.array([-30, -20, -10, 0.1, 10, 20, 30])

    # test set
    saved_model.eval()
    nr_batches = int(Nte / batch_size)
    Rsum_batches = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf_agc = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf_noquant = np.zeros((nr_batches, len(snr_points)))

    with torch.no_grad():
        running_vloss = 0
        for i, batch in enumerate(test_dataloader):
            print(f'batch of test set {i} / {nr_batches}')
            H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel
            bs = H.shape[0]
            x_init = torch.zeros((bs, M, 2)).to(device)  # zeros as initial input for antennanode features

            # move input data to the GPU
            H, s = H.to(device), s.to(device)

            # forward pass (bs x M x nr_symbol_per_channel)
            outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
            for sidx in range(s.shape[-1]):
                outputs[:, :, sidx] = saved_model(H, s[:, :, sidx], x_init)  # NN takes 1 channel and 1 symbol as input

            # normalization accross the symbol dimension (when multiple bits are considered)
            l2_norm = torch.linalg.vector_norm(outputs, ord=2, dim=1)  # bs x nr_symbols
            expt_x2 = torch.mean(l2_norm ** 2, dim=-1)  # bs
            epsilon = 1e-7  # to avoid NaN
            alpha = torch.sqrt(Pt / (expt_x2 + epsilon))
            alpha = alpha[:, None, None]  # add two dimensions for broadcasting
            normalized_output = alpha * outputs
            l2_norm_post_normalization = torch.linalg.vector_norm(normalized_output, ord=2, dim=1)
            expt_x2_post_normalization = torch.mean(l2_norm_post_normalization ** 2, dim=-1)

            # todo amp phase plot for one precoded vector
            y = normalized_output[0, :, 0].detach().numpy()
            h = H[0, :, 0].cpu().detach().numpy()
            plt.stem(np.arange(M), np.abs(y), label=r'$|y_m|$')
            plt.stem(np.arange(M), np.abs(h), 'g', markerfmt='gx', label=r'$|h_m|$')
            plt.legend()
            plt.xlabel('Antenna index, m')
            plt.ylabel(r'$|y_m|$')
            plt.show()

            plt.stem(np.arange(M), np.angle(y))
            plt.xlabel('Antenna index, m')
            plt.ylabel(r'$\angle y_m$')
            plt.show()

            plt.stem(np.arange(M), np.angle(y*h))
            plt.xlabel('Antenna index, m')
            plt.ylabel(r'$\angle (y_m h_m)$')
            plt.show()
            break





