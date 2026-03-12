
from utils.quantization import quantize_nonuniform
import tikzplotlib
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



def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def s_vs_shat_mrt_zf(s, w, h, bits, AGC):
    """
    :param s: transmit symbols K x Nr_symb_per_channel
    :param x: precoded symbols M x Nr_symb_per_channel (not yet quantized
    :param w: precoding matrix M x K
    :param h: channel M x K
    :param bits: nr bits to quantize with
    :return:
    """

    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))

    # precode
    x = w @ s # M X nrdata

    # quantize

    if AGC:
        # compute alpha to scale DAC input per antenna
        alpha_m = np.zeros(M)
        for m in range(M):
            wm = w[m, :]
            alpha_m[m] = wm.T @ wm.conj()

        # scale DAC input
        x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
        varx_scaled = np.var(x_scaled, axis=-1)

        # quantize
        xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

        # rescale the output of the DAC
        y = np.sqrt(alpha_m)[:, np.newaxis] * xq

    else:
        # quantize
        y = quantize_nonuniform(x, thresholds, outputlevels)
        #print(f'NO AGC')

    # compute received signal
    r = h.T @ y


    # compute Bussgang gain
    shat = np.zeros((K, s.shape[-1]), dtype=np.complex64)
    for k in range(K):
        # compute bussgang gain
        sk = s[k, :]
        rk = r[k, :]  # we 'look' at what we receive in the theta direction so same thing for both users
        Css = np.mean(sk * sk.conj())
        G = np.mean(rk * sk.conj()) / Css
        shat[k, :] = (1 / G) * rk  # sanity check

    #ideal case
    rideal = h.T @ x
    shat_ideal = np.zeros((K, s.shape[-1]), dtype=np.complex64)
    for k in range(K):
        # compute bussgang gain
        sk = s[k, :]
        rk = rideal[k, :]  # we 'look' at what we receive in the theta direction so same thing for both users
        Css = np.mean(sk * sk.conj())
        G = np.mean(rk * sk.conj()) / Css
        shat_ideal[k, :] = (1 / G) * rk  # sanity check

    s = s.flatten()
    shat = shat.flatten()
    shat_ideal = shat_ideal.flatten()
    error_re = np.abs(np.real(shat) - np.real(s))

    plt.scatter(np.real(s), np.real(shat), label=r'$\Re \{\hat{s}\}$')
    plt.scatter(np.imag(s), -1 * np.imag(shat), label=r'$-\Im \{\hat{s}\}$')
    plt.scatter(np.real(s), np.real(shat_ideal), label=r'$\Re \{s_{\mathrm{ideal}}\}$')
    plt.scatter(np.imag(s), -1*np.imag(shat_ideal), label=r'$-\Im \{s_{\mathrm{ideal}}\}$')
    plt.scatter(np.real(s), error_re, marker='x', label=r'$|s - \hat{s}|$')
    plt.title(f'MRT M={M} - K={K} - b={bits}')
    plt.legend()
    plt.ylim([-2.5, 2.5])
    plt.xlim([-2.5, 2.5])
    plt.xlabel('s')
    plt.ylabel('shat')
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(os.path.join(results_folder, f's_vs_shat_mrt.tex'))
    fig.savefig(os.path.join(results_folder, f's_vs_shat_mrt.pdf'))
    plt.show()

    return shat_ideal




def s_vs_shat_gnn(s, h, y, bits, shat_ideal, plot=True):
    """
    :param s: transmit symbols K x Nr_symb_per_channel
    :param h: channel M x K
    :param y: precoded quantized symbols M x Nr_symb_per_channel
    :param bits: nr bits used during quantized precoding
    :return:
    """

    # compute received signal
    r = h.T @ y

    # compute Bussgang gain
    shat = np.zeros((K, s.shape[-1]), dtype=np.complex64)
    for k in range(K):
        # compute bussgang gain
        sk = s[k, :]
        rk = r[k, :]  # we 'look' at what we receive in the theta direction so same thing for both users
        Css = np.mean(sk * sk.conj())
        G = np.mean(rk * sk.conj()) / Css
        shat[k, :] = (1 / G) * rk  # sanity check

    s = s.flatten()
    shat = shat.flatten()
    shat_ideal = shat_ideal.flatten()

    error_re = np.abs(np.real(shat) - np.real(s))
    p = np.real(s).argsort()
    s_sorted = s[p]
    error_re_sorted = error_re[p]

    #print(f'avg error: {np.mean(error_re_sorted)}')
    #print(f'var error: {np.var(error_re_sorted)}')

    mse = np.mean((shat - s) * (shat-s).conj())
    #print(f'mse: {mse}')
    if plot:
        plt.scatter(np.real(s), np.real(shat), label=r'$\Re \{\hat{s}\}$')
        plt.scatter(np.imag(s), -1 * np.imag(shat), label=r'$-\Im \{\hat{s}\}$')
        plt.scatter(np.real(s), np.real(shat_ideal), label=r'$\Re \{s_{\mathrm{ideal}}\}$')
        plt.scatter(np.imag(s), -1*np.imag(shat_ideal), label='$-\Im \{s_{\mathrm{ideal}}\}$')
        plt.plot(s_sorted, error_re_sorted, marker='x', label=r'$|\Re(s) - \Re(\hat{s})|$', color='red' )

        plt.title(f'GNN M={M} - K={K} b={bits}')
        plt.legend()
        plt.ylim([-2.5, 2.5])
        plt.xlim([-2.5, 2.5])
        plt.xlabel('s')
        plt.ylabel('shat')
        fig = plt.gcf()
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(os.path.join(results_folder, f's_vs_shat_gnn.tex'))
        fig.savefig(os.path.join(results_folder, f's_vs_shat_gnn.pdf'))
        plt.show()

        plt.scatter(np.real(shat), np.imag(shat))
        plt.xlabel('Re axis')
        plt.ylabel('Im axis')
        plt.title('shat plot GNN')
        plt.show()

    return mse

def plot_error(s, w, h, bits, AGC, plot=True):
    """
    :param s: transmit symbols K x Nr_symb_per_channel
    :param x: precoded symbols M x Nr_symb_per_channel (not yet quantized
    :param w: precoding matrix M x K
    :param h: channel M x K
    :param bits: nr bits to quantize with
    :return:
    """

    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))

    # precode
    x = w @ s  # M X nrdata

    # quantize

    if AGC:
        # compute alpha to scale DAC input per antenna
        alpha_m = np.zeros(M)
        for m in range(M):
            wm = w[m, :]
            alpha_m[m] = wm.T @ wm.conj()

        # scale DAC input
        x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
        varx_scaled = np.var(x_scaled, axis=-1)

        # quantize
        xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

        # rescale the output of the DAC
        y = np.sqrt(alpha_m)[:, np.newaxis] * xq

    else:
        # quantize
        y = quantize_nonuniform(x, thresholds, outputlevels)
        print(f'NO AGC')

    # compute received signal
    r = h.T @ y

    # compute Bussgang gain
    shat = np.zeros((K, s.shape[-1]), dtype=np.complex64)
    for k in range(K):
        # compute bussgang gain
        sk = s[k, :]
        rk = r[k, :]  # we 'look' at what we receive in the theta direction so same thing for both users
        Css = np.mean(sk * sk.conj())
        G = np.mean(rk * sk.conj()) / Css
        shat[k, :] = (1 / G) * rk  # sanity check

    # ideal case
    rideal = h.T @ x
    shat_ideal = np.zeros((K, s.shape[-1]), dtype=np.complex64)
    for k in range(K):
        # compute bussgang gain
        sk = s[k, :]
        rk = rideal[k, :]  # we 'look' at what we receive in the theta direction so same thing for both users
        Css = np.mean(sk * sk.conj())
        G = np.mean(rk * sk.conj()) / Css
        shat_ideal[k, :] = (1 / G) * rk  # sanity check

    s = s.flatten()
    shat = shat.flatten()
    shat_ideal = shat_ideal.flatten()
    error_re = np.abs(np.real(shat) - np.real(s))
    p = np.real(s).argsort()
    s_sorted = s[p]
    error_re_sorted = error_re[p]

    #print(f'avg error: {np.mean(error_re_sorted)}')
    #print(f'var error: {np.var(error_re_sorted)}')

    mse = np.mean((shat - s) * (shat-s).conj())
    #print(f'mse: {mse}')
    if plot:
        plt.scatter(np.real(s), np.real(shat), label=r'$\Re \{\hat{s}\}$')
        #plt.scatter(np.imag(s), -1 * np.imag(shat), label=r'$-\Im \{\hat{s}\}$')
        plt.scatter(np.real(s), np.real(shat_ideal), label=r'$\Re \{s_{\mathrm{ideal}}\}$')
        #plt.scatter(np.imag(s), -1 * np.imag(shat_ideal), label=r'$-\Im \{s_{\mathrm{ideal}}\}$')
        #plt.scatter(np.real(s), error_re, marker='x', label=r'$|s - \hat{s}|$')
        plt.plot(s_sorted, error_re_sorted, marker='x', label=r'$|\Re(s) - \Re(\hat{s})|$', color='red' )
        plt.title(f'MRT M={M} - K={K} - b={bits}')
        plt.legend()
        plt.ylim([-2.5, 2.5])
        plt.xlim([-2.5, 2.5])
        plt.xlabel('s')
        plt.ylabel('shat')
        fig = plt.gcf()
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(os.path.join(results_folder, f's_vs_shat_mrt.tex'))
        fig.savefig(os.path.join(results_folder, f's_vs_shat_mrt.pdf'))
        plt.show()

        plt.scatter(np.real(shat), np.imag(shat))
        plt.xlabel('Re axis')
        plt.ylabel('Im axis')
        plt.title('shat plot MRT')
        plt.show()

    return shat_ideal, mse

if __name__ == '__main__':
    M, K = 32, 5
    Pt = M
    bits = 3
    AGC = True


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

    # load model and dataset
    # load model and sim parameters
    #b=4
    model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\4_bits_GNN_gumbel_softmax_hard_2024-07-21_08-15-32'
    #b=1
    #model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-26_07-57-53'
    #b=2
    #model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\2_bits_GNN_gumbel_softmax_hard_2024-07-24_16-06-13'
    #b=3
    #model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\3_bits_GNN_gumbel_softmax_hard_2024-07-23_00-14-39'
    # b=4
    model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\4_bits_GNN_gumbel_softmax_hard_2024-07-21_08-15-32'

    #test M=8
    model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_8_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-14_19-00-57'

    #test M=2
    model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_iid\M_2_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-09-09_13-32-17'

    # M = 32 - K = 1 - b=1-4
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-26_07-57-53'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\2_bits_GNN_gumbel_softmax_hard_2024-07-24_16-06-13'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\3_bits_GNN_gumbel_softmax_hard_2024-07-23_00-14-39'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\4_bits_GNN_gumbel_softmax_hard_2024-07-21_08-15-32'



    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if AGC:
        results_folder = os.path.join(model_dir, f'AGC_s_vs_shat_{timestamp}')
    else:
        results_folder = os.path.join(model_dir, f'NO_AGC_s_vs_shat_{timestamp}')
    create_folder(results_folder)

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
    if train_params.get('channel_model') == None:
        channel_model = 'iid'
    else:
        channel_model = train_params['channel_model']

    # load output levels  of the DAC
    if bits == 1:
        output_levels = np.sqrt(Pt / (2 * M)) * torch.Tensor([-1, 1])  # only valid for 1 bit case
    else:
        output_levels = torch.from_numpy(np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))).type(
            torch.float32)
        print(f'{output_levels=}')

    # set GPU or CPU
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if GPU is available
    print(f'device: {device}')

    # load the best model
    saved_model = GNNmodel(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels.to(device), quantize=True,
                           output_type='softmax_hard').to(
        device)  # set output type to softmax hard so that discrete outputlevels are selected during inference!
    saved_model.load_state_dict(torch.load(os.path.join(model_dir, model)))

    # load the data
    datapath = os.path.join(root_dir, r'Quantization\precoding_quantization\non_lin_precoding\datasets',
                            f'{channel_model}')
    print(f'{datapath=}')
    Htrain, Hval, Htest, strain, sval, stest = getdata_nonlinprec(nr_symbols_per_channel, datapath, M, K, Ntr, Nval,
                                                                  Nte)
    # construct test set
    test_set = ChannelSymbolsDataset(Htest.astype(np.complex64), stest.astype(np.complex64),
                                     nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)


    # test set
    saved_model.eval()
    nr_batches = int(Nte / batch_size)
    mse_gnn_all = np.zeros(nr_batches * batch_size)
    mse_mrt_all = np.zeros(nr_batches * batch_size)
    with torch.no_grad():
        running_vloss = 0
        for i, batch in enumerate(test_dataloader):
            print(f'---------------------batch of test set {i} / {nr_batches}-------------------------------------------')
            H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel
            bs = H.shape[0]
            x_init = torch.zeros((bs, M, 2)).to(device)  # zeros as initial input for antennanode features

            # move input data to the GPU
            H, s = H.to(device), s.to(device)

            # forward pass (bs x M x nr_symbol_per_channel)
            outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
            for sidx in range(s.shape[-1]):
                outputs[:, :, sidx] = saved_model(H, s[:, :, sidx],
                                                  x_init)  # NN takes 1 channel and 1 symbol as input

            # normalization accross the symbol dimension (when multiple bits are considered)
            l2_norm = torch.linalg.vector_norm(outputs, ord=2, dim=1)  # bs x nr_symbols
            expt_x2 = torch.mean(l2_norm ** 2, dim=-1)  # bs
            epsilon = 1e-7  # to avoid NaN
            alpha = torch.sqrt(Pt / (expt_x2 + epsilon))
            alpha = alpha[:, None, None]  # add two dimensions for broadcasting
            normalized_output = alpha * outputs
            l2_norm_post_normalization = torch.linalg.vector_norm(normalized_output, ord=2, dim=1)
            expt_x2_post_normalization = torch.mean(l2_norm_post_normalization ** 2, dim=-1)

            # loop over channels in batch
            for h_idx in range(bs):
                # get one precoded vector for the channel h
                y = normalized_output[h_idx, :, :].detach().numpy()
                h = H[h_idx, :, :].cpu().detach().numpy()
                # userangle = 150
                # theta = userangle * np.pi / 180
                # h[:, 0] = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))
                s_select = s[h_idx, :, :].cpu().detach().numpy()

                #s_select = np.random.normal(0, np.sqrt(0.5), (1, K, 125)).astype(np.complex64)

                # todo at the moment only for 1 channel! extend
                # zf precoder
                if K == 1:
                    Wlin = h.conj()
                    norm = np.sqrt(Pt / np.linalg.norm(Wlin, ord='fro') ** 2)
                    Wlin *= norm
                else:
                    Wlin = h.conj() @ np.linalg.inv(h.T @ h.conj())  # M x K
                    norm = np.sqrt(Pt / np.linalg.norm(Wlin, ord='fro') ** 2)
                    Wlin *= norm


                s_hat_ideal = s_vs_shat_mrt_zf(s_select, Wlin, h, bits, AGC)
                shatidea, mse_mrt = plot_error(s_select, Wlin, h, bits, AGC, plot=True)
                mse_gnn = s_vs_shat_gnn(s_select, h, y, bits, shatidea, plot=True)

                #mse_gnn_all[i*batch_size+h_idx] = mse_gnn
                #mse_mrt_all[i*batch_size+h_idx] = mse_mrt

                #print(f'{mse_mrt=}  -  {mse_gnn=}')

                # in case we only want to look at 1 channel
                break


    # avg it out
    print(f'{mse_gnn_all=}')
    print(f'{mse_mrt_all=}')
    print(f'mean gnn: {np.mean(mse_gnn_all)}')
    print(f'mean mrt: {np.mean(mse_mrt_all)}')