
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


def get_radiation_pattern_numerically_r_gnn(M, K, y, S, bits):
    """
    :param M: nr Tx antennas
    :param K: nr UEs
    :param y: precoded and quantized signal
    :param s: user symbol vector
    :return:
    """

    # compute received signal
    N_pt = 2000
    theta_range = np.linspace(0, np.pi, N_pt)
    Ptheta = np.zeros((N_pt, K))
    Ptheta_dist = np.zeros((N_pt, K))
    for i, theta in enumerate(theta_range):
        # get received sig without noise
        H = array_pattern(theta, M)[:, np.newaxis]
        r = H.T @ y

        # compute Bussgang gain
        Shat = np.zeros((K, S.shape[-1]))
        for k in range(K):
            # compute bussgang gain
            sk = S[k, :]
            rk = r  # we 'look' at what we receive in the theta direction so same thing for both users
            Css = np.mean(sk * sk.conj())
            G = np.mean(rk * sk.conj()) / Css
            Shat[k, :] = (1 / G) * rk  # sanity check

            # compute usefull sig power and dist power for user k
            Ptheta[i, k] = np.abs(G) ** 2 * Css
            Ptheta_dist[i, k] = np.mean(rk * rk.conj()) - np.abs(G) ** 2 * Css

    for k in range(K):
        # plot radiation pattern
        # plot radiation pattern of the intended signal
        fig = plt.figure()
        ax = plt.subplot(projection='polar')
        ax.plot(theta_range, 10 * np.log10(Ptheta[:, k]), 'tab:red', label='Intended signal')
        ax.plot(theta_range, 10 * np.log10(Ptheta_dist[:, k]), 'tab:blue', label='Distortion')
        ax.plot(theta_range, 10 * np.log10(Ptheta[:, k] / Ptheta_dist[:, k]), 'tab:orange', label='SDR')
        max_r = max(np.max(10 * np.log10(Ptheta[:, k])), np.max(10 * np.log10(Ptheta[:, k] / Ptheta_dist[:, k])))
        max_r = max(max_r, np.max(10 * np.log10(Ptheta_dist[:, k]))) + 1
        ax.set_rmax(max_r)
        ax.set_rmin(max_r - 30)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rticks(
            np.round(np.max(10 * np.log10(Ptheta[:, k]))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
        ax.grid(True)
        plt.xlabel('Power [dB]')
        plt.ylabel('Angle [Rad]')
        plt.title(f'GNN M={M} K={K} - bits={bits} - k={k}')
        plt.legend()

        # tikzplotlib.save(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}.tex'))
        fig = plt.gcf()
        fig.savefig(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}_GNN.pdf'))
        plt.show()

        #to get data in tikz format
        plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta[:, k]), 'tab:red', label='intended signal')
        plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta_dist[:, k]), 'tab:blue', label='distortion signal')
        plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta[:, k]/Ptheta_dist[:, k]), 'tab:green', label='SDR')
        plt.legend()
        fig = plt.gcf()
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.clean_figure()
        tikzplotlib.save(os.path.join(results_folder, 'intended_and_distortion_signal_GNN.tex'))
        plt.show()

    if K != 1:
        print(f'todo adjust to user sizeeeeeeee!')
        # plot radiation pattern sum for all users
        # plot radiation pattern of the intended signal
        fig = plt.figure()
        ax = plt.subplot(projection='polar')
        ax.plot(theta_range, 10 * np.log10(Ptheta[:, 0] + Ptheta[:, 1]), 'tab:red', label='Intended signal')
        ax.plot(theta_range, 10 * np.log10(Ptheta_dist[:, 0] + Ptheta_dist[:, 1]), 'tab:blue', label='Distortion')
        ax.plot(theta_range, 10 * np.log10((Ptheta[:, 0] + Ptheta[:, 1]) / (Ptheta_dist[:, 0] + Ptheta_dist[:, 1])),
                'tab:orange', label='SDR')
        max_r = max(np.max(10 * np.log10(Ptheta[:, 0] + Ptheta[:, 1])),
                    np.max(10 * np.log10((Ptheta[:, 0] + Ptheta[:, 1]) / (Ptheta_dist[:, 0] + Ptheta_dist[:, 1])))) + 2
        ax.set_rmax(max_r)
        ax.set_rmin(max_r - 30)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rticks(
            np.round(np.max(10 * np.log10(Ptheta[:, k]))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
        ax.grid(True)
        plt.xlabel('Power [dB]')
        plt.ylabel('Angle [Rad]')
        plt.title(f'ZF M={M} K={K} - bits={bits} - k=sum')
        plt.legend()

        tikzplotlib.save(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}_GNN.tex'))
        fig = plt.gcf()
        fig.savefig(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}_GNN.pdf'))
        plt.show()

    # # plot radiation pattern of the distortion
    # fig = plt.figure()
    # ax = plt.subplot(projection='polar')
    # ax.plot(theta_range, 10 * np.log10(Ptheta_dist), 'tab:blue', label='distortion')
    # ax.set_rmax(np.max(10 * np.log10(Ptheta_dist)))
    # ax.set_rmin(np.max(10 * np.log10(Ptheta_dist)) - 30)
    # ax.set_thetamin(0)
    # ax.set_thetamax(180)
    # ax.set_rticks(np.round(np.max(10 * np.log10(Ptheta))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
    # ax.grid(True)
    # plt.xlabel('Power [dB]')
    # plt.title(f'zf distortion signal')
    # plt.legend()
    # # tikzplotlib.clean_figure()
    # # tikzplotlib.save(os.path.join(path, f'{title}', 'distortion_signal.tex'))
    # # fig = plt.gcf()
    # # fig.savefig(os.path.join(path, f'{title}', 'distortion_signal.pdf'))
    # plt.show()

    return Ptheta  # , Ptheta_dist


def get_radiation_pattern_numerically_r_zf_test(M, K, y, S, w, bits):
    """
    :param M: nr Tx antennas
    :param K: nr UEs
    :param y: precoded and quantized signal
    :param s: user symbol vector
    :return:
    """
    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))

    # # do it numerically as test
    # nrdata = 1000
    # S = np.zeros((K, nrdata), dtype=complex)  # K x nrdata
    # for k in range(K):
    #     S[k, :] = getSymbols(nrdata, p=1)

    # precode
    x = w @ S  # M X nrdata

    # quantize
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

    # todo compute received signal
    N_pt = 2000
    theta_range = np.linspace(0, np.pi, N_pt)
    Ptheta = np.zeros((N_pt, K))
    Ptheta_dist = np.zeros((N_pt, K))
    for i, theta in enumerate(theta_range):
        # get received sig without noise
        H = array_pattern(theta, M)[:, np.newaxis]
        r = H.T @ y

        # compute Bussgang gain
        Shat = np.zeros((K, S.shape[-1]))
        for k in range(K):
            # compute bussgang gain
            sk = S[k, :]
            rk = r # we 'look' at what we receive in the theta direction so same thing for both users
            Css = np.mean(sk * sk.conj())
            G = np.mean(rk * sk.conj()) / Css
            Shat[k, :] = (1 / G) * rk  # sanity check

            # compute usefull sig power and dist power for user k
            Ptheta[i, k] = np.abs(G) ** 2 * Css
            Ptheta_dist[i, k] = np.mean(rk * rk.conj()) - np.abs(G) ** 2 * Css

    for k in range(K):
        # plot radiation pattern
        # plot radiation pattern of the intended signal
        fig = plt.figure()
        ax = plt.subplot(projection='polar')
        ax.plot(theta_range, 10 * np.log10(Ptheta[:, k]), 'tab:red', label='Intended signal')
        ax.plot(theta_range, 10 * np.log10(Ptheta_dist[:, k]), 'tab:blue', label='Distortion')
        ax.plot(theta_range, 10 * np.log10(Ptheta[:, k]/Ptheta_dist[:, k]), 'tab:orange', label='SDR')
        max_r = max(np.max(10 * np.log10(Ptheta[:, k])), np.max(10 * np.log10(Ptheta[:, k]/Ptheta_dist[:, k]))) + 2
        ax.set_rmax(max_r)
        ax.set_rmin(max_r - 30)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rticks(np.round(np.max(10 * np.log10(Ptheta[:, k]))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
        ax.grid(True)
        plt.xlabel('Power [dB]')
        plt.ylabel('Angle [Rad]')
        plt.title(f'ZF M={M} K={K} - bits={bits} - k={k}')
        plt.legend()

        # tikzplotlib.save(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}.tex'))
        fig = plt.gcf()
        fig.savefig(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}_zf.pdf'))
        plt.show()

        #to get data in tikz format
        plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta[:, k]), 'tab:red', label='intended signal')
        plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta_dist[:, k]), 'tab:blue', label='distortion signal')
        plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta[:, k]/Ptheta_dist[:, k]), 'tab:green', label='SDR')
        plt.legend()
        fig = plt.gcf()
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.clean_figure()
        tikzplotlib.save(os.path.join(results_folder, 'intended_and_distortion_signal_ZF.tex'))
        plt.show()

    if K != 1:
        print(f'todo adjust to user sizeeeeeeee!')
        # plot radiation pattern sum for all users
        # plot radiation pattern of the intended signal
        fig = plt.figure()
        ax = plt.subplot(projection='polar')
        ax.plot(theta_range, 10 * np.log10(Ptheta[:, 0] + Ptheta[:, 1]), 'tab:red', label='Intended signal')
        ax.plot(theta_range, 10 * np.log10(Ptheta_dist[:, 0] + Ptheta_dist[:, 1]), 'tab:blue', label='Distortion')
        ax.plot(theta_range, 10 * np.log10((Ptheta[:, 0] + Ptheta[:, 1]) / (Ptheta_dist[:, 0] + Ptheta_dist[:, 1])), 'tab:orange', label='SDR')
        max_r = max(np.max(10 * np.log10(Ptheta[:, 0] + Ptheta[:, 1])), np.max(10 * np.log10((Ptheta[:, 0] + Ptheta[:, 1]) / (Ptheta_dist[:, 0] + Ptheta_dist[:, 1])))) + 2
        ax.set_rmax(max_r)
        ax.set_rmin(max_r - 30)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rticks(np.round(np.max(10 * np.log10(Ptheta[:, k]))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
        ax.grid(True)
        plt.xlabel('Power [dB]')
        plt.ylabel('Angle [Rad]')
        plt.title(f'ZF M={M} K={K} - bits={bits} - k=sum')
        plt.legend()

        # tikzplotlib.save(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}.tex'))
        fig = plt.gcf()
        fig.savefig(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}_test.pdf'))
        plt.show()

    # # plot radiation pattern of the distortion
    # fig = plt.figure()
    # ax = plt.subplot(projection='polar')
    # ax.plot(theta_range, 10 * np.log10(Ptheta_dist), 'tab:blue', label='distortion')
    # ax.set_rmax(np.max(10 * np.log10(Ptheta_dist)))
    # ax.set_rmin(np.max(10 * np.log10(Ptheta_dist)) - 30)
    # ax.set_thetamin(0)
    # ax.set_thetamax(180)
    # ax.set_rticks(np.round(np.max(10 * np.log10(Ptheta))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
    # ax.grid(True)
    # plt.xlabel('Power [dB]')
    # plt.title(f'zf distortion signal')
    # plt.legend()
    # # tikzplotlib.clean_figure()
    # # tikzplotlib.save(os.path.join(path, f'{title}', 'distortion_signal.tex'))
    # # fig = plt.gcf()
    # # fig.savefig(os.path.join(path, f'{title}', 'distortion_signal.pdf'))
    # plt.show()

    return Ptheta#, Ptheta_dist


def array_pattern(theta, M):
    h = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))  # *np.sin(theta) #todo expand to multiple users
    return h

def get_radiation_pattern_numerically(M, K, w, bits, s_select):
    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))

    # do it numerically as test
    nrdata = 1000
    # S = np.zeros((K, nrdata), dtype=complex)  # K x nrdata
    # for k in range(K):
    #     S[k, :] = getSymbols(nrdata, p=1)

    S = s_select

    # precode
    x = w @ S  # M X nrdata

    # quantize
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

    # compute bussganggain after each DAC (x = M x Nrdata)
    alpha_m = np.diag(np.mean(y * x.conj(), axis=-1) / np.mean(np.abs(x) ** 2, axis=-1))

    # compute distortion
    q = y - alpha_m @ x

    # compute linear part after dac
    alpha_m_xm = alpha_m @ x


    # compute radiation pattern in all directions of theta
    N_pt = 2000
    theta_range = np.linspace(0, np.pi, N_pt)
    Ptheta = np.zeros(N_pt)
    Ptheta_dist = np.zeros(N_pt)
    for i, theta in enumerate(theta_range):
        # get received sig without noise
        H = array_pattern(theta, M)
        r = H.T @ alpha_m_xm
        rdist = H.T @ q

        # get average received signal power
        Ptheta[i] = np.mean(np.abs(r) ** 2, axis=-1)
        Ptheta_dist[i] = np.mean(np.abs(rdist) ** 2, axis=-1)

    # plot radiation pattern
    # plot radiation pattern of the intended signal
    fig = plt.figure()
    ax = plt.subplot(projection='polar')
    ax.plot(theta_range, 10 * np.log10(Ptheta), 'tab:red', label='Intended signal')
    ax.plot(theta_range, 10 * np.log10(Ptheta_dist), 'tab:blue', label='Distortion')
    ax.plot(theta_range, 10 * np.log10(Ptheta/Ptheta_dist), 'tab:orange', label='SDR')
    max_r = max(np.max(10 * np.log10(Ptheta)), np.max(10 * np.log10(Ptheta/Ptheta_dist))) + 2
    ax.set_rmax(max_r)
    ax.set_rmin(max_r - 30)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rticks(np.round(np.max(10 * np.log10(Ptheta))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
    ax.grid(True)
    plt.xlabel('Power [dB]')
    plt.ylabel('Angle [Rad]')
    plt.title(f'ZF M={M} K={K} - bits={bits}')
    plt.legend()

    #tikzplotlib.save(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}.tex'))
    fig = plt.gcf()
    fig.savefig(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}.pdf'))
    plt.show()

    # # plot radiation pattern of the distortion
    # fig = plt.figure()
    # ax = plt.subplot(projection='polar')
    # ax.plot(theta_range, 10 * np.log10(Ptheta_dist), 'tab:blue', label='distortion')
    # ax.set_rmax(np.max(10 * np.log10(Ptheta_dist)))
    # ax.set_rmin(np.max(10 * np.log10(Ptheta_dist)) - 30)
    # ax.set_thetamin(0)
    # ax.set_thetamax(180)
    # ax.set_rticks(np.round(np.max(10 * np.log10(Ptheta))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
    # ax.grid(True)
    # plt.xlabel('Power [dB]')
    # plt.title(f'zf distortion signal')
    # plt.legend()
    # # tikzplotlib.clean_figure()
    # # tikzplotlib.save(os.path.join(path, f'{title}', 'distortion_signal.tex'))
    # # fig = plt.gcf()
    # # fig.savefig(os.path.join(path, f'{title}', 'distortion_signal.pdf'))
    # plt.show()

    return Ptheta#, Ptheta_dist


if __name__ == '__main__':
    M, K = 32, 5
    Pt = M
    bits = 3


    # for local pc or server
    local = True
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

    # fig 9, M=16
    #model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los_diff_M\M_16_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-09-05_16-00-39'

    # fig 9, M=32
    #model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los_diff_M\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-08-31_05-30-39'

    #M=16 los: r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los_diff_M\M_16_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-08-30_22-39-36'

    # fig 9, M=8
    model_dir = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los_diff_M\M_8_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-08-30_15-50-11'

    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-08-20_15-17-46'

    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los\M_32_K_1_bs_128_layers_4_dl_128_tau_1\4_bits_GNN_gumbel_softmax_hard_2024-08-21_05-39-56'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los\M_32_K_1_bs_128_layers_4_dl_128_tau_1\2_bits_GNN_gumbel_softmax_hard_2024-08-20_22-28-06'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-26_07-57-53'

    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_los\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-08-20_15-17-46'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_2_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-25_23-46-03'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-26_07-57-53'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\4_bits_GNN_gumbel_softmax_hard_2024-07-21_08-15-32'
    # r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\3_bits_GNN_gumbel_softmax_hard_2024-07-23_00-14-39'
    # r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\2_bits_GNN_gumbel_softmax_hard_2024-07-24_16-06-13'
    # r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_1_bs_128_layers_4_dl_128_tau_1\1_bits_GNN_gumbel_softmax_hard_2024-07-26_07-57-53'
    # r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\stored_models_v2\M_32_K_2_bs_128_layers_4_dl_128_tau_1\3_bits_GNN_gumbel_softmax_hard_2024-07-22_16-03-00'

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_folder = os.path.join(model_dir, f'V2_rad_plots_{timestamp}')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if GPU is available
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
    with torch.no_grad():
        running_vloss = 0
        for i, batch in enumerate(test_dataloader):
            print(f'batch of test set {i} / {nr_batches}')
            H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel
            bs = H.shape[0]
            x_init = torch.zeros((bs, M, 2)).to(device)  # zeros as initial input for antennanode features

            # move input data to the GPU
            H, s = H.to(device), s.to(device)
            userangle = 110 #fix userangle of first channel to get consistent plots
            theta = userangle * np.pi / 180
            h_fixed = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))
            H[0, :, :] = torch.from_numpy(h_fixed[:, np.newaxis])


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

            # get one precoded vector for the channel h
            y = normalized_output[0, :, :].detach().numpy()
            h = H[0, :, :].cpu().detach().numpy()
            # userangle = 150
            # theta = userangle * np.pi / 180
            # h[:, 0] = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))
            s_select = s[0, :, :].cpu().detach().numpy()

            # zf precoder
            Wzf = h.conj() @ np.linalg.inv(h.T @ h.conj())  # M x K
            norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
            Wzf *= norm
            #pzf = get_radiation_pattern_numerically(M, K, Wzf, bits, s_select)
            pzf_test = get_radiation_pattern_numerically_r_zf_test(M, K, y, s_select, Wzf, bits)
            pgnn = get_radiation_pattern_numerically_r_gnn(M, K, y, s_select, bits)

            break

    # # generate LoS channel
    # anglelist = np.array([90, 150, 30, 115, 45, 0])
    # h = np.zeros((M, K), dtype=complex)
    # for k in range(K):
    #     userangle = anglelist[k]#np.random.randint(0, 180)
    #     print(f'angle {k}: {userangle}')
    #     theta = userangle * np.pi / 180
    #     h[:, k] = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))




    # # zf precoder
    # Wzf = h.conj() @ np.linalg.inv(h.T @ h.conj())  # M x K
    # norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
    # Wzf *= norm
    #
    # pzf = get_radiation_pattern_numerically(M, K, Wzf, bits)


