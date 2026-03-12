import numpy as np
from utils.utils import getSymbols
from utils.quantization import quantize_nonuniform
import matplotlib.pyplot as plt
import os
import tikzplotlib


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def array_pattern(theta, M):
    h = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))  # *np.sin(theta) #todo expand to multiple users
    return h

def get_radiation_pattern_numerically(M, K, w, bits, angle):
    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))

    # do it numerically as test
    nrdata = 1000
    S = np.zeros((K, nrdata), dtype=complex)  # K x nrdata
    for k in range(K):
        S[k, :] = getSymbols(nrdata, p=1)

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
    ax.set_rmax(np.max(10 * np.log10(Ptheta)))
    ax.set_rmin(np.max(10 * np.log10(Ptheta)) - 30)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rticks(np.round(np.max(10 * np.log10(Ptheta))) + [-30, -25, -20, -15, -10, -5, 0])  # Less radial ticks
    ax.grid(True)
    plt.xlabel('Power [dB]')
    plt.ylabel('Angle [Rad]')
    plt.title(f'ZF M={M} K={K} - bits={bits} - angles={anglelist[0]}_{anglelist[1]} ')
    plt.legend()

    #tikzplotlib.save(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}.tex'))
    fig = plt.gcf()
    fig.savefig(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}_angles_{anglelist[0]}_{anglelist[1]}.pdf'))
    plt.show()

    # to get data in tikz format
    plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta), 'tab:red', label='intended signal')
    plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta_dist), 'tab:blue', label='distortion signal')
    plt.plot(theta_range * (180 / np.pi), 10 * np.log10(Ptheta/ Ptheta_dist), 'tab:green', label='SDR')
    plt.legend()
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    #tikzplotlib.clean_figure()
    tikzplotlib.save(os.path.join(results_folder, f'rad_plot_M_{M}_K_{K}_B_{bits}_angles_{anglelist[0]}_{anglelist[1]}.tex'))
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
    M, K = 32, 6
    Pt = M
    bits = 1
    results_folder = r'D:\thomas.feys\Quantization\precoding_quantization\figs_radiation_numerical'

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

    # generate LoS channel
    #anglelist = np.array([90, 150, 30, 115, 45, 0])
    #for angle in np.array([30, 180]):
    anglelist = np.array([155, 110, 55, 25, 135, 85])
    h = np.zeros((M, K), dtype=complex)
    for k in range(K):
        userangle = anglelist[k]#anglelist[k]#np.random.randint(0, 180)
        print(f'angle {k}: {userangle}')
        theta = userangle * np.pi / 180
        h[:, k] = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))

    # zf precoder
    Wzf = h.conj() @ np.linalg.inv(h.T @ h.conj())  # M x K
    norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
    Wzf *= norm

    pzf = get_radiation_pattern_numerically(M, K, Wzf, bits, anglelist)


