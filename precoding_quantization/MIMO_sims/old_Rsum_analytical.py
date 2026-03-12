import numpy as np
import matplotlib.pyplot as plt
from utils.utils import rayleigh_channel_MU, los_channel_MU
from utils.precoding import ZF_precoding, MRT_precoding
import os

def Rsum_loopy(H, W, diag_alpha, diag_beta, noisevar):
    """
    :param H: M x K channel matrix
    :param W: M x K precoding matrix
    :return: sum rate
    """

    K = H.shape[-1]
    M = H.shape[0]
    Rsum = 0
    for k in range(K):
        hk = H[:, k]
        wk = W[:, k]

        # usefull sig
        alpha = diag_alpha[0, 0]
        sig = alpha**2 * np.abs(hk.T @ wk)**2
        check = np.abs(hk.T @ diag_alpha @ wk)**2

        # inteference
        interference = 0
        for kprime in range(K):
            hkprime = H[:, kprime]
            wkprime = W[:, kprime]
            if kprime != k:
                interference += np.abs(hk.T @ diag_alpha @ wkprime)**2

        # dist
        """ diag(WW^H) or just WW^H"""
        Rqq = diag_alpha @ diag_beta @ np.diag(np.diag(W @ W.conj().T))
        beta = diag_beta[0, 0]
        wwh = W @ W.conj().T

        # not taking diagWW^H
        abswwh = np.abs(wwh)
        Rqq_check = beta * alpha * W @ W.conj().T

        dist = hk.T @ Rqq_check @ hk.conj()

        # sinqdr
        sinqdrk = np.real(sig / (interference + dist + noisevar))

        Rk = np.log2(1 + sinqdrk)

        Rsum += Rk

    return Rsum

def Rsum_analytical(H, W, diag_alpha, diag_beta, noisevar, Pt=64):
    """
    :param H: M x K channel matrix
    :param W: M x K precoding matrix
    :return: sum rate
    """

    # compute usefull sig
    HdiagalphaW = H.T @ diag_alpha @ W
    usefull_sig = np.diag(HdiagalphaW)
    usefull_sig_pwr = np.abs(usefull_sig)**2

    # inter user interference
    interference = np.sum(np.abs(HdiagalphaW - np.diag(usefull_sig))**2, axis=-1)

    # distortion
    inttest = np.diag(np.diag(W @ W.conj().T))
    Rqq = diag_alpha @ diag_beta @ np.diag(np.diag(W @ W.conj().T))
    Rqq = diag_alpha @ diag_beta @ W @ W.conj().T
    dist = np.diag(H.T @ Rqq @ H.conj())

    # SINQDR
    noisevar = noisevar[:, np.newaxis] #extend axis so that devision is done for all noisevars
    #noisevar = noisevar[-1] # test only do it for last noisevar
    #noisevar = 0 # todo sanity check
    sinqdr = usefull_sig_pwr / (interference + dist + noisevar) #nr_noisevars x K

    # rate per user
    Rk = np.real(np.log2(1 + sinqdr)) #nr_noisevars x K

    # sum rate
    Rsum = np.sum(Rk, axis=-1)

    # sanity check using for loops
    # Rsum_check = Rsum_loopy(H, W, diag_alpha, diag_beta, noisevar)
    # print(f'{Rsum=}')
    # print(f'{Rsum_check=}')
    # print(f'diff= {Rsum-np.squeeze(Rsum_check)}')

    return Rsum


def get_sumrate(b, H, Pt, noise_vars, path=''):

    if b == 'inf':
        beta = 0
    else:
        # read params of quantizer
        beta = np.load(os.path.join(path, f'{b}bits_nmse.npy')) # todo check if this gives MSE or NMSE


    # construct diag_alpha and diag_beta (with loaded values from file)
    # todo maybe different pwr norm needed as in notes, but the differences are very big so not sure?
    diag_beta = np.identity(M) * beta
    diag_alpha = np.identity(M) - diag_beta

    # get sum rate
    Rsum_all = np.zeros((channel_realizations, nr_snr_points), dtype=float)
    for i in range(H.shape[0]):
        # compute precoding matrix (ZF)
        if H.shape[-1] == 1:
            W = MRT_precoding(H[i, :, :], Pt)
        else:
            W = ZF_precoding(H[i, :, :], Pt)

        #W = MRT_precoding(H[i, :, :], Pt) # sanity check

        #todo turn this on for final checks
        # alpha = np.sqrt(Pt / np.trace(diag_alpha @ W @ W.conj().T))
        # W *= alpha
        # this has minimal impact as this 'amplifies' both the linear signal and the distortion
        # it only affects the performance at low SNR, as then you are noise limited and not distortion limited
        # for high snr distortion is the limiting factor so increasing Pt doesnt help



        pwr_test = np.linalg.norm(W, ord='fro')**2

        # print(f'pwr before Q: {pwr_test}')
        # # check
        # print(f'pwr after Q: {np.trace(diag_alpha @ W @ W.conj().T)}')

        # compute sum rate
        #Rsum_all[i, :] = Rsum_analytical(H[i, :, :], W, diag_alpha, diag_beta, noise_vars, Pt)
        #test2 = Rsum_analytical(H[i, :, :], W, diag_alpha, diag_beta, noise_vars, Pt)
        #check = Rsum_loopy(H[i, :, :], W, diag_alpha, diag_beta, noise_vars)

        Rsum_all[i, :] = Rsum_loopy(H[i, :, :], W, diag_alpha, diag_beta, noise_vars)
        print('check')

    # take avg over channel realizations
    Rsum_avg = np.mean(Rsum_all, axis=0)



    return Rsum_avg





if __name__ == '__main__':
    bits = np.arange(1, 2, 1) #nr bits
    M, K = 64, 16#nr antennas and users
    Pt = M

    # quantizer params path
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # snr range
    nr_snr_points = 20
    snr_tx = np.linspace(-30, 35, nr_snr_points)
    noise_vars = Pt / (10 ** (snr_tx / 10))


    # generate channels
    channel_realizations = 100
    H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
    np.random.seed(0)
    for i in range(channel_realizations):
        H[i, :, :] = rayleigh_channel_MU(M, K)
        #H[i, :, :] = np.ones((M, K)) / K #sanity check
        #H[i, :, :] = los_channel_MU(M, K)


    # compute sumrate for diff nrs of bits
    for b in bits:
        print(f'-------------------------------------{b} bits -----------------------------------')
        Rsum = get_sumrate(b, H, Pt, noise_vars, path=quant_params_path)
        print(f'sindr: {10*np.log10(2**Rsum-1)}')
        plt.plot(snr_tx, Rsum, label=f'{b} bits')

    Rsum = get_sumrate('inf', H, Pt, noise_vars, path=quant_params_path)
    plt.grid()
    plt.plot(snr_tx, Rsum, label=f'inf res')
    plt.title(f'param N(0,{varx}) - analytical M: {M} K:{K} Pt: {Pt}')
    plt.xlabel(r'$P_T / \sigma_{\nu}^2$ [dB]')
    plt.ylabel(r'$R_{\mathrm{sum}}$')
    plt.legend()
    plt.show()
