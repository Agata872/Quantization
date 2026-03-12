import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from utils.utils import rayleigh_channel_MU, symbols_MU, los_channel_MU
from utils.quantization import quantize_uniform, quantize_nonuniform
from utils.precoding import ZF_precoding, MRT_precoding
import os
import scipy.stats

def sumrate(S, r, noise_vars, x=None):
    # MMSE of x given r
    Shat = np.zeros_like(r)
    K = r.shape[0]
    sindr = np.zeros((K, len(noise_vars)))
    for k in range(K):
        # compute bussgang gain
        sk = S[k, :]
        rk = r[k, :]
        Css = np.mean(sk * sk.conj())
        G = np.mean(rk * sk.conj()) / Css
        Shat[k, :] = (1 / G) * rk  # sanity check

        # compute SINDR_k
        pwr_usefull_sig = np.abs(G) ** 2 * Css
        test = np.mean(rk * rk.conj())
        varr = np.var(rk)
        pwr_dist_interference = np.mean(rk * rk.conj()) - np.abs(G) ** 2 * Css
        # noise_vars = 0  # todo sanity check
        # # test
        # M = 64
        # diag_alpha = np.diag(0.6366197723675793 * np.ones(M))
        # diag_beta = np.diag(0.3633802276324208 * np.ones(M))
        #
        # Exxh = 0
        # for i in range(x.shape[-1]):
        #     xint = x[:, i]
        #     xext = xint[:, np.newaxis]
        #     Exxh += xext @ xext.conj().T
        # Exxh /= x.shape[-1]
        # diagExxh = np.diag(np.diag(Exxh))
        #
        # xxhtest = np.mean(x @ x.conj().T, axis=-1)
        # diagExx = np.diag(np.mean(x @ x.conj().T, axis=-1))

        # Rqq = M * diag_alpha @ diag_beta @ diagExxh
        # H = np.ones((M, K))
        # dist_check = np.diag(H.T @ Rqq @ H.conj())

        sindr[k, :] = np.real(pwr_usefull_sig / (pwr_dist_interference + noise_vars))
    R = np.real(np.sum(np.log2(1 + sindr), axis=0))

    # for i in range(5):
    #     plt.scatter(S[0, i].real, S[0, i].imag, label=f'S_{i}')
    #     plt.scatter(Shat[0, i].real, Shat[0, i].imag, label=f'Shat_{i}')
    #     plt.legend()
    # plt.show()

    #assert R.shape == noise_vars.shape, f'shape of R should be {noise_vars.shape} but is {R.shape}'

    # # MMSE of x given r as journal gnn
    # Shat_check = np.zeros_like(r)
    # sindr_check = np.zeros((K, len(noise_vars)))
    # for k in range(K):
    #     sk = S[k, :]
    #     rk = r[k, :]
    #     G = np.mean(sk * rk.conj()) / np.mean(sk * sk.conj())
    #     Shat_check[k, :] = (1 / G) * rk
    #
    #     Css = np.mean(sk * sk.conj())  # should be 1
    #     sig2_s = np.real(np.abs(G) ** 2 * Css)
    #     sig2_id = np.real(np.mean(rk * rk.conj()) - np.abs(G) ** 2 * Css)
    #     sindr_check[k, :] = sig2_s / (sig2_id + noise_vars)
    # R = np.real(np.sum(np.log2(1 + sindr_check), axis=0))

    return R

def main_sim(H, noise_vars, bits=32, quant='uniform', Pt=64):
    K = H.shape[2]
    M = H.shape[1]
    if quant == 'non-uniform':
        #print(f' path used: {quant_params_path}')
        thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
        #print(f'nr outputlevels: {len(outputlevels)}')
        beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))
        diag_beta = np.identity(M) * beta
        diag_alpha = np.identity(M) - diag_beta



    # container for sumrates per channel realization
    R_all = np.zeros((H.shape[0], len(noise_vars)))

    # loop over channel realizations
    for i in trange(H.shape[0], desc='channel realizations'):

        #todo if K=1 do mrt
        # compute precoding matrix
        if H.shape[-1] == 1:
            W = MRT_precoding(H[i, :, :], Pt)
        else:
            W = ZF_precoding(H[i, :, :], Pt)
        #W = MRT_precoding(H[i, :, :], Pt) # sanity check
        #shapew = W.shape
        #assert W.shape == (M, K), f'W should have shape [{M}, {K}] but has shape: {W.shape}'

        # generate symbols
        nrdata = 5000
        S = symbols_MU(K, nrdata=nrdata)
        vars = np.var(S)
        #assert S.shape == (K, nrdata), f'S should have shape [{K}, {nrdata}] but has shape: {S.shape}'

        # precode
        # if quant == 'non-uniform':
        #     alpha = np.sqrt(Pt / np.trace(diag_alpha @ W @ W.conj().T))
        #     W *= alpha
        x = W @ S
        varx = np.var(x, axis=-1)
        #assert x.shape == (M, nrdata), f'x should have shape [{M}, {nrdata}] but has shape: {x.shape}'
        sigvar = np.mean(np.abs(x)**2, axis=-1)


        # pwr_test = np.linalg.norm(x, ord='fro')**2
        # print(f'pwr test before Q: {pwr_test/S.shape[-1]}')
        # print(f'pwr per antenna: {np.linalg.norm(x, axis=-1)**2/ S.shape[-1]}')





        # todo Quantize the precoded vector (Re and Im part seperate)
        # y = x
        if quant == 'uniform':
            y = quantize_uniform(x, bits)
        elif quant == 'non-uniform':
            AGC = False
            if AGC:
                alpha_m = np.zeros(M)
                for m in range(M):
                    Wm = W[m, :]
                    check = Wm.T @ Wm.conj()
                    alpha_m[m] = Wm.T @ Wm.conj()

                x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
                varx_scaled = np.var(x_scaled, axis=-1)
                xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)
                y = np.sqrt(alpha_m)[:, np.newaxis] * xq
                vary = np.var(y, axis=-1)
                mse = np.mean(np.abs(x - y) ** 2, axis=-1)
                nmse = mse / sigvar
                nmse_avg = np.mean(nmse)

                #no AGC as test
                ycheck = quantize_nonuniform(x, thresholds, outputlevels)
                mse_check = np.mean(np.abs(x - ycheck) ** 2, axis=-1)
                nmse_check = mse_check / sigvar
                nmse_avg_check = np.mean(nmse_check)
                print(f'nmse NO AGC: {nmse_avg}')


                print(f'nmse AGC: {nmse_avg}')

            else:
                y = quantize_nonuniform(x, thresholds, outputlevels)
                mse = np.mean(np.abs(x - y) ** 2, axis=-1)
                nmse = mse / sigvar
                nmse_avg = np.mean(nmse)
                Exqxqh = y @ y.T.conj() / y.shape[-1]
                test = np.cov(x-y)
                offdiag = test - np.diag(np.diag(test))
                sumoffdiag = np.sum(offdiag)
                print(f'nmse NO AGC: {nmse=}')


            # pwr_test = np.linalg.norm(y, ord='fro') ** 2
            # print(f'pwr test after Q: {pwr_test/S.shape[-1]}')
            #
            # #PA function to set pwr back to Pt
            # alpha = np.sqrt(Pt / (pwr_test/S.shape[-1]))
            # y *= alpha
            # pwr_test = np.linalg.norm(y, ord='fro') ** 2
            # print(f'pwr test after Q after PA: {pwr_test/S.shape[-1]}')

        elif quant == 'none':
            y = x


        # send over channel
        r = H[i, :, :].T @ y
        #assert r.shape == (K, nrdata), f'r should have shape ({K}, {nrdata}) but has shape: {r.shape}'

        # compute sumrate
        R_all[i, :] = sumrate(S, r, noise_vars, x=x)

    Ravg = np.mean(R_all, axis=0)
    return Ravg

if __name__ == "__main__":
    #todo comment out asserts to speed up simulations!!!

    # basic sim params
    M, K = 64, 16
    Pt = M
    nr_snr_points = 20
    snr_tx = np.linspace(-30, 35, nr_snr_points)
    #snr_tx = np.linspace(-10, 15, nr_snr_points)
    noise_vars = Pt / (10 ** (snr_tx / 10))
    snr_tx_final = 10 * np.log10(Pt / noise_vars)
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')
    # generate channels
    channel_realizations = 100
    H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
    np.random.seed(0)
    for i in range(channel_realizations):
        H[i, :, :] = rayleigh_channel_MU(M, K)
        #H[i, :, :] = np.ones((M, K)) # sanity check
        #H[i, :, :] = los_channel_MU(M, K)



    # # single test for debugging
    # Ravg = main_sim(H[0:10, :, :], noise_vars, bits=5, quant='non-uniform')
    # plt.plot(snr_tx_final, Ravg, label=f'{4} bits')
    # plt.xlabel(r'SNR = $\frac{P_T }{ \sigma_{\nu}^2}$ [dB]')
    # plt.ylabel(r'$R_{\mathrm{sum}}$ [bits/channel use]')
    # plt.legend()
    # plt.show()


    # test for different nrs of bits
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    nr_channels = 100

    for i in range(1, 2):
        print(f'-------------------------------------{i} bits -----------------------------------')
        Ravg = main_sim(H[0:nr_channels, :, :], noise_vars, bits=i, quant='non-uniform', Pt=Pt)
        print(f'SINDR: {10*np.log10(2**Ravg-1)}')
        #plt.plot(snr_tx_final, 10*np.log10(2**Ravg-1), label=f'sindr [dB] {i} bits')
        plt.plot(snr_tx_final, Ravg, color=colors[i-1], label=f'{i} bits non-uniform')
        # compute for uniform quantizer
        # Ravg = main_sim(H[0:nr_channels, :, :], noise_vars, bits=i, quant='uniform', Pt=Pt)
        # plt.plot(snr_tx_final, Ravg, color=colors[i-1], linestyle='dashed', label=f'{i} bits uniform')

    Ravg = main_sim(H[0:nr_channels, :, :], noise_vars, bits=i, quant='none', Pt=Pt)
    print(f'SINDR: {10 * np.log10(2 ** Ravg - 1)}')
    #plt.plot(snr_tx_final, 10*np.log10(2**Ravg-1), label=f'sindr [dB] inf bits')
    plt.plot(snr_tx_final, Ravg, color='tab:cyan', label=f'inf res')
    plt.title('numerical')
    plt.xlabel(r'SNR = $\frac{P_T }{ \sigma_{\nu}^2}$ [dB]')
    plt.ylabel(r'$R_{\mathrm{sum}}$ [bits/channel use]')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title(f'param N(0,{varx}) - numerical M: {M} K:{K} Pt: {Pt}')
    #plt.savefig("r_vs_bits.svg")
    plt.show()

    #print(f'{Ravg=}')








