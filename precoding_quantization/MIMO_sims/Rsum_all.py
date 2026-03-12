import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from utils.utils import rayleigh_channel_MU, symbols_MU, los_channel_MU
from utils.quantization import quantize_uniform, quantize_nonuniform
from utils.precoding import ZF_precoding, MRT_precoding
import os
import scipy.stats
import seaborn as sns

def bussgang_wrt_s(H, y, S, x, noise_vars):
    """
    :param H: channel M x K
    :param y: precoded quantized signals M x nr_symb
    :param S: inteded symbls K x nr_symb
    :param noise_vars: noise variances to evaluate sumrate at
    :return: r sum per channel
    """
    nr_symb = y.shape[-1]

    # decompose y = Gs + q (G: MxK)
    G = ((y @ S.conj().T) / nr_symb)  # = E(y s^H)
    cov_s = (S @ S.conj().T) / nr_symb # = E(s s^H) = diag(1)?
    G = G @ np.linalg.inv(cov_s)
    q = y - G @ S

    # sanity check on G for linear precoding: todo
    W = ZF_precoding(H, H.shape[0])
    I_alpha = np.diag(np.mean(y*x.conj(), axis=-1) / np.mean(np.abs(x)**2, axis=-1))
    I_alpha_W = I_alpha @ W

    diff = G - I_alpha_W

    # alpha_m = np.zeros(M)
    # for m in range(M):
    #     Wm = W[m, :]
    #     alpha_m[m] = Wm.T @ Wm.conj()



    # some stuff we will reuse later
    HT = H.T
    HTG = HT @ G

    # intended sig
    inteded_sig = np.abs(np.diag(HTG))**2

    # user interference
    user_interference = np.sum(np.abs(HTG)**2, axis=-1) - inteded_sig

    # distortion
    Cq = (q @ q.conj().T)/nr_symb # = E(qq^H)
    dist = np.diag(HT @ Cq @ H.conj())

    # compute sindr
    sindr = inteded_sig[:, np.newaxis] / (user_interference[:, np.newaxis] + dist[:, np.newaxis] + noise_vars[np.newaxis, :])

    # rate per user
    R = np.log2(1 + sindr)

    # sum rate
    R_sum = np.real(np.sum(R, axis=0))

    print(f'debug')


    return R_sum

def Rsum_Bussgang_generalized_wrt_s(H, snr_points, bits=32, quant='uniform', Pt=64, correlated_dist=True, automatic_gain_control=True,
                     quant_params_path='', precoding='zf-mrt', precoding_weights=None, x_nonlin=None, s_provided=None,
                     normalize_across_symbols=False):
    """ Computes the generalized Bussgang decomposition between y and s
    - Fully numerical simulation:
    :param H: channel realizations, nr_channels x M x K
    :param snr_points: SNR points to compute the sum rate at
    :param bits: nr of bits in the DAC
    :param quant: type of quantization 'uniform' or 'non-unirform'
    :param Pt: avg pwr constraint
    :param correlated_dist: unused variable for consistency
    :param automatic_gain_control: automatically scale input to quantizer and rescale the output to achieve optimal quantization
    :params quant_params_path: path to thresholds and outputlevels of the considered DAC/quantizer
    :params precoding: type of precoding used
    :params precoding_weights: weights for precoding (only used if linear neural net based precoding is considered, nr_channels x M x K
    :params x_nonlin: precoded vector (only used when nonlinear precoding is considered), nr_channels x M x 1
    :params s_provided: symbols to use for computing the sumrate (only used when nonlinear precoding is considred or
    when control over the used symbols is desired), nr_channels x K x 1
    :return:
    """
    assert correlated_dist == True, \
        f'Distortion is always correlated in the numerical computation of Rsum, change correlated_dist argument to True'

    if precoding == 'non-linear' and s_provided is None:
        raise AssertionError('non linear precoding is selected but no symbols are provided!')


    # construct vector of noise variances connected to the SNR points
    noise_vars = Pt / (10 ** (snr_points / 10))

    K = H.shape[-1]
    M = H.shape[-2]

    if quant == 'non-uniform':
        print(f'non uniform quantization used')
        print(f'automatic gain control: {automatic_gain_control}')
        thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
        # outputlevels = np.sqrt(M / (2 * M)) * np.array([-1, 1])
        # print(f'{thresholds=}')
        # print(f'{outputlevels=}')

    elif quant == 'uniform':
        print(f'uniform quantization used')
        raise Exception("Uniform quantization is not implemented yet")
        #todo
        pass
    elif quant == 'none':
        print(f'no quantization used')
    else:
        raise Exception("invalid quantization type")

    # container for sumrates per channel realization
    R_all = np.zeros((H.shape[0], len(noise_vars)))

    # loop over channel realizations
    for i in trange(H.shape[0], desc='channel realizations'):

        # compute precoding matrix
        if precoding == 'zf-mrt':
            if K == 1:
                W = MRT_precoding(H[i, :, :], Pt)
            else:
                W = ZF_precoding(H[i, :, :], Pt)
        elif precoding == 'gnn':
            y_pred = precoding_weights[i, :, :, :] #(1 , M, K, 2) GNN output for current channel
            Wre = y_pred[:, :, 0]
            Wim = y_pred[:, :, 1]
            W = Wre + 1j * Wim  # M x K
        elif precoding == 'non-linear': #precoding vector is given, so no need to precode
            print(f'non linear precoding is used')
        else:
            assert False, 'invalid precoding type selected in Rsum_Bussgang_Rx function'

        # generate symbols
        if s_provided is None:
            nrdata = 5000
            S = symbols_MU(K, nrdata=nrdata)
        else:
            S = s_provided[i, :, :]

        # precode
        if precoding != 'non-linear':
            x = W @ S
        else:
            x = x_nonlin[i, :]
            S = s_provided[i, :, :]
            quant = 'none' # nonlinear precoding implies that the output is directly quantized => no more quantization needed

        sigvar = np.var(x, axis=-1)

        # Quantize the precoded vector (Re and Im part seperate)
        if quant == 'uniform':
            y = quantize_uniform(x, bits)
        elif quant == 'non-uniform':
            if automatic_gain_control:
                # compute alpha to scale DAC input per antenna
                alpha_m = np.zeros(M)
                for m in range(M):
                    Wm = W[m, :]
                    alpha_m[m] = Wm.T @ Wm.conj()

                # scale DAC input
                x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
                varx_scaled = np.var(x_scaled, axis=-1)

                # quantize
                xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

                # rescale the output of the DAC
                y = np.sqrt(alpha_m)[:, np.newaxis] * xq
                vary = np.var(y, axis=-1)

            else:
                # quantize
                y = quantize_nonuniform(x, thresholds, outputlevels)

        elif quant == 'none':
            y = x

        if normalize_across_symbols:
            l2_norm = np.linalg.norm(y, ord=2, axis=0)  # bs x nr_symbols
            expt_x2 = np.mean(l2_norm ** 2, axis=-1)
            epsilon = 1e-7  # to avoid NaN
            alpha = np.sqrt(Pt / (expt_x2 + epsilon))
            y = alpha * y


        # check avg output pwr (y has shape M x nr_symbols)
        l2_norm_post_norm = np.linalg.norm(y, ord=2, axis=0)  # bs x nr_symbols
        expt_x2_post_norm = np.mean(l2_norm_post_norm ** 2, axis=-1)
        #print(f'precding used: {precoding} - quantization: {quant} - avg pwr post norm {expt_x2_post_norm=}')


        R_all[i, :] = bussgang_wrt_s(H[i, :, :], y, S, x, noise_vars)

        # # send over channel
        # r = H[i, :, :].T @ y
        #
        # # fully numerical sim: compute bussgang decomposition at the receiver
        # R_all[i, :] = bussgang_at_receiver(S, r, noise_vars, x=None)

    Rsum_avg = np.mean(R_all, axis=0)
    return Rsum_avg




def bussgang_at_receiver(S, r, noise_vars, x=None):
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

        sindr[k, :] = np.real(pwr_usefull_sig / (pwr_dist_interference + noise_vars))
    R = np.real(np.sum(np.log2(1 + sindr), axis=0))
    return R



def rsum_analytical_loopy(H, W, diag_alpha, Rqq, noise_vars):
    alpha = diag_alpha[0, 0]
    Rsum = 0
    M, K = H.shape[-2], H.shape[-1]
    for k in range(K):
        hk = H[:, k]
        wk = W[:, k]

        # usefull sig
        alpha = diag_alpha[0, 0]
        #sig = alpha ** 2 * np.abs(hk.T @ wk) ** 2 # seems to be wrong
        sig = np.abs(hk.T @ diag_alpha @ wk) ** 2

        # inteference
        interference = 0
        for kprime in range(K):
            hkprime = H[:, kprime]
            wkprime = W[:, kprime]
            if kprime != k:
                interference += np.abs(hk.T @ diag_alpha @ wkprime) ** 2

        # dist
        dist = hk.T @ Rqq @ hk.conj()

        # sinqdr
        sinqdrk = np.real(sig / (interference + dist + noise_vars))

        Rk = np.log2(1 + sinqdrk)

        Rsum += Rk
    return Rsum

def rsum_analytical_vectorized(H, W, diag_alpha, Rqq, noise_vars):
    """
    :param H: M x K channel matrix
    :param W: M x K precoding matrix
    :param diag_alpha: diagonal matrix containing the bussgang gains
    :param Rqq: covariance matrix of the distortion
    :param noise_vars: noise variances at which to compute the sum rate
    :return: sumrate at different SNR points
    """

    # compute usefull sig
    HdiagalphaW = H.T @ diag_alpha @ W
    usefull_sig = np.diag(HdiagalphaW)
    usefull_sig_pwr = np.abs(usefull_sig)**2

    # inter user interference
    interference = np.sum(np.abs(HdiagalphaW - np.diag(usefull_sig)) ** 2, axis=-1)

    # distortion
    dist = np.diag(H.T @ Rqq @ H.conj())

    # noise var
    noisevar = noise_vars[:, np.newaxis]

    # sindr
    sindr = usefull_sig_pwr / (interference + dist + noisevar)

    # rate per user
    Rk = np.real(np.log2(1 + sindr)) # nr_noisevars x K

    # sum rate
    Rsum = np.sum(Rk, axis=-1)

    return Rsum

def Rsum_Bussgang_DAC(H, snr_points, bits=32, quant='uniform', Pt=64, correlated_dist=True,
                      automatic_gain_control=True, quant_params_path='',  precoding='zf-mrt', precoding_weights=None):
    """
    :param H: channel realizations
    :param snr_points: SNR points to compute the sum rate at
    :param bits: nr of bits in the DAC
    :param quant: type of quantization 'uniform' or 'non-unirform'
    :param Pt: avg pwr constraint
    :param correlated_dist: approximate distortion as uncorrelated or not
    :param automatic_gain_control: automatically scale input to quantizer and rescale the output to achieve optimal quantization
    :return:
    """
    # assert correlated_dist == True, \
    #     f'Distortion is always correlated in the numerical computation of Rsum, change correlated_dist argument to True'


    # construct vector of noise variances connected to the SNR points
    noise_vars = Pt / (10 ** (snr_points / 10))

    K = H.shape[-1]
    M = H.shape[-2]

    if quant == 'non-uniform':
        print(f'non uniform quantization used')
        print(f'automatic gain control: {automatic_gain_control}')
        #print(f' path used: {quant_params_path}')
        thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
        #print(f'nr outputlevels: {len(outputlevels)}')
        beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))
        diag_beta = np.identity(M) * beta
        diag_alpha = np.identity(M) - diag_beta
    elif quant == 'uniform':
        print(f'uniform quantization used')
        raise Exception("Uniform quantization is not implemented yet")
        #todo
        pass
    elif quant == 'none':
        print(f'no quantization used')
    else:
        raise Exception("invalid quantization type")



    # container for sumrates per channel realization
    R_all = np.zeros((H.shape[0], len(noise_vars)))

    # loop over channel realizations
    for i in trange(H.shape[0], desc='channel realizations'):

        # compute precoding matrix
        if precoding == 'zf-mrt':
            if K == 1:
                W = MRT_precoding(H[i, :, :], Pt)
            else:
                W = ZF_precoding(H[i, :, :], Pt)
        elif precoding == 'gnn':
            y_pred = precoding_weights[i, :, :, :]  # (1 , M, K, 2) GNN output for current channel
            Wre = y_pred[:, :, 0]
            Wim = y_pred[:, :, 1]
            W = Wre + 1j * Wim  # M x K
        else:
            assert False, 'invalid precoding type selected in Rsum_Bussgang_Rx function'

        # generate symbols
        nrdata = 5000
        S = symbols_MU(K, nrdata=nrdata)

        # precode
        x = W @ S
        sigvar = np.var(x, axis=-1)

        # Quantize the precoded vector (Re and Im part seperate)
        if quant == 'uniform':
            y = quantize_uniform(x, bits)
        elif quant == 'non-uniform':
            if automatic_gain_control:
                # compute alpha to scale DAC input per antenna
                alpha_m = np.zeros(M)
                for m in range(M):
                    Wm = W[m, :]
                    alpha_m[m] = Wm.T @ Wm.conj()

                # scale DAC input
                x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
                varx_scaled = np.var(x_scaled, axis=-1)

                # # check for normality
                # sns.displot(np.real(x_scaled[0, :]), kde=True, bins=30)
                # plt.title(f'var: {np.var(np.real(x_scaled), axis=-1)[0]}')
                # plt.show()


                # quantize
                xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

                # rescale the output of the DAC
                y = np.sqrt(alpha_m)[:, np.newaxis] * xq
                vary = np.var(y, axis=-1)

                # compute mse and NMSE (for debugging)
                # mse = np.mean(np.abs(x - y) ** 2, axis=-1)
                # nmse = mse / sigvar
                # nmse_avg = np.mean(nmse)
                #print(f'nmse AGC: {nmse_avg}')

                # #no AGC as test
                # ycheck = quantize_nonuniform(x, thresholds, outputlevels)
                # mse_check = np.mean(np.abs(x - ycheck) ** 2, axis=-1)
                # nmse_check = mse_check / sigvar
                # nmse_avg_check = np.mean(nmse_check)
                # print(f'nmse NO AGC: {nmse_avg}')
            else:
                # quantize
                y = quantize_nonuniform(x, thresholds, outputlevels)

                # compute mse and NMSE (for debugging)
                mse = np.mean(np.abs(x - y) ** 2, axis=-1)
                nmse = mse / sigvar
                nmse_avg = np.mean(nmse)

                # Exqxqh = y @ y.T.conj() / y.shape[-1]
                # test = np.cov(x-y)
                # offdiag = test - np.diag(np.diag(test))
                # sumoffdiag = np.sum(offdiag)
                # print(f'nmse NO AGC: {nmse=}')

        elif quant == 'none':
            y = x

        # compute bussganggain after each DAC (x = M x Nrdata)
        alpha_m = np.diag(np.mean(y * x.conj(), axis=-1) / np.mean(np.abs(x)**2, axis=-1))
        q = y - alpha_m @ x
        covq = np.cov(q)
        if correlated_dist:
            Rsum = rsum_analytical_loopy(H[i, :, :], W, alpha_m, covq, noise_vars)
        elif correlated_dist == False:
            Rsum = rsum_analytical_loopy(H[i, :, :], W, alpha_m, np.diag(np.diag(covq)), noise_vars)

        # store Rsum for this channel realization
        R_all[i, :] = Rsum

        # todo remove this to a testing function
        # # theoretical alpha based on NMSE (beta)
        # alpha_m_theoretical = np.eye(M) - diag_beta
        # alpha = 1 - beta
        # covq_theoretical = beta * alpha * np.diag(np.diag(W @ W.conj().T))
        # Rsum_theoretical = rsum_analytical_loopy(H, W, alpha_m_theoretical, covq_theoretical)
        # check = np.diag(covq) - np.diag(covq_theoretical)
        # print(f'check diag elements of covq - covq_theoretical: {check}')


        # # fully numerical sim with bussgang decomposition at the receiver
        # # send over channel
        # r = H.T @ y
        # Rsum_num_bussgang_at_Rx = sumrate_bussgang_at_receiver(S, r, x=None)
        # print("debug")

    Rsum_avg = np.mean(R_all, axis=0)
    return Rsum_avg

def Rsum_Bussgang_Rx(H, snr_points, bits=32, quant='uniform', Pt=64, correlated_dist=True, automatic_gain_control=True,
                     quant_params_path='', precoding='zf-mrt', precoding_weights=None, x_nonlin=None, s_provided=None,
                     normalize_across_symbols=False):
    """
    - Fully numerical simulation:
    :param H: channel realizations, nr_channels x M x K
    :param snr_points: SNR points to compute the sum rate at
    :param bits: nr of bits in the DAC
    :param quant: type of quantization 'uniform' or 'non-unirform'
    :param Pt: avg pwr constraint
    :param correlated_dist: unused variable for consistency
    :param automatic_gain_control: automatically scale input to quantizer and rescale the output to achieve optimal quantization
    :params quant_params_path: path to thresholds and outputlevels of the considered DAC/quantizer
    :params precoding: type of precoding used
    :params precoding_weights: weights for precoding (only used if linear neural net based precoding is considered, nr_channels x M x K
    :params x_nonlin: precoded vector (only used when nonlinear precoding is considered), nr_channels x M x 1
    :params s_provided: symbols to use for computing the sumrate (only used when nonlinear precoding is considred or
    when control over the used symbols is desired), nr_channels x K x 1
    :return:
    """
    assert correlated_dist == True, \
        f'Distortion is always correlated in the numerical computation of Rsum, change correlated_dist argument to True'

    if precoding == 'non-linear' and s_provided is None:
        raise AssertionError('non linear precoding is selected but no symbols are provided!')


    # construct vector of noise variances connected to the SNR points
    noise_vars = Pt / (10 ** (snr_points / 10))

    K = H.shape[-1]
    M = H.shape[-2]

    if quant == 'non-uniform':
        print(f'non uniform quantization used')
        print(f'automatic gain control: {automatic_gain_control}')
        thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
        # outputlevels = np.sqrt(M / (2 * M)) * np.array([-1, 1])
        # print(f'{thresholds=}')
        # print(f'{outputlevels=}')

    elif quant == 'uniform':
        print(f'uniform quantization used')
        raise Exception("Uniform quantization is not implemented yet")
        #todo
        pass
    elif quant == 'none':
        print(f'no quantization used')
    else:
        raise Exception("invalid quantization type")

    # container for sumrates per channel realization
    R_all = np.zeros((H.shape[0], len(noise_vars)))

    # loop over channel realizations
    for i in trange(H.shape[0], desc='channel realizations'):

        # compute precoding matrix
        if precoding == 'zf-mrt':
            if K == 1:
                W = MRT_precoding(H[i, :, :], Pt)
            else:
                W = ZF_precoding(H[i, :, :], Pt)
        elif precoding == 'gnn':
            y_pred = precoding_weights[i, :, :, :] #(1 , M, K, 2) GNN output for current channel
            Wre = y_pred[:, :, 0]
            Wim = y_pred[:, :, 1]
            W = Wre + 1j * Wim  # M x K
        elif precoding == 'non-linear': #precoding vector is given, so no need to precode
            print(f'non linear precoding is used')
        else:
            assert False, 'invalid precoding type selected in Rsum_Bussgang_Rx function'

        # generate symbols
        if s_provided is None:
            nrdata = 5000
            S = symbols_MU(K, nrdata=nrdata)
        else:
            S = s_provided[i, :, :]

        # precode
        if precoding != 'non-linear':
            x = W @ S
        else:
            x = x_nonlin[i, :]
            S = s_provided[i, :, :]
            quant = 'none' # nonlinear precoding implies that the output is directly quantized => no more quantization needed

        sigvar = np.var(x, axis=-1)

        # Quantize the precoded vector (Re and Im part seperate)
        if quant == 'uniform':
            y = quantize_uniform(x, bits)
        elif quant == 'non-uniform':
            if automatic_gain_control:
                # compute alpha to scale DAC input per antenna
                alpha_m = np.zeros(M)
                for m in range(M):
                    Wm = W[m, :]
                    alpha_m[m] = Wm.T @ Wm.conj()

                # scale DAC input
                x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
                varx_scaled = np.var(x_scaled, axis=-1)

                # quantize
                xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

                # rescale the output of the DAC
                y = np.sqrt(alpha_m)[:, np.newaxis] * xq
                vary = np.var(y, axis=-1)

            else:
                # quantize
                y = quantize_nonuniform(x, thresholds, outputlevels)

        elif quant == 'none':
            y = x

        if normalize_across_symbols:
            l2_norm = np.linalg.norm(y, ord=2, axis=0)  # bs x nr_symbols
            expt_x2 = np.mean(l2_norm ** 2, axis=-1)
            epsilon = 1e-7  # to avoid NaN
            alpha = np.sqrt(Pt / (expt_x2 + epsilon))
            y = alpha * y


        # check avg output pwr (y has shape M x nr_symbols)
        l2_norm_post_norm = np.linalg.norm(y, ord=2, axis=0)  # bs x nr_symbols
        expt_x2_post_norm = np.mean(l2_norm_post_norm ** 2, axis=-1)
        #print(f'precding used: {precoding} - quantization: {quant} - avg pwr post norm {expt_x2_post_norm=}')

        # send over channel
        r = H[i, :, :].T @ y

        # fully numerical sim: compute bussgang decomposition at the receiver
        R_all[i, :] = bussgang_at_receiver(S, r, noise_vars, x=None)

    Rsum_avg = np.mean(R_all, axis=0)
    return Rsum_avg


def Rsum_Bussgang_Rx_per_channel(H, snr_points, bits=32, quant='uniform', Pt=64, correlated_dist=True, automatic_gain_control=True,
                     quant_params_path='', precoding='zf-mrt', precoding_weights=None, x_nonlin=None, s_provided=None,
                     normalize_across_symbols=False):
    """
    - Fully numerical simulation:
    :param H: channel realizations, nr_channels x M x K
    :param snr_points: SNR points to compute the sum rate at
    :param bits: nr of bits in the DAC
    :param quant: type of quantization 'uniform' or 'non-unirform'
    :param Pt: avg pwr constraint
    :param correlated_dist: unused variable for consistency
    :param automatic_gain_control: automatically scale input to quantizer and rescale the output to achieve optimal quantization
    :params quant_params_path: path to thresholds and outputlevels of the considered DAC/quantizer
    :params precoding: type of precoding used
    :params precoding_weights: weights for precoding (only used if linear neural net based precoding is considered, nr_channels x M x K
    :params x_nonlin: precoded vector (only used when nonlinear precoding is considered), nr_channels x M x 1
    :params s_provided: symbols to use for computing the sumrate (only used when nonlinear precoding is considred or
    when control over the used symbols is desired), nr_channels x K x 1
    :return:
    """
    assert correlated_dist == True, \
        f'Distortion is always correlated in the numerical computation of Rsum, change correlated_dist argument to True'

    if precoding == 'non-linear' and s_provided is None:
        raise AssertionError('non linear precoding is selected but no symbols are provided!')


    # construct vector of noise variances connected to the SNR points
    noise_vars = Pt / (10 ** (snr_points / 10))

    K = H.shape[-1]
    M = H.shape[-2]

    if quant == 'non-uniform':
        print(f'non uniform quantization used')
        print(f'automatic gain control: {automatic_gain_control}')
        thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
        # outputlevels = np.sqrt(M / (2 * M)) * np.array([-1, 1])
        # print(f'{thresholds=}')
        # print(f'{outputlevels=}')

    elif quant == 'uniform':
        print(f'uniform quantization used')
        raise Exception("Uniform quantization is not implemented yet")
        #todo
        pass
    elif quant == 'none':
        print(f'no quantization used')
    else:
        raise Exception("invalid quantization type")

    # container for sumrates per channel realization
    R_all = np.zeros((H.shape[0], len(noise_vars)))

    # loop over channel realizations
    for i in trange(H.shape[0], desc='channel realizations'):

        # compute precoding matrix
        if precoding == 'zf-mrt':
            if K == 1:
                W = MRT_precoding(H[i, :, :], Pt)
            else:
                W = ZF_precoding(H[i, :, :], Pt)
        elif precoding == 'gnn':
            y_pred = precoding_weights[i, :, :, :] #(1 , M, K, 2) GNN output for current channel
            Wre = y_pred[:, :, 0]
            Wim = y_pred[:, :, 1]
            W = Wre + 1j * Wim  # M x K
        elif precoding == 'non-linear': #precoding vector is given, so no need to precode
            print(f'non linear precoding is used')
        else:
            assert False, 'invalid precoding type selected in Rsum_Bussgang_Rx function'

        # generate symbols
        if s_provided is None:
            nrdata = 5000
            S = symbols_MU(K, nrdata=nrdata)
        else:
            S = s_provided[i, :, :]

        # precode
        if precoding != 'non-linear':
            x = W @ S
        else:
            x = x_nonlin[i, :]
            S = s_provided[i, :, :]
            quant = 'none' # nonlinear precoding implies that the output is directly quantized => no more quantization needed

        sigvar = np.var(x, axis=-1)

        # Quantize the precoded vector (Re and Im part seperate)
        if quant == 'uniform':
            y = quantize_uniform(x, bits)
        elif quant == 'non-uniform':
            if automatic_gain_control:
                # compute alpha to scale DAC input per antenna
                alpha_m = np.zeros(M)
                for m in range(M):
                    Wm = W[m, :]
                    alpha_m[m] = Wm.T @ Wm.conj()

                # scale DAC input
                x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
                varx_scaled = np.var(x_scaled, axis=-1)

                # quantize
                xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

                # rescale the output of the DAC
                y = np.sqrt(alpha_m)[:, np.newaxis] * xq
                vary = np.var(y, axis=-1)

            else:
                # quantize
                y = quantize_nonuniform(x, thresholds, outputlevels)

        elif quant == 'none':
            y = x

        if normalize_across_symbols:
            l2_norm = np.linalg.norm(y, ord=2, axis=0)  # bs x nr_symbols
            expt_x2 = np.mean(l2_norm ** 2, axis=-1)
            epsilon = 1e-7  # to avoid NaN
            alpha = np.sqrt(Pt / (expt_x2 + epsilon))
            y = alpha * y


        # check avg output pwr (y has shape M x nr_symbols)
        l2_norm_post_norm = np.linalg.norm(y, ord=2, axis=0)  # bs x nr_symbols
        expt_x2_post_norm = np.mean(l2_norm_post_norm ** 2, axis=-1)
        #print(f'precding used: {precoding} - quantization: {quant} - avg pwr post norm {expt_x2_post_norm=}')

        # send over channel
        r = H[i, :, :].T @ y

        # fully numerical sim: compute bussgang decomposition at the receiver
        R_all[i, :] = bussgang_at_receiver(S, r, noise_vars, x=None)

    #Rsum_avg = np.mean(R_all, axis=0)
    return R_all #nr_channels x nr_snr_points

def Rsum_analytical_wrapper(H, snr_points, bits=32, quant='uniform', Pt=64, correlated_dist=False, automatic_gain_control=True, quant_params_path='',
                            precoding='zf-mrt', precoding_weights=None):
    """
    Compute the sum rate fully analytical way
    :param H: channel realizations
    :param snr_points: SNR points to compute the sum rate at
    :param bits: nr of bits in the DAC
    :param quant: type of quantization 'uniform' or 'non-unirform'
    :param Pt: avg pwr constraint
    :param correlated_dist: always false for analytical expression (unused variable for consistency)
    :param automatic_gain_control: always assumed true for analytical expression (unused variable for consistency)
    :return:
    :return:
    """

    assert correlated_dist == False, \
        f'Distortion is always assumed to be uncorrelated when using the analytical expression, change correlated_dist argument to True'
    assert automatic_gain_control == True, \
        f'analytical expression assumes perfect AGC, change automatic_gain_control argument to True'

    # get nr antennas and users
    M, K = H.shape[-2], H.shape[-1]

    # load beta parameters based on quantization type
    if bits == 'inf' or quant == 'none':
        beta = 0
    elif quant == 'non-uniform':
        print(f'non uniform quantization used')
        # read params of quantizer
        beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))
    elif quant == 'uniform':
        print(f'uniform quantization used')
        raise Exception("Uniform quantization is not implemented yet")
    else:
        raise Exception("invalid quantization type")

    # construct vector of noise variances connected to the SNR points
    noise_vars = Pt / (10 ** (snr_points / 10))

    # construct diag_alpha and diag_beta (with loaded values from file)
    diag_beta = np.identity(M) * beta
    diag_alpha = np.identity(M) - diag_beta

    # container to store sumrates
    Rsum_all = np.zeros((H.shape[0], len(snr_points)), dtype=float)

    # loop over channel realizations
    for i in range(H.shape[0]):

        # compute precoding matrix
        if precoding == 'zf-mrt':
            if K == 1:
                W = MRT_precoding(H[i, :, :], Pt)
            else:
                W = ZF_precoding(H[i, :, :], Pt)
        elif precoding == 'gnn':
            y_pred = precoding_weights[i, :, :, :]  # (1 , M, K, 2) GNN output for current channel
            Wre = y_pred[:, :, 0]
            Wim = y_pred[:, :, 1]
            W = Wre + 1j * Wim  # M x K
        else:
            assert False, 'invalid precoding type selected in Rsum_Bussgang_Rx function'


        #todo turn this on for final checks
        # alpha = np.sqrt(Pt / np.trace(diag_alpha @ W @ W.conj().T))
        # W *= alpha
        # this has minimal impact as this 'amplifies' both the linear signal and the distortion
        # it only affects the performance at low SNR, as then you are noise limited and not distortion limited
        # for high snr distortion is the limiting factor so increasing Pt doesnt help

        # print(f'pwr before Q: {pwr_test}')
        # # check
        # print(f'pwr after Q: {np.trace(diag_alpha @ W @ W.conj().T)}')

        Rqq = diag_alpha @ diag_beta @ np.diag(np.diag(W @ W.conj().T))
        Rsum_all[i, :] = rsum_analytical_loopy(H[i, :, :], W, diag_alpha, Rqq, noise_vars)

    # take avg over channel realizations
    Rsum_avg = np.mean(Rsum_all, axis=0)
    return Rsum_avg

if __name__ == "__main__":
    # basic sim params
    M, K = 8, 1
    Pt = M
    nr_snr_points = 20
    snr_tx = np.array([-30, -20, -10, 0.1, 10, 20, 30])#np.linspace(-30, 30, nr_snr_points)

    # path to quantizer parameters
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params' #r'D:\thomas.feys\Quantization\precoding_quantization\non-uniform-quant-params'#
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # generate channels
    channel_realizations = 1
    H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
    np.random.seed(1)
    for i in range(channel_realizations):
        H[i, :, :] = rayleigh_channel_MU(M, K)
        #H[i, :, :] = np.ones((M, K)) # sanity check
        #H[i, :, :] = los_channel_MU(M, K)



    # test for different nrs of bits
    for i in range(1, 2):
        print(f'-------------------------------------{i} bits -----------------------------------')
        Ravg_bussgang_dac = Rsum_Bussgang_DAC(H, snr_tx, bits=i, quant='non-uniform',
                                 Pt=Pt, correlated_dist=True, automatic_gain_control=True, quant_params_path=quant_params_path)

        Ravg_bussgang_dac_uncorllated_dist = Rsum_Bussgang_DAC(H, snr_tx, bits=i, quant='non-uniform',
                                 Pt=Pt, correlated_dist=False, automatic_gain_control=True, quant_params_path=quant_params_path)

        Ravg_bussgang_rx = Rsum_Bussgang_Rx(H, snr_tx, bits=i, quant='non-uniform',
                                 Pt=Pt, correlated_dist=True, automatic_gain_control=True, quant_params_path=quant_params_path)

        Ravg_fully_analytical = Rsum_analytical_wrapper(H, snr_tx, bits=i, quant='non-uniform',
                                 Pt=Pt, correlated_dist=False, automatic_gain_control=True, quant_params_path=quant_params_path)

        plt.plot(snr_tx, Ravg_bussgang_dac, label=f'{i} bits non-uniform, Bussgang DAC')
        plt.plot(snr_tx, Ravg_bussgang_rx, label=f'{i} bits non-uniform, Bussgang Rx')
        plt.plot(snr_tx, Ravg_bussgang_dac_uncorllated_dist, label=f'{i} bits non-uniform, Bussgang DAC uncorrelated dist')
        plt.plot(snr_tx, Ravg_fully_analytical, label=f'{i} bits non-uniform, Analytical')
        print(f'bits: {i}')
        print(f'{snr_tx=}')
        print(f'{Ravg_bussgang_rx=}')



    plt.title('Bussgang after DAC')
    plt.xlabel(r'SNR = $\frac{P_T }{ \sigma_{\nu}^2}$ [dB]')
    plt.ylabel(r'$R_{\mathrm{sum}}$ [bits/channel use]')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title(f'param N(0,{varx}) - numerical M: {M} K:{K} Pt: {Pt}')
    #plt.savefig("r_vs_bits.svg")
    plt.show()






