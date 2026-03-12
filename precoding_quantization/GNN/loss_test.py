from losses import quant_loss_uncorrelated_Rqq, quantize_nonuniform_tf
from MIMO_sims.Rsum_all import rsum_analytical_loopy, rsum_analytical_vectorized
from utils.utils import rayleigh_channel_MU, los_channel_MU, C2R, symbols_MU
from utils.precoding import ZF_precoding, MRT_precoding
from utils.quantization import quantize_nonuniform
import os
import numpy as np
import tensorflow as tf


def test_tf_quant_equals_numpy_multiple_batches():
    # basic sim params
    M, K = 16, 2
    Pt = M
    nr_snr_points = 20
    snr_tx = 20
    noise_var = Pt / (10 ** (snr_tx / 10))

    # path to quantizer parameters
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # generate channel
    bs = 5
    H_batched = np.zeros((bs, M, K), dtype=np.complex64)
    for i in range(bs):
        H_batched[i, :, :] = rayleigh_channel_MU(M, K)


    # precode
    W_batched = np.zeros((bs, M, K), dtype=np.complex64)
    for i in range(bs):
        if K == 1:
            W_batched[i, :, :] = MRT_precoding(H_batched[i, :, :], Pt)
        else:
            W_batched[i, :, :] = ZF_precoding(H_batched[i, :, :], Pt)

    # generate symbols
    nrdata = 100
    S = symbols_MU(K, nrdata=nrdata)
    S_batched = S[np.newaxis, :]

    # precode
    x_batched = np.zeros((bs, M, nrdata), dtype=np.complex64)
    for i in range(bs):
        x_batched[i, :, :] = W_batched[i, :, :] @ S

    for i in range(1, 8):
        # load DAC params
        beta = np.load(os.path.join(quant_params_path, f'{i}bits_nmse.npy'))
        thresholds = np.load(os.path.join(quant_params_path, f'{i}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{i}bits_outputlevels.npy'))

        # quantize numpy (with AGC )
        # compute alpha to scale DAC input per antenna (AGC - automatic gain control)
        xq_batched = np.zeros_like(x_batched)
        for b in range(bs):
            alpha_m = np.zeros(M)
            for m in range(M):
                Wm = W_batched[b, m, :]
                alpha_m[m] = Wm.T @ Wm.conj()
            #print(f'alpha m numpy: {alpha_m}')

            # scale DAC input
            x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x_batched[b, :, :]
            #print(f' x_scaled np: {x_scaled}')
            #print(f' var x scaled np: {np.var(x_scaled, axis=-1)}')
            xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

            # rescale the output of the DAC
            xq_batched[b, :, :] = np.sqrt(alpha_m)[:, np.newaxis] * xq

        # precode tf
        x_tf = W_batched @ S_batched

        # quantize tf
        x_q_tf = quantize_nonuniform_tf(tf.convert_to_tensor(x_tf, dtype=tf.complex64),
                                        tf.convert_to_tensor(W_batched, dtype=tf.complex64),
                                        tf.convert_to_tensor(thresholds, dtype=tf.float32),
                                        tf.convert_to_tensor(outputlevels, dtype=tf.float32)).numpy()

        assert np.sum(np.abs(xq_batched - x_q_tf)) <= 0.001

def test_tf_quant_equals_numpy():
    # basic sim params
    M, K = 16, 2
    Pt = M
    nr_snr_points = 20
    snr_tx = 20
    noise_var = Pt / (10 ** (snr_tx / 10))

    # path to quantizer parameters
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # generate channel
    H = rayleigh_channel_MU(M, K)
    H_batched = H[np.newaxis, :]

    # precode
    if K == 1:
        W = MRT_precoding(H, Pt)
    else:
        W = ZF_precoding(H, Pt)
    W_batched = W[np.newaxis, :]

    # generate symbols
    nrdata = 100
    S = symbols_MU(K, nrdata=nrdata)
    S_batched = S[np.newaxis, :]

    # precode
    x = W @ S

    for i in range(1, 8):
        # load DAC params
        beta = np.load(os.path.join(quant_params_path, f'{i}bits_nmse.npy'))
        thresholds = np.load(os.path.join(quant_params_path, f'{i}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{i}bits_outputlevels.npy'))

        # quantize numpy (with AGC )
        # compute alpha to scale DAC input per antenna (AGC - automatic gain control)
        alpha_m = np.zeros(M)
        for m in range(M):
            Wm = W[m, :]
            alpha_m[m] = Wm.T @ Wm.conj()
        #print(f'alpha m numpy: {alpha_m}')

        # scale DAC input
        x_scaled = (1 / np.sqrt(alpha_m))[:, np.newaxis] * x
        #print(f' x_scaled np: {x_scaled}')
        #print(f' var x scaled np: {np.var(x_scaled, axis=-1)}')
        xq = quantize_nonuniform(x_scaled, thresholds, outputlevels)

        # rescale the output of the DAC
        xq = np.sqrt(alpha_m)[:, np.newaxis] * xq

        # precode tf
        x_tf = W_batched @ S_batched

        # quantize tf
        x_q_tf = quantize_nonuniform_tf(tf.convert_to_tensor(x_tf, dtype=tf.complex64),
                                        tf.convert_to_tensor(W_batched, dtype=tf.complex64),
                                        tf.convert_to_tensor(thresholds, dtype=tf.float32),
                                        tf.convert_to_tensor(outputlevels, dtype=tf.float32)).numpy()

        print(f'sum error: {np.sum(np.abs(xq-x_q_tf))}')

        assert np.sum(np.abs(xq - x_q_tf)) <= 0.001

def test_loss_tf_equals_numpy():
    # basic sim params
    M, K = 16, 2
    Pt = M
    nr_snr_points = 20
    snr_tx = 20
    noise_var = Pt / (10 ** (snr_tx / 10))

    # path to quantizer parameters
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # generate channel
    H = rayleigh_channel_MU(M, K)
    H_batched = H[np.newaxis, :]

    # precode
    if K == 1:
        W = MRT_precoding(H, Pt)
    else:
        W = ZF_precoding(H, Pt)
    W_batched = W[np.newaxis, :]

    # convert complex numbers to real numbers bs x M x K => bs x M x K x 2
    H_batched = C2R(H_batched)
    W_batched = C2R(W_batched)

    for i in range(1, 8):
        # load DAC params
        beta = np.load(os.path.join(quant_params_path, f'{i}bits_nmse.npy'))
        alpha = 1 - beta

        # construct diag_alpha and diag_beta (with loaded values from file)
        diag_beta = np.identity(M) * beta
        diag_alpha = np.identity(M) - diag_beta

        # construct Rqq based on the NMSE of the DACs (assumes uncorrelated distortion!)
        Rqq = diag_alpha @ diag_beta @ np.diag(np.diag(W @ W.conj().T))

        # compute Rsum analytically (assumes uncorrelated distortion)
        Rsum_vectorized = rsum_analytical_vectorized(H, W, diag_alpha, Rqq, np.array([noise_var]))

        # compute Rsum using tf loss function
        loss = quant_loss_uncorrelated_Rqq(alpha, noise_var)
        Rsum_tf = -loss(tf.convert_to_tensor(H_batched, dtype=tf.float32),
                         tf.convert_to_tensor(W_batched, dtype=tf.float32)).numpy()

        assert Rsum_tf - Rsum_vectorized <= 0.001


