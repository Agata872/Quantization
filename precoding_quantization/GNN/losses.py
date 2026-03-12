import numpy as np
import tensorflow as tf
from math import comb



def quant_numerical_MLP_DAC(alpha, noise_var, quant_model):
    def loss(H_batch_train, y_pred):
        """
        :param H_batch_train: batch of training channels (bs x M x K x 2)
        :param y_pred: output of the neuralnet (bs x M x K x 2)
        :return:
        """
        M, K = H_batch_train.shape[-3], H_batch_train.shape[-2]

        # reconstruct the channel
        Hre = H_batch_train[:, :, :, 0]
        Him = H_batch_train[:, :, :, 1]
        H = tf.complex(Hre, Him)

        #reconstruct the precoding matrix
        Wre = y_pred[:, :, :, 0]
        Wim = y_pred[:, :, :, 1]
        W = tf.complex(Wre, Wim) #bs x M x K

        # get diag_alpha #todo check this
        diag_alpha = alpha * tf.eye(M, batch_shape=[tf.shape(H_batch_train)[0]], dtype=tf.complex64)
        diag_beta = tf.eye(M, batch_shape=[tf.shape(H_batch_train)[0]], dtype=tf.complex64) - diag_alpha
        beta = 1 - alpha

        # compute usefull sig
        Htrans = tf.transpose(H, perm=(0, 2, 1))  # compute H^T
        Hconj = tf.math.conj(H)  # compute H^*
        H_diagalpha_W = Htrans @ diag_alpha @ W
        usefull_sig_pwr = tf.math.abs(tf.linalg.diag_part(H_diagalpha_W)) ** 2

        # compute inter user interference
        interference = tf.reduce_sum(tf.math.abs(H_diagalpha_W) ** 2, axis=2) - usefull_sig_pwr

        # compute distortion
        Rqq = compute_rqq_MLP_DAC(W, H, diag_alpha, quant_model) # compute cov(qq) numerically
        dist = tf.math.real(tf.linalg.diag_part(Htrans @ Rqq @ Hconj))

        # compute SINDR
        sindr = usefull_sig_pwr / (interference + dist + noise_var)

        # compute rate per user
        R = tf.math.log(1 + sindr) / tf.math.log(2.0)

        # compute sum rate
        Rsum = tf.reduce_sum(R, axis=1)
        return -Rsum
    return loss

def compute_rqq_MLP_DAC(W, H, diag_alpha, quant_model):
    """
    :param W: bs x M x K precoding matrix (output of NN)
    :param H: bs x M x K channel matrix
    :param diag_alpha:
    :param thresholds:
    :param outputlevels:
    :return:
    """
    M, K = H.shape[-2], H.shape[-1]

    # generate symbols
    nr = 500
    s = get_symbols(K, nr, p=1) #same over different batches? => yes

    # precode
    x = W @ s

    # quantize
    y = quantize_MLP_DAC(x, W, quant_model)

    # todo check which is best
    # method 1: compute Rqq = E(qq^H)
    # compute q = y - diag(alpha_m)x
    q = y - diag_alpha @ x
    covq = (1/nr) * q @ tf.transpose(q, conjugate=True, perm=(0, 2, 1))

    # # method 2: compute Rqq = E[xq xq^H] - |B|^2 E[xx^H]
    # # compute R_{yy} = E(y y^H)
    # ryy = (1/nr) * y @ tf.transpose(y, conjugate=True, perm=(0, 2, 1))
    # # compute cov(q) = E(qq^H) = E(yy^H) + diag(alpha_m) W W^H diag(alpha_m)^H
    # rqq = ryy - diag_alpha @ W @ tf.transpose(W, conjugate=True, perm=(0, 2, 1)) @ tf.transpose(diag_alpha, conjugate=True, perm=(0, 2, 1))
    # print(f'diff: {covq - rqq}')

    return covq

def quantize_MLP_DAC(x, W, quant_model):
    """
    :param x: precoded symbols bs x M x nr_data
    :param W: precoding matrix bs x M x K
    :param thresholds:
    :param outputlevels:
    :return:
    """

    # AGC normalize (automatic gain control)
    # var_x = tf.math.reduce_variance(x, axis=-1)
    # print(f'varx: {var_x}')
    alpha_m = tf.norm(W, ord='euclidean', axis=-1, keepdims=True)**2
    x_scaled = (1 / tf.math.sqrt(alpha_m)) * x
    # var_x_scaled = tf.math.reduce_variance(x_scaled, axis=-1)
    # print(f'varx schaled: {var_x_scaled}')

    # quantize
    # flatten x
    x_flat = tf.reshape(x_scaled, [-1])

    # quantize real and imag part using a pre trained MLP that mimics the quantizer
    # Freeze the weights of the pretrained model
    quant_model.trainable = False
    quantized_re_values = quant_model(tf.math.real(x_flat)[:, tf.newaxis])
    quantized_im_values = quant_model(tf.math.imag(x_flat)[:, tf.newaxis])

    # construct complex vector
    x_quant_flat = tf.complex(quantized_re_values, quantized_im_values)

    # reshape to original shape
    x_quant = tf.reshape(x_quant_flat, tf.shape(x))

    # denormalize
    x_quant_denorm = tf.math.sqrt(alpha_m) * x_quant

    # output (straight-through estimator for the gradients)
    x_quant_out = x + tf.stop_gradient(x_quant_denorm - x)
    """
    explanation: 
    - in the forward pass we have x + x_quant - x = x_quant as output
    - in the backwards pass we just have grad(x)
    """
    return x_quant_out

def quant_numerical(alpha, noise_var, thresholds, outputlevels):
    def loss(H_batch_train, y_pred):
        """
        :param H_batch_train: batch of training channels (bs x M x K x 2)
        :param y_pred: output of the neuralnet (bs x M x K x 2)
        :return:
        """
        M, K = H_batch_train.shape[-3], H_batch_train.shape[-2]

        # reconstruct the channel
        Hre = H_batch_train[:, :, :, 0]
        Him = H_batch_train[:, :, :, 1]
        H = tf.complex(Hre, Him)

        #reconstruct the precoding matrix
        Wre = y_pred[:, :, :, 0]
        Wim = y_pred[:, :, :, 1]
        W = tf.complex(Wre, Wim) #bs x M x K

        # get diag_alpha #todo check this
        diag_alpha = alpha * tf.eye(M, batch_shape=[tf.shape(H_batch_train)[0]], dtype=tf.complex64)
        diag_beta = tf.eye(M, batch_shape=[tf.shape(H_batch_train)[0]], dtype=tf.complex64) - diag_alpha
        beta = 1 - alpha

        # compute usefull sig
        Htrans = tf.transpose(H, perm=(0, 2, 1))  # compute H^T
        Hconj = tf.math.conj(H)  # compute H^*
        H_diagalpha_W = Htrans @ diag_alpha @ W
        usefull_sig_pwr = tf.math.abs(tf.linalg.diag_part(H_diagalpha_W)) ** 2

        # compute inter user interference
        interference = tf.reduce_sum(tf.math.abs(H_diagalpha_W) ** 2, axis=2) - usefull_sig_pwr

        # compute distortion
        Rqq = compute_rqq(W, H, diag_alpha, thresholds, outputlevels) # compute cov(qq) numerically
        dist = tf.math.real(tf.linalg.diag_part(Htrans @ Rqq @ Hconj))

        # compute SINDR
        sindr = usefull_sig_pwr / (interference + dist + noise_var)

        # compute rate per user
        R = tf.math.log(1 + sindr) / tf.math.log(2.0)

        # compute sum rate
        Rsum = tf.reduce_sum(R, axis=1)
        return -Rsum
    return loss

def compute_rqq(W, H, diag_alpha, thresholds, outputlevels):
    """
    :param W: bs x M x K precoding matrix (output of NN)
    :param H: bs x M x K channel matrix
    :param diag_alpha:
    :param thresholds:
    :param outputlevels:
    :return:
    """
    M, K = H.shape[-2], H.shape[-1]

    # generate symbols
    nr = 1000
    s = get_symbols(K, nr, p=1) #same over different batches? => yes

    # precode
    x = W @ s

    # quantize
    y = quantize_nonuniform_tf(x, W, thresholds, outputlevels)

    # todo check which is best
    # method 1: compute Rqq = E(qq^H)
    # compute q = y - diag(alpha_m)x
    q = y - diag_alpha @ x
    covq = (1/nr) * q @ tf.transpose(q, conjugate=True, perm=(0, 2, 1))


    # # method 2: compute Rqq = E[xq xq^H] - |B|^2 E[xx^H]
    # # compute R_{yy} = E(y y^H)
    # ryy = (1/nr) * y @ tf.transpose(y, conjugate=True, perm=(0, 2, 1))
    # # compute cov(q) = E(qq^H) = E(yy^H) + diag(alpha_m) W W^H diag(alpha_m)^H
    # rqq = ryy - diag_alpha @ W @ tf.transpose(W, conjugate=True, perm=(0, 2, 1)) @ tf.transpose(diag_alpha, conjugate=True, perm=(0, 2, 1))
    # print(f'diff: {covq - rqq}')

    return covq

def quantize_nonuniform_tf(x, W, thresholds, outputlevels):
    """
    :param x: precoded symbols bs x M x nr_data
    :param W: precoding matrix bs x M x K
    :param thresholds:
    :param outputlevels:
    :return:
    """
    # todo add straight through gradient see VQVAE keras

    # AGC normalize (automatic gain control)
    # var_x = tf.math.reduce_variance(x, axis=-1)
    # print(f'varx: {var_x}')
    alpha_m = tf.norm(W, ord='euclidean', axis=-1, keepdims=True)**2
    x_scaled = (1 / tf.math.sqrt(alpha_m)) * x
    # var_x_scaled = tf.math.reduce_variance(x_scaled, axis=-1)
    # print(f'varx schaled: {var_x_scaled}')

    # quantize
    # flatten x
    x_flat = tf.reshape(x_scaled, [-1])

    # find indices of where each entry of x falls in the thresholds
    indices_re = tf.searchsorted(thresholds, tf.math.real(x_flat)) #same functionality as np.digitize
    indices_im = tf.searchsorted(thresholds, tf.math.imag(x_flat))

    # map indices to output levels
    quantized_re_values = tf.gather(outputlevels, indices_re)
    quantized_im_values = tf.gather(outputlevels, indices_im)

    # construct complex vector
    x_quant_flat = tf.complex(quantized_re_values, quantized_im_values)

    # reshape to original shape
    x_quant = tf.reshape(x_quant_flat, tf.shape(x))

    # denormalize
    x_quant_denorm = tf.math.sqrt(alpha_m) * x_quant

    # output (straight-through estimator for the gradients)
    x_quant_out = x + tf.stop_gradient(x_quant_denorm - x)
    """
    explanation: 
    - in the forward pass we have x + x_quant - x = x_quant as output
    - in the backwards pass we just have grad(x)
    """
    return x_quant_out

def get_symbols(K, Ndata, p=1):
    """
    :param K: nr of users, generates (K x Ndata) outputs
    :param nr: number of symbols to generate
    :param p: signal variance
    :return: Ndata symbols sampled from a complex gaussian with variance p
    Note that this generates a variable which is drawn from a complex gaussian distribution with variance p
    which is equivalent to a + bj with a and b sampled from a gaussian distriobution with variance p/2.
    Here we first sample a and b from a gaussian with mean 0 and variance 1, by multiplying with sqrt(p)/sqrt(2)
    we obtain variance p/2 for both a and b, given that var(constant * X) = constant^2 var(X)
    """

    factor = tf.math.sqrt(tf.cast(p, tf.float32)) / tf.math.sqrt(tf.cast(2, tf.float32))
    s = tf.cast(factor, tf.complex64) * (tf.complex(tf.random.normal([K, Ndata]), tf.random.normal([K, Ndata])))
    return s

def quant_loss_uncorrelated_Rqq(alpha, noise_var):
    def loss(H_batch_train, y_pred):
        """
        :param H_batch_train: batch of training channels (bs x M x K x 2)
        :param y_pred: output of the neuralnet (bs x M x K x 2)
        :return:
        """
        M, K = H_batch_train.shape[-3], H_batch_train.shape[-2]

        # reconstruct the channel
        Hre = H_batch_train[:, :, :, 0]
        Him = H_batch_train[:, :, :, 1]
        H = tf.complex(Hre, Him)

        #reconstruct the precoding matrix
        Wre = y_pred[:, :, :, 0]
        Wim = y_pred[:, :, :, 1]
        W = tf.complex(Wre, Wim) #bs x M x K

        # get diag_alpha #todo check this
        diag_alpha = alpha * tf.eye(M, batch_shape=[tf.shape(H_batch_train)[0]], dtype=tf.complex64)
        diag_beta = tf.eye(M, batch_shape=[tf.shape(H_batch_train)[0]], dtype=tf.complex64) - diag_alpha
        beta = 1 - alpha

        # compute usefull sig
        Htrans = tf.transpose(H, perm=(0, 2, 1)) #compute H^T
        Hconj = tf.math.conj(H) #compute H^*
        H_diagalpha_W = Htrans @ diag_alpha @ W
        usefull_sig_pwr = tf.math.abs(tf.linalg.diag_part(H_diagalpha_W)) ** 2

        # compute inter user interference
        interference = tf.reduce_sum(tf.math.abs(H_diagalpha_W) ** 2, axis=2) - usefull_sig_pwr

        # compute distortion
        WH = tf.transpose(W, conjugate=True, perm=(0, 2, 1))
        # Rqq = diag_alpha @ diag_beta @ tf.linalg.diag(tf.linalg.diag_part(W @ WH))
        Rqq = alpha * beta * tf.linalg.diag(tf.linalg.diag_part(W @ WH)) #more efficient
        dist = tf.math.real(tf.linalg.diag_part(Htrans @ Rqq @ Hconj))

        # compute SINDR
        sindr = usefull_sig_pwr / (interference + dist + noise_var)

        # compute rate per user
        R = tf.math.log(1 + sindr) / tf.math.log(2.0)

        # compute sum rate
        Rsum = tf.reduce_sum(R, axis=1)

        return -Rsum

    return loss

def polynomial_loss(Bs, noisevar, Gw=True):
    """
    :param Bs:  polynomialcoefficients [B1, B3, B5, ..., B2N+1]
    :param noisevar:
    :param Gw: if False use the approximation G(w)= I
    :return:
    """
    def loss(H_batch_train, y_pred):
        """
        :param H_batch_train: batch of training channels (bs x M x K x 2)
        :param y_pred: output of the neuralnet (bs x M x K x 2)
        :return:
        """
        #(2N+1)th order polynomial
        N = Bs.shape[0] - 1#order of the polynomial [B1, B3, B5, ..., B2N+1]

        #reconstruct the channel
        Hre = H_batch_train[:, :, :, 0]
        Him = H_batch_train[:, :, :, 1]
        H = tf.complex(Hre, Him)

        #reconstruct the precoding matrix
        Wre = y_pred[:, :, :, 0]
        Wim = y_pred[:, :, :, 1]
        W = tf.complex(Wre, Wim) #bs x M x K

        #compute the Bussgang gain matrix G(W)
        Cx = W @ tf.transpose(W, conjugate=True, perm=(0, 2, 1))

        #compute H^T W
        Htrans = tf.transpose(H, perm=(0, 2, 1))

        #todo update GW to work with higher order polynomials
        if Gw:
            G = compute_Gw(Bs, Cx)
            HGW = Htrans @ G @ W

            """manual checks"""
            # Gcheck_third_order = tf.cast(tf.eye(tf.shape(W)[-2]), dtype=tf.complex64) \
            #     + tf.cast(2 * Bs[1], dtype=tf.complex64) \
            #     * tf.linalg.diag(tf.linalg.diag_part(Cx)) # I_m + 2 * b3 * I_m *diag(WWh)

            #
            # Gcheck_fifth_order = tf.cast(tf.eye(tf.shape(W)[-2]) * Bs[0], dtype=tf.complex64) \
            #     + 2 * Bs[1] * tf.cast(tf.eye(tf.shape(W)[-2]), dtype=tf.complex64) \
            #     @ tf.linalg.diag(tf.linalg.diag_part(Cx)) \
            #     + 6 * Bs[2] * tf.cast(tf.eye(tf.shape(W)[-2]), dtype=tf.complex64) \
            #     @ (tf.linalg.diag(tf.linalg.diag_part(Cx))**2)
            # #I_m + 2 * b3 * I_m *diag(WWh) + 6 * b5 * Im * diag(WWh)**2
            # print('G from function: ')
            # print(G)
            # print('G check: ')
            # print(Gcheck_fifth_order)
            # print('test this should be all zeros: ')
            # print(G-Gcheck_fifth_order)
            # print('debug')

        else: #simplify G = I
            HGW = Htrans @ W

        #get desired signal
        desiredsignal = tf.math.abs(tf.linalg.diag_part(HGW)) ** 2

        #get user interference: sum_k |HW|^2 - desiredsignal
        userInterference = tf.reduce_sum(tf.math.abs(HGW) ** 2, axis=2) - desiredsignal

        #compute the disotrtion power
        Hconj = tf.math.conj(H)
        Ce = compute_ce(N, Bs, Cx)#tf.cast(2 * tf.math.abs(b3)**2, dtype=tf.complex64) * Cx * tf.cast(tf.math.abs(Cx)**2, dtype=tf.complex64)
        distortion = tf.cast(tf.linalg.diag_part(Htrans @ Ce @ Hconj), dtype=tf.float32)

        '''manual checks'''
        # # Ce_check_thirdorder = tf.cast(2 * tf.math.abs(Bs[1])**2, dtype=tf.complex64) * Cx * tf.cast(tf.math.abs(Cx)**2, dtype=tf.complex64)
        #M = tf.shape(W)[1]
        # Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])
        # diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
        # L1 = tf.cast((2 / tf.math.sqrt(tf.constant(2.0))), dtype=tf.complex64)* Bs[1] * tf.cast(Im, dtype=tf.complex64) \
        #      + tf.cast((12/tf.math.sqrt(tf.constant(2.0))), dtype=tf.complex64) * Bs[2] * tf.cast(Im, dtype=tf.complex64) * diagCx
        # L2 = tf.cast((6 / tf.math.sqrt(tf.constant(3.0))), dtype=tf.complex64) * Bs[2] * tf.cast(Im, dtype=tf.complex64)
        # Ce_check_fifthorder = L1 @ (Cx * tf.cast(tf.math.abs(Cx)**2, dtype=tf.complex64)) @ tf.transpose(L1, perm=(0, 2, 1), conjugate=True) \
        #                       + L2 @ (Cx * tf.cast(tf.math.abs(Cx)**4, dtype=tf.complex64)) @ tf.transpose(L2, perm=(0, 2, 1), conjugate=True)
        # print('Ce function: ')
        # print(Ce)
        # print('Ce check: ')
        # print(Ce_check_fifthorder)
        # print('ce check this should be zero: ')
        # print(Ce - Ce_check_fifthorder)

        #compute sinr per user
        sinr = desiredsignal / (userInterference + distortion + noisevar)

        #rate per user
        R = tf.math.log(1 + sinr) / tf.math.log(2.0)

        #sumrate
        Rsum = tf.reduce_sum(R, axis=1)

        return -Rsum
    return loss

def compute_ce(N, Bs, Cx):
    Ce = 0
    for n in range(1, N+1):
        Ln = compute_Ln(n, N, Bs, Cx)
        Ce += Ln @ Cx * tf.cast(tf.abs(Cx)**(2*n), dtype=tf.complex64) @ tf.transpose(Ln, perm=(0, 2, 1), conjugate=True)
    return Ce

def compute_Ln(n, N, Bs, Cx):
    """
    :param n:
    :param N:
    :param Bs: array of polynomial coeffs [B1, B3, B5, ..., B2N+1]
    :param M:
    :return:
    """
    M = Cx.shape[1]
    diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
    Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])

    #compute Ln =
    Ln = 0
    for l in range(n, N+1):
        Ln += tf.cast(comb(l, n) * np.math.factorial(l+1), dtype=tf.complex64) * Bs[l] * \
              tf.cast(Im, dtype=tf.complex64) * diagCx**(l-n)
    Ln *= (1 / np.sqrt(n+1))
    return Ln

def compute_Gw(Bs, Cx):
    """
    :param Bs: poly coefs [b1, b3, b5, ..., b2N+1]
    :param Cx: input covariance matrix
    :return: bussgang gain matrix
    """
    M = Cx.shape[1]
    diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
    Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])

    #compute G(W) = B1 Im + (n+1)! B3 Im diag(Cx) + ... + (N+1)! B2N+1 diag(Cx)^N
    Gw = tf.cast(np.math.factorial(1), dtype=tf.complex64) * Bs[0] * tf.cast(Im, dtype=tf.complex64) #1e order term
    for n in range(1, Bs.shape[0]):#higher order terms
        Gw += tf.cast(np.math.factorial(n+1), dtype=tf.complex64) * Bs[n] * tf.cast(Im, dtype=tf.complex64) @ diagCx**n
    return Gw