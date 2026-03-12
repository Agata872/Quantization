import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os
from utils.utils import rayleigh_channel_MU, symbols_MU, los_channel_MU
from utils.quantization import quantize_uniform, quantize_nonuniform
from utils.precoding import ZF_precoding, MRT_precoding
from MIMO_sims.Rsum_all import Rsum_Bussgang_Rx, Rsum_Bussgang_generalized_wrt_s



if __name__ == "__main__":
    # basic sim params
    M, K = 8, 2
    Pt = M
    nr_snr_points = 20
    snr_tx = np.array([-30, -20, -10, 0.1, 10, 20, 30])

    # path to quantizer parameters
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # generate channels
    channel_realizations = 100
    H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
    np.random.seed(1)
    for i in range(channel_realizations):
        H[i, :, :] = rayleigh_channel_MU(M, K)
        #H[i, :, :] = np.ones((M, K)) # sanity check
        #H[i, :, :] = los_channel_MU(M, K)

    # test for b bits
    b = 3
    Ravg_bussgang_rx = Rsum_Bussgang_Rx(H, snr_tx, bits=b, quant='non-uniform',
                                        Pt=Pt, correlated_dist=True, automatic_gain_control=True,
                                        quant_params_path=quant_params_path)

    Ravg_generalized = Rsum_Bussgang_generalized_wrt_s(H, snr_tx, bits=b, quant='non-uniform',
                                        Pt=Pt, correlated_dist=True, automatic_gain_control=True,
                                        quant_params_path=quant_params_path)


    plt.plot(snr_tx, Ravg_bussgang_rx, label=f'{b} bits non-uniform, Bussgang Rx')
    plt.plot(snr_tx, Ravg_generalized, linestyle='dashed', label=f'{b} bits non-uniform, Bussgang General')

    plt.xlabel(r'SNR = $\frac{P_T }{ \sigma_{\nu}^2}$ [dB]')
    plt.ylabel(r'$R_{\mathrm{sum}}$ [bits/channel use]')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title(f'param N(0,{varx}) - numerical M: {M} K:{K} Pt: {Pt}')
    #plt.savefig("r_vs_bits.svg")
    plt.show()

