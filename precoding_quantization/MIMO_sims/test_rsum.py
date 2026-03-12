from MIMO_sims.Rsum_all import Rsum_Bussgang_Rx, Rsum_Bussgang_DAC, Rsum_analytical_wrapper, rsum_analytical_loopy, rsum_analytical_vectorized
import numpy as np
import os
from utils.utils import rayleigh_channel_MU, los_channel_MU
from utils.precoding import ZF_precoding, MRT_precoding

"""
unit tests for different ways to compute the sum rate
"""

# general parameters for testing (can be changed to test different things)
M, K = 64, 8
channel_model = 'iid' #'iid' # 'los' #'all_ones'

def test_Rsum_analytical_equals_Rsum_dac_uncorrelated_q():
   """
   check if Rsum computed analytically (which assumes that the distortion is uncorrelated) is
   equal to Rsum computed using the bussgang decomposition after each DAC, and only considering the diagonal
   elements of the covariance matrix of the distortion Rqq
   """

   # basic sim params
   #M, K = 64, 1
   print(f'K= {K}')
   Pt = M
   nr_snr_points = 20
   snr_tx = np.linspace(-30, 35, nr_snr_points)

   # path to quantizer parameters
   varx = 0.5
   quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
   quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

   # generate channels
   channel_realizations = 10
   H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
   np.random.seed(1)
   for i in range(channel_realizations):
      if channel_model == 'iid':
         H[i, :, :] = rayleigh_channel_MU(M, K)
      elif channel_model == 'los':
         H[i, :, :] = los_channel_MU(M, K)
      elif channel_model == 'all_ones':
         H[i, :, :] = np.ones((M, K)) # sanity check



   for i in range(1, 8):
      # compute Rsum using Bussgang after DACs and only consider the diag of Rqq
      print(f'Rsum_Bussgang_DAC')
      Ravg_bussgang_dac_uncorllated_dist = Rsum_Bussgang_DAC(H, snr_tx, bits=i, quant='non-uniform',
                                                             Pt=Pt, correlated_dist=False, automatic_gain_control=True,
                                                             quant_params_path=quant_params_path)

      # compute Rsum analytically (assumes uncorrelated distortion)
      print(f'Rsum_analytical')
      Ravg_fully_analytical = Rsum_analytical_wrapper(H, snr_tx, bits=i, quant='non-uniform',
                                                      Pt=Pt, correlated_dist=False, automatic_gain_control=True,
                                                      quant_params_path=quant_params_path)

      max_diff = max(np.abs(Ravg_bussgang_dac_uncorllated_dist - Ravg_fully_analytical))
      print(f'max diff = {max_diff}')

      assert max_diff <= 0.1

def test_Rsum_dac_equals_Rsum_rx():
   """
   check if Rsum computed  Rsum computed using the bussgang decomposition after each DAC is
   equal to Rsum computed using the bussgang at RX
   """

   # basic sim params
   #M, K = 64, 1
   Pt = M
   nr_snr_points = 20
   snr_tx = np.linspace(-30, 35, nr_snr_points)

   # path to quantizer parameters
   varx = 0.5
   quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
   quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

   # generate channels
   channel_realizations = 10
   H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
   np.random.seed(1)
   for i in range(channel_realizations):
      if channel_model == 'iid':
         H[i, :, :] = rayleigh_channel_MU(M, K)
      elif channel_model == 'los':
         H[i, :, :] = los_channel_MU(M, K)
      elif channel_model == 'all_ones':
         H[i, :, :] = np.ones((M, K)) # sanity check

   for i in range(1, 8):
      # compute Rsum using Bussgang after DACs and only consider the diag of Rqq
      print(f'Rsum_Bussgang_DAC')
      Ravg_bussgang_dac = Rsum_Bussgang_DAC(H, snr_tx, bits=i, quant='non-uniform',
                                                             Pt=Pt, correlated_dist=True, automatic_gain_control=True,
                                                             quant_params_path=quant_params_path)

      # compute Rsum analytically (assumes uncorrelated distortion)
      print(f'Rsum_Bussgang_RX')
      Ravg_bussgang_rx = Rsum_Bussgang_Rx(H, snr_tx, bits=i, quant='non-uniform',
                                 Pt=Pt, correlated_dist=True, automatic_gain_control=True,
                                          quant_params_path=quant_params_path)

      max_diff = max(np.abs(Ravg_bussgang_dac - Ravg_bussgang_rx))
      print(f'max diff = {max_diff}')

      assert max_diff <= 0.5

def test_Rsum_analytical_largerthen_Rsum_numerical():
   """
   check if Rsum computed analytically (which assumes that the distortion is uncorrelated) is
   larger then to Rsum computed numerically, using the bussgang decomposition at RX
   (which takes into account correlation in the distortion, and should thus be lower)

   """

   # basic sim params
   #M, K = 64, 1
   Pt = M
   nr_snr_points = 20
   snr_tx = np.linspace(-20, 35, nr_snr_points)

   # path to quantizer parameters
   varx = 0.5
   quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
   quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

   # generate channels
   channel_realizations = 10
   H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
   np.random.seed(1)
   for i in range(channel_realizations):
      if channel_model == 'iid':
         H[i, :, :] = rayleigh_channel_MU(M, K)
      elif channel_model == 'los':
         H[i, :, :] = los_channel_MU(M, K)
      elif channel_model == 'all_ones':
         H[i, :, :] = np.ones((M, K)) # sanity check


   for i in range(1, 8):
      # compute Rsum using Bussgang after DACs and only consider the diag of Rqq
      print(f'Rsum_Bussgang_DAC')
      Ravg_fully_numerical = Rsum_Bussgang_Rx(H, snr_tx, bits=i, quant='non-uniform',
                                                             Pt=Pt, correlated_dist=True, automatic_gain_control=True,
                                                             quant_params_path=quant_params_path)

      # compute Rsum analytically (assumes uncorrelated distortion)
      print(f'Rsum_analytical')
      Ravg_fully_analytical = Rsum_analytical_wrapper(H, snr_tx, bits=i, quant='non-uniform',
                                                      Pt=Pt, correlated_dist=False, automatic_gain_control=True,
                                                      quant_params_path=quant_params_path)

      min_diff = min(Ravg_fully_analytical - Ravg_fully_numerical)
      print(f'min diff = {min_diff}')
      print(f'diff= {Ravg_fully_analytical-Ravg_fully_numerical}')

      assert min_diff >= -0.1

def test_Rsum_analytical_equals_Rsum_numerical_for_many_users():
   """
   check if Rsum computed analytically (which assumes that the distortion is uncorrelated) is
   equal to Rsum computed numerically
   when many users are present and iid channels are assumed as this should lead to uncorllelated distortion
   """

   # basic sim params
   M, K = 64, 40
   Pt = M
   nr_snr_points = 20
   snr_tx = np.linspace(-30, 35, nr_snr_points)

   # path to quantizer parameters
   varx = 0.5
   quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
   quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

   # generate channels
   channel_realizations = 1
   H = np.zeros((channel_realizations, M, K), dtype=np.csingle)
   np.random.seed(1)
   for i in range(channel_realizations):
      H[i, :, :] = rayleigh_channel_MU(M, K)



   for i in range(1, 8):
      # compute Rsum using Bussgang after DACs and only consider the diag of Rqq
      print(f'Rsum_Bussgang_DAC')
      Ravg_fully_numerical = Rsum_Bussgang_Rx(H, snr_tx, bits=i, quant='non-uniform',
                                                             Pt=Pt, correlated_dist=True, automatic_gain_control=True,
                                                             quant_params_path=quant_params_path)

      # compute Rsum analytically (assumes uncorrelated distortion)
      print(f'Rsum_analytical')
      Ravg_fully_analytical = Rsum_analytical_wrapper(H, snr_tx, bits=i, quant='non-uniform',
                                                      Pt=Pt, correlated_dist=False, automatic_gain_control=True,
                                                      quant_params_path=quant_params_path)

      max_diff = max(np.abs(Ravg_fully_analytical - Ravg_fully_numerical))
      print(f'max diff = {max_diff}')

      assert max_diff <= 1

def test_Rsum_loopy_equals_Rsum_vectorized():

   # basic sim params
   #M, K = 64, 1
   Pt = M
   nr_snr_points = 20
   snr_tx = np.linspace(-20, 35, nr_snr_points)
   noise_vars = Pt / (10 ** (snr_tx / 10))

   # path to quantizer parameters
   varx = 0.5
   quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
   quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

   # generate channel
   H = rayleigh_channel_MU(M, K)

   # precode
   if K == 1:
      W = MRT_precoding(H, Pt)
   else:
      W = ZF_precoding(H, Pt)

   for i in range(1, 8):
      # load DAC params
      beta = np.load(os.path.join(quant_params_path, f'{i}bits_nmse.npy'))

      # construct diag_alpha and diag_beta (with loaded values from file)
      diag_beta = np.identity(M) * beta
      diag_alpha = np.identity(M) - diag_beta

      # construct Rqq based on the NMSE of the DACs (assumes uncorrelated distortion!)
      Rqq = diag_alpha @ diag_beta @ np.diag(np.diag(W @ W.conj().T))

      # compute Rsum using Bussgang after DACs and only consider the diag of Rqq
      R_sum_loopy = rsum_analytical_loopy(H, W, diag_alpha, Rqq, noise_vars)

      # compute Rsum analytically (assumes uncorrelated distortion)
      Rsum_vectorized = rsum_analytical_vectorized(H, W, diag_alpha, Rqq, noise_vars)

      diff = R_sum_loopy - Rsum_vectorized

      assert np.sum(diff) <= 0.0001