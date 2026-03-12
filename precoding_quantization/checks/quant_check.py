import numpy as np
import matplotlib.pyplot as plt
import os
from utils.quantization import quantize_nonuniform

if __name__ == '__main__':
    """ Check quantization for one quantizer analytical exprssion vs numerical """

    # quantizer params path
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # load quantizer params
    bits = 1
    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
    beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))

    # construct signal to quantize CN(0, 2*varx)
    nr_data = 1000
    stdev = np.sqrt(varx)
    x = np.random.normal(0, stdev, nr_data) + 1j * np.random.normal(0, stdev, nr_data)

    # quantize
    y = quantize_nonuniform(x, thresholds, outputlevels)
    mse = np.mean(np.abs(x - y)**2)
    sigvar = np.mean(np.abs(x)**2)
    nmse = mse / sigvar

    print(f'done')
