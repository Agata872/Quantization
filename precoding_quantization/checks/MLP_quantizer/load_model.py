import keras.losses
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.quantization import quantize_nonuniform
from utils.utils import create_folder
import tensorflow as tf
from tensorflow.keras import Sequential, layers


if __name__ == '__main__':
    path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\checks\MLP_quantizer\4_hidden_layers_16_neurons'
    model = keras.models.load_model(os.path.join(path, 'model.h5'))
    print(model.summary())

    # quantizer params path
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # load quantizer params
    bits = 3
    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
    beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))

    # construct data
    nr_data = 100000
    alpha = 5 #3
    x_in_re = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in_im = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in = x_in_re + 1j* x_in_im

    # quantize
    y = quantize_nonuniform(x_in, thresholds, outputlevels)

    # quantize using nn
    y_nn_re = np.squeeze(model(x_in_re[:, np.newaxis]))
    y_nn_im = np.squeeze(model(x_in_im[:, np.newaxis]))
    y_nn = y_nn_re + 1j * y_nn_im

    plt.plot(x_in_re, y_nn_re, label='nn')
    plt.plot(x_in_re, np.real(y), label='DAC')
    plt.legend()
    plt.show()
    print('est')


    #test with real data
    # construct signal to quantize CN(0, 2*varx)
    nr_data = 1000
    stdev = np.sqrt(varx)
    x = np.random.normal(0, stdev, nr_data) + 1j * np.random.normal(0, stdev, nr_data)
    y = quantize_nonuniform(x, thresholds, outputlevels)
    # quantize using nn
    y_nn_re = np.squeeze(model(np.real(x)[:, np.newaxis]))
    y_nn_im = np.squeeze(model(np.imag(x)[:, np.newaxis]))
    y_nn = y_nn_re + 1j * y_nn_im

    plt.scatter(np.real(x), np.real(y), label='DAC')
    plt.scatter(np.real(x), np.real(y_nn), label='NN')
    plt.legend()
    plt.show()