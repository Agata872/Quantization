import numpy as np
import matplotlib.pyplot as plt
import os
from utils.quantization import quantize_nonuniform
from utils.utils import create_folder

if __name__ == '__main__':
    """ Check quantization for one quantizer analytical exprssion vs numerical """

    # quantizer params path
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # load quantizer params
    bits = 4
    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
    beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))

    # construct data
    nr_data = 10000
    alpha = 4
    x_in_re = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in_im = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in = x_in_re + 1j* x_in_im

    # quantize
    y = quantize_nonuniform(x_in, thresholds, outputlevels)

    orders = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
    mses = np.zeros(len(orders))
    for j, order in enumerate(orders):
        print(f'---------------order: {order}--------------------')
        # fit polynomial
        index = int((order-1) / 2)
        X = np.zeros((nr_data, index), dtype=complex) # marix (nr_data x index) [x, x*|x|**2, x*|x|**4, ...]
        for i in range(0, index):
            print(f'i: {i}')
            l = 2 * (i+1) + 1
            print(f'coeficient: {l}')
            X[:, i] = x_in * np.abs(x_in)**(2*i)

        # LS fit of beta => bhat = (X^H X)^-1 x^H y | bhat = [b1, b3, ...]
        bhat = np.linalg.inv(X.conj().T @ X) @ X.conj().T @ y
        print(f'{bhat=}')

        # reconstrct the quantizer characteristic
        yhat = X @ bhat
        mses[j] = np.mean(np.abs(y-yhat)**2)
        print(f'MSE = {mses[j]}')

        if order == 21:
            # construct signal to quantize CN(0, 2*varx)
            nr_data = 1000
            stdev = np.sqrt(varx)
            x = np.random.normal(0, stdev, nr_data) + 1j * np.random.normal(0, stdev, nr_data)


            plt.plot(np.real(x_in), np.real(y), label='quantizer Re')
            plt.plot(np.real(x_in), np.real(yhat), label='poly Re')
            plt.scatter(x, np.zeros_like(x), label='input distribtion')
            plt.title(f'order: {order}')
            plt.legend()
            plt.show()

            plt.plot(np.imag(x_in), np.imag(y), label='quantizer Im')
            plt.plot(np.imag(x_in), np.imag(yhat), label='poly Im')
            plt.title(f'order: {order}')
            plt.legend()
            plt.show()

            # save coefficients
            results_folder = os.path.dirname(quant_params_path)
            results_folder = os.path.join(results_folder, 'poly_params')
            create_folder(results_folder)
            np.save(os.path.join(results_folder, f'betas_{bits}_bits.npy'), bhat)

            range_multiplier = 1.5
            x_in_re = np.linspace(-range_multiplier*alpha * np.sqrt(varx), range_multiplier*alpha * np.sqrt(varx), nr_data)
            x_in_im = np.linspace(-range_multiplier*alpha * np.sqrt(varx), range_multiplier*alpha * np.sqrt(varx), nr_data)
            x_in_extended = x_in_re + 1j * x_in_im
            index = int((order - 1) / 2)
            Xextended = np.zeros((nr_data, index), dtype=complex)  # marix (nr_data x index) [x, x*|x|**2, x*|x|**4, ...]
            for i in range(0, index):
                print(f'i: {i}')
                l = 2 * (i + 1) + 1
                print(f'coeficient: {l}')
                Xextended[:, i] = x_in_extended * np.abs(x_in_extended) ** (2 * i)
            yhat_extended = Xextended @ bhat
            plt.plot(np.real(x_in_extended), np.real(yhat_extended))
            plt.show()


    plt.stem(orders, 10*np.log10(mses))
    plt.xlabel(f'order')
    plt.ylabel(f'MSE [dB]')
    plt.show()
    print(f'{mses=}')
    print(f'best order = {orders[np.argmin(mses)]}')

    # # construct signal to quantize CN(0, 2*varx)
    # nr_data = 1000
    # stdev = np.sqrt(varx)
    # x = np.random.normal(0, stdev, nr_data) + 1j * np.random.normal(0, stdev, nr_data)
    #
    # # quantize
    # y = quantize_nonuniform(x, thresholds, outputlevels)
    # mse = np.mean(np.abs(x - y)**2)
    # sigvar = np.mean(np.abs(x)**2)
    # nmse = mse / sigvar
    #
    # print(f'done')