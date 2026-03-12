import numpy as np
import matplotlib.pyplot as plt

def quantize_uniform(x, b=2):
    # testing quantization
    L = 2 ** b  # nr of quantization labels
    optimal_steps = np.array([1.62114453e+00, 1.00819096e+00, 5.89546149e-01, 3.36032800e-01,
       1.88313891e-01, 1.04096628e-01, 5.68736647e-02, 3.07633948e-02,
       1.64991206e-02, 8.78548969e-03, 4.64984563e-03, 2.44841165e-03,
       1.28362082e-03, 6.70451805e-04, 3.49059387e-04, 1.86771379e-05])
    delta = optimal_steps[b-1]  # optimal step size
    i = np.arange(L)
    labels = delta * (i - (L - 1) / 2)  # todo after quantization, multiply with alpha to respect the avg pwr constraint
    i = np.arange(1, L)
    thresholds = delta * (i - (L / 2))  # thresholds

    # Find the indices where each entry of x falls in the thresholds
    indices_re = np.digitize(np.real(x), thresholds, right=False)
    indices_imag = np.digitize(np.imag(x), thresholds, right=False)

    # Map the indices to quantization labels
    quantized_re_values = labels[indices_re]
    quantized_im_values = labels[indices_imag]

    # Use the labels to construct the quantized output
    xquant = quantized_re_values + 1j * quantized_im_values

    # compute normalized mean squared quantization error as a check
    NMSQE_m = np.mean(np.abs(x - xquant)**2, axis=1) / np.mean(np.abs(x)**2, axis=1)
    # print(f'mean NMSQE: {np.mean(NMSQE_m)}')
    # print(f'{NMSQE_m=}')

    test = False
    if test:
        # check pwr constraint
        pwr_x = np.mean(np.linalg.norm(x, ord=2, axis=0)**2)
        pwr_x_quant = np.mean(np.linalg.norm(xquant, ord=2, axis=0)**2)
        print(f'{pwr_x=}')
        print(f'{pwr_x_quant=}')
        # todo adjust power level after quantization, check if they do this on the fly in the paper?

        #todo plot labels and thresholds and labels for check
        for label in labels:
            plt.axhline(y=label, color='r', linestyle='dashed')
        plt.axhline(y=label, color='r', linestyle='dashed', label='quantization labels')

        for threshold in thresholds:
            plt.axhline(y=threshold, color='black', linestyle='dotted')
        plt.axhline(y=threshold, color='black', linestyle='dotted', label='threshold')



        sample_size = 100
        sample_re_values = np.real(x).flatten()[0:sample_size]
        sample_im_values = np.imag(x).flatten()[0:sample_size]
        sample_quantized_re_values = quantized_re_values.flatten()[0:sample_size]
        sample_quantized_im_values = quantized_im_values.flatten()[0:sample_size]

        plt.scatter(np.arange(sample_size), sample_re_values, label='Re part', marker='+')
        plt.scatter(np.arange(sample_size, 2*sample_size), sample_im_values, label='Im part',  marker='+')
        plt.scatter(np.arange(sample_size), sample_quantized_re_values, label='Re part quantized')
        plt.scatter(np.arange(sample_size, 2*sample_size), sample_quantized_im_values, label='Im part quantized')
        plt.legend()
        plt.show()

    return xquant

def quantize_nonuniform(x, thresholds, outputlevels):
    """
    :param x: input to quant
    :param thresholds:
    :param outputlevels:
    :return: if thresholds[i-1] < x <= thresholds[i]: Q(x) = outputlevels[i]
    """

    # Find the indices where each entry of x falls in the thresholds
    indices_re = np.digitize(np.real(x), thresholds, right=True)
    indices_imag = np.digitize(np.imag(x), thresholds, right=True)

    # Map the indices to quantization labels
    quantized_re_values = outputlevels[indices_re]
    quantized_im_values = outputlevels[indices_imag]

    # Use the labels to construct the quantized output
    xquant = quantized_re_values + 1j * quantized_im_values

    # compute normalized mean squared quantization error as a check
    # NMSQE_m = np.mean(np.abs(x - xquant)**2, axis=1) / np.mean(np.abs(x)**2, axis=1)
    # print(f'mean NMSQE: {np.mean(NMSQE_m)}')
    # print(f'{NMSQE_m=}')
    # print(f'nonuniform')

    test = False
    if test:
        # check pwr constraint
        pwr_x = np.mean(np.linalg.norm(x, ord=2, axis=0)**2)
        pwr_x_quant = np.mean(np.linalg.norm(xquant, ord=2, axis=0)**2)
        print(f'{pwr_x=}')
        print(f'{pwr_x_quant=}')
        # todo adjust power level after quantization, check if they do this on the fly in the paper?

        #todo plot labels and thresholds and labels for check
        for label in outputlevels:
            plt.axhline(y=label, color='r', linestyle='dashed')
        plt.axhline(y=label, color='r', linestyle='dashed', label='quantization labels')

        for threshold in thresholds:
            plt.axhline(y=threshold, color='black', linestyle='dotted')
        plt.axhline(y=threshold, color='black', linestyle='dotted', label='threshold')



        sample_size = 100
        sample_re_values = np.real(x).flatten()[0:sample_size]
        sample_im_values = np.imag(x).flatten()[0:sample_size]
        sample_quantized_re_values = quantized_re_values.flatten()[0:sample_size]
        sample_quantized_im_values = quantized_im_values.flatten()[0:sample_size]

        plt.scatter(np.arange(sample_size), sample_re_values, label='Re part', marker='+')
        plt.scatter(np.arange(sample_size, 2*sample_size), sample_im_values, label='Im part',  marker='+')
        plt.scatter(np.arange(sample_size), sample_quantized_re_values, label='Re part quantized')
        plt.scatter(np.arange(sample_size, 2*sample_size), sample_quantized_im_values, label='Im part quantized')
        plt.legend()
        plt.show()

    return xquant