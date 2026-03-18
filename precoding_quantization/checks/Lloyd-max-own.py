import numpy as np
from scipy import integrate
from scipy.stats import norm
import math
from scipy.stats import norm
import os
import sys
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.utils import create_folder

def truncated_normal_mean(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    phi_alpha = norm.pdf(alpha)
    phi_beta = norm.pdf(beta)
    Phi_alpha = norm.cdf(alpha)
    Phi_beta = norm.cdf(beta)
    expected_value = mu + sigma * ((phi_alpha - phi_beta) / (Phi_beta - Phi_alpha))
    return expected_value

def integral_x_gaussian_pdf(x0, x1, sigma):
    """
    :param x0:
    :param x1:
    :param sigma:
    :return: compute mean value over interval x0 - x1,
    for a random variable X that is Gaussian distributed with variance sigma^2
    """
    value0 = (- sigma / (math.sqrt(2*math.pi))) * math.exp(-0.5 * (x0/sigma)**2)
    value1 = (- sigma / (math.sqrt(2*math.pi))) * math.exp(-0.5 * (x1/sigma)**2)
    return value1 - value0

def pdf(x):
    """
    :param x:
    :return: pdf of the input signal to the quantizer N(0,1) or N(0,0.5)
    """
    #sigma is a global variable
    mean = 0
    f = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x-mean)/sigma)**2)
    return f

def mse_per_partition(x0, x1, y):
    """
    :param x0: lower threshold
    :param x1: upper threshold
    :param y: output level
    :return: MSE distortion for this partition
    """
    se = lambda a: ((a-y)**2) * pdf(a)
    dist = integrate.quad(se, x0, x1)
    return dist[0]


def mse(x, y):
    N = len(y) #nr output levels

    #lower partition
    D = mse_per_partition(-float('Inf'), x[0], y[0])
    #upper partiton
    D += mse_per_partition(x[-1], float('Inf'), y[-1])

    for i in range(0, len(x)-1): #from threshold 1 upto N-1
        D += mse_per_partition(x[i], x[i+1], y[i+1])

    return D

def optimal_partitions(x, y):
    for i, xi in enumerate(x):
        x[i] = (y[i] + y[i+1]) / 2
    return x

def centroids(x, y, method=True):

    if method == 'numerical':
        int_f = lambda a: a * pdf(a)
        y[0] = integrate.quad(int_f, -float('Inf'), x[0])[0] / integrate.quad(pdf, -float('Inf'), x[0])[0]
        y[-1] = integrate.quad(int_f, x[-1], float('Inf'))[0] / integrate.quad(pdf, x[-1], float('Inf'))[0]
        for i in range(1, len(y)-1):
            y[i] = integrate.quad(int_f, x[i-1], x[i])[0] / integrate.quad(pdf, x[i-1], x[i])[0]

    elif method == 'builtin_numerical':
        y[0] = norm.expect(lambda a: a, loc=0, scale=sigma, lb=-float('Inf'), ub=x[0], conditional=True)
        y[-1] = norm.expect(lambda a: a, loc=0, scale=sigma, lb=x[-1], ub=float('Inf'), conditional=True)
        for i in range(1, len(y) - 1):
            y[i] = norm.expect(lambda a: a, loc=0, scale=sigma, lb=x[i - 1], ub=x[i], conditional=True)

    elif method == 'analytical':
        y[0] = truncated_normal_mean(0, sigma, -float('Inf'), x[0])
        y[-1] = truncated_normal_mean(0, sigma, x[-1], float('Inf'))
        for i in range(1, len(y) - 1):
            y[i] = truncated_normal_mean(0, sigma, x[i - 1], x[i])

    return y

def lloyd_max(x, y, varx=1, method='analytical', plot=True):
    D = [mse(x, y)]
    diff_d = 100
    epsilon = 10**(-10)

    idx = 0
    while diff_d > epsilon and idx < 1000:
        idx+=1
        x = optimal_partitions(x, y)
        y = centroids(x, y, method=method)
        D.append(mse(x, y))
        diff_d = D[idx-1] - D[idx]
        #print(f'{idx}) mse: {D[idx]} diff: {diff_d}')
    print(f'{idx} iterations ')
    if plot:
        plt.plot(D)
        plt.xlabel('iterations')
        plt.ylabel('distortion')
        plt.title(f'{np.log2(len(y))} bits')
        plt.show()
    nmse = mse(x, y) / varx
    return x, y, nmse, 1/nmse


if __name__ == '__main__':
    methods = [ 'numerical', 'builtin_numerical', 'analytical']

    #todo something wrong with analytical, to check

    for method in methods:
        input_distr = 'Gaussian'
        var = 0.5
        sigma = np.sqrt(var)
        basepath = os.path.join(PROJECT_ROOT, 'non-uniform-quant-params')
        results_path = os.path.join(os.path.join(basepath, f'{input_distr}_var_{var}'), f'{method}')
        create_folder(results_path)
        print(f'Writing quantizer parameters to: {results_path}')

        # loop over nr of bits
        for b in range(1, 16):
            print(f'bits: {b}')
            nr_inits = 10
            levels = 2 ** b  # nr output levels
            all_x = np.zeros((nr_inits, int(levels-1)))
            all_y = np.zeros((nr_inits, int(levels)))
            all_nmse = np.zeros(nr_inits)
            all_snqr = np.zeros(nr_inits)

            for it in range(nr_inits):
                print(f'init {it}')
                if it == 0:
                    # initial partitions and output levels uniformly spaced over 3sigma
                    x0 = np.linspace(-3*sigma, 3*sigma, int(levels-1))
                    y0 = np.linspace(-3*sigma, 3*sigma, int(levels))
                    plot = True #only plot MSE for first init as a check
                else:
                    # random distributed over -10sigma - 10sigma
                    x0 = np.sort(np.random.uniform(-10*sigma, 10*sigma, size=int(levels-1)))
                    y0 = np.sort(np.random.uniform(-10*sigma, 10*sigma, size=int(levels)))
                    if it == nr_inits-1:
                        plot = True
                    else: plot = False

                # run Lloyd-max algorithm
                x, y, nmse, snqr = lloyd_max(x0, y0, varx=sigma**2, method=method, plot=plot)
                all_x[it] = x
                all_y[it] = y
                all_nmse[it] = nmse
                all_snqr[it] = snqr

            # get best partitions and output levels
            idx_min = np.argmin(all_nmse)
            best_nmse = all_nmse[idx_min]
            best_snqr = all_snqr[idx_min]
            best_x = all_x[idx_min]
            best_y = all_y[idx_min]

            # todo save as np arrays
            np.save(os.path.join(results_path, f'{b}bits_nmse.npy'), best_nmse)
            np.save(os.path.join(results_path, f'{b}bits_snqr_abs.npy'), best_snqr)
            np.save(os.path.join(results_path, f'{b}bits_thresholds.npy'), best_x)
            np.save(os.path.join(results_path, f'{b}bits_outputlevels.npy'), best_y)


            print(f'------------ {b} bits - {levels} output levels ------------ ')
            print(f'{all_x=}')
            print(f'{all_y=}')
            print(f'{all_nmse=}')
            print(f'{all_snqr=}')

            print(f'-----------------best----------------')
            print(f'{best_nmse=}')
            print(f'{best_snqr=}')
            print(f'{best_x=}')
            print(f'{best_y=}')


