import math
import random
import scipy
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np

def f(t):
    """
    :param t:
    :return: function studied: chose between uniform and gaussian functions

    """
    return math.exp(-t**2/2)/math.sqrt(2*math.pi)

def interval_MSE(x, t1, t2):
    """

    :param x:
    :param t1:
    :param t2:
    :return: computes MSE between 2 adjacent decision thresholds (on one segment)
    """
    return integrate.quad(lambda t: ((t - x)**2) * f(t), t1, t2)[0]

def MSE(t,x):
    """
    :param t:
    :param x:
    :return: computes mean squared error on R
    """
    s = interval_MSE(x[0], -float('Inf'), t[0]) + interval_MSE(x[-1], t[-1], float('Inf'))
    for i in range(1,len(x)-1):
        s = s + interval_MSE(x[i], t[i-1], t[i])
    return s


def centroid(t1, t2):
    """
    :param t1: lower boundary of the interval on which the centroid is calculated
    :param t2: upper boundary of the interval on which the centroid is calculated
    :return: centroid
    """
    if integrate.quad(f, t1, t2)[0] == 0 or t1 == t2:
        return 0
    else:
        return integrate.quad(lambda t:t*f(t), t1, t2)[0] / integrate.quad(f, t1, t2)[0]

def maxlloyd(t, y, eta=0.001):
    """
    :param t: vector containing initial thresholds
    :param y: vector containing initial outputlevels
    :param error_threshold: error at which to stop the algorithm => seems stupid, should be succesive error based
    :return:
    """
    e = MSE(t, y)
    error = [e]
    e_diff = 100
    nr_it = 0
    while e_diff > eta and nr_it < 10000:
        nr_it += 1

        # adjust thresholds
        for i in range(len(t)):
            t[i] = 0.5 * (y[i] + y[i+1])

        # adjust output levels
        y[0] = centroid(-float('Inf'), t[0])
        y[-1] = centroid(t[-1], float('Inf'))
        for i in range(1, len(y)-1):
            y[i] = centroid(t[i-1], t[i])

        # compute MSE
        e = MSE(t, y)
        error.append(e)

        # error reduction
        e_diff = error[nr_it-1] - error[nr_it]
        print(f'it: {nr_it}) MSE: {e}')

    return y, t, error

if __name__ == '__main__':
    bits = 4
    levels = 2**bits

    t = np.linspace(-10, 10, num=levels-1) #thresholds
    y = np.linspace(-10, 10, num=levels) #output levels
    y2, t2, error = maxlloyd(t, y, 0.0000000000)

    print(f'thresholds: {t2}')
    print(f'outputs: {y2}')
    print(f'MSE: {error[-1]}')
    print(f'SNQR: {1/error[-1]} abs - {10*np.log10(1/error[-1])} dB')

    plt.plot(error)
    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.show()

    # todo put it in a loop and do it for a number of different intializatons, keep the best one with the thresholds and output,
    # save them to npy data

    # todo verify for b > 5 with the approx formula found for b > 5
