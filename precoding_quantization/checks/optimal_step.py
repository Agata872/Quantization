import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy import special
from scipy.optimize import fsolve
# for uniform quantizer

def q_function(x):
    return 0.5 - 0.5 * special.erf(x / np.sqrt(2))  # Q(x) = 0.5 - 0.5 erf(x/sqrt(2))

def myfunc(copt, b):

    a1 = 4 * copt * q_function(copt) - np.sqrt(8/np.pi) * np.exp(-(copt**2)/(2))
    a2 = np.sqrt(8/np.pi) * np.exp(-(copt**2)/(2)) - 8*copt * q_function(copt)
    a3 = (2/3) * copt + 4*copt*q_function(copt) - np.sqrt(2/np.pi) * copt**2 * np.exp(-(copt**2)/(2))
    return a1 * (2**b)**2 + a2 * (2**b) + a3


if __name__ == '__main__':
    # todo where is the dependence of the variance of the input signal? => variance one probably ok so if we have CN(0, Pt/M) then we are good. 

    # A1 (2^b)^2 + A2 2^b + A3 = 0
    nr_bits = np.arange(16) + 1
    copts = np.zeros_like(nr_bits, dtype=float)
    for i in range(len(nr_bits)):
        b = nr_bits[i]
        roots = fsolve(myfunc, 1, args=b)
        copts[i] = roots[0]

    plt.scatter(nr_bits, copts)
    plt.grid()
    plt.xlabel('nr of bits')
    plt.ylabel('optimal clip level')
    plt.show()
    print(f'{copts=}')


    optimal_steps = copts / (2**(nr_bits-1))
    print(f'{optimal_steps=}')

    plt.scatter(nr_bits, optimal_steps)
    plt.grid()
    plt.xlabel('nr of bits')
    plt.ylabel('optimal step size')
    plt.show()
