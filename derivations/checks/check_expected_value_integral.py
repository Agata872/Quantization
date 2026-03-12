import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

def funct(x):
    return x * (1 / np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

if __name__ == '__main__':
    a0 = 0.22
    b0 = -0.123545
    mu_0 = 0
    sigma_0 = 1

    # analtycial solved solution of integral
    analytical_int = (1 / np.sqrt(2*np.pi)) * np.exp(-0.5 * (b0/a0)**2)
    print(f'{analytical_int=}')

    # using truncated normal distribution
    a, b = (-b0/a0), np.inf
    alpha = (a - mu_0) / (sigma_0)
    beta = (b - mu_0) / (sigma_0)
    truncated_norm_formula = mu_0 - sigma_0 * (-norm.pdf(alpha))
    print(f'{truncated_norm_formula=}')

    # numerical integral
    num_int = integrate.quad(funct, -b0/a0, np.inf)
    print(f'{num_int=}')