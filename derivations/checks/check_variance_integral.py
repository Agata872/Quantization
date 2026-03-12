import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def funct(x):
    return x**2 * (1 / np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

if __name__ == '__main__':
    a0 = 1.568
    b0 = -1.568
    mu_0 = 0
    sigma_0 = 1

    # todo can this be found analytically?
    analytical_int = (1/(np.sqrt(2*np.pi))) * (-b0/a0) * np.exp(-0.5*(-b0/a0)**2) + qfunc(-b0/a0)
    print(f'{analytical_int=}')

    # using truncated normal distribution todo check if this will match the numerical one
    a, b = (-b0/a0), np.inf
    alpha = (a - mu_0) / (sigma_0)
    beta = (b - mu_0) / (sigma_0)
    pdf1 = -alpha * norm.pdf(alpha, loc=0, scale=1)
    cdf1 = 1
    cdf2 = norm.cdf(alpha, loc=0, scale=1)
    pdf2 = -norm.pdf(alpha, loc=0, scale=1)
    truncated_norm_formula = sigma_0**2 * (1 - (pdf1 / (cdf1 - cdf2)) - (pdf2 / (cdf1 - cdf2))**2)
    print(f'{truncated_norm_formula=}')

    # numerical integral
    num_int = integrate.quad(funct, -b0/a0, np.inf)
    print(f'{num_int=}')

    # compute bussgang gain according to obtained results and check against numerical ones
    B = a0 * qfunc(-b0/a0)
    print(f'{B=}')

    # B numerically
    x = np.random.normal(loc=0, scale=1, size=10000)
    z = a0*x+b0
    fx = np.maximum(np.zeros_like(z), z)
    varx = np.mean(x**2)
    print(f'{varx=}')
    B_num = np.mean(fx*x) / varx
    print(f'{B_num=}')

    # error variance
    var_eta_analytical = (((a0*b0)/(np.sqrt(2*np.pi))) * np.exp(-0.5*(-b0/a0)**2) + (a0**2 + b0**2) * qfunc(-b0/a0)
                          - B**2)
    print(f'{var_eta_analytical=}')

    # error variance numerically
    eta = fx - B_num*x
    var_eta_num = np.mean(eta**2)
    print(f'{var_eta_num=}')