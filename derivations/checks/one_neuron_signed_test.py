import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def phi(x):
    # pdf N(0, 1)
    return (1/(np.sqrt(2*np.pi))) * np.exp(-0.5*x**2)

def B_analytical(a0, b0):
    # alpha = 0.5 sign(a0)
    alpha = 0.5 * np.abs(a0)/a0
    # Bussganggain
    B = (0.5-alpha) * a0 + 2*alpha*a0*qfunc(-b0/a0)
    return B

def exptf2_analytical(a0, b0, exptx, varx):
    # alpha = 0.5 sign(a0)
    alpha = 0.5 * np.abs(a0)/a0

    # compute E(f^2(x))
    term1 = (0.5-alpha) * (a0**2 * varx + 2*a0*b0*exptx + b0**2)
    term2 = 2 * alpha * (a0*b0*phi(-b0/a0) + (a0**2 + b0**2)*qfunc(-b0/a0))
    return term1 + term2

def error_var_analytical(a0, b0, exptx, varx):
    errorvar = exptf2_analytical(a0, b0, exptx, varx) - B_analytical(a0, b0)**2 * varx
    return errorvar

if __name__ == '__main__':
    a0 = np.random.uniform(low=-5, high=5)
    b0 = np.random.uniform(low=-5, high=5)
    print(f'a0: {a0}, b0: {b0}')
    mu_0 = 0
    sigma_0 = 1

    """------------numerical--------------"""
    # B numerically
    x = np.random.normal(loc=0, scale=1, size=10000)
    z = a0 * x + b0
    fx = np.maximum(np.zeros_like(z), z)
    varx = np.mean(x ** 2)
    B_num = np.mean(fx * x) / varx

    # expt(f^2(x)) numerically
    exptf2_num = np.mean(fx**2)

    # error variance numerically
    eta = fx - B_num*x
    var_eta_num = np.mean(eta**2)

    """------------analytical--------------"""
    # B analytical
    B_a = B_analytical(a0, b0)
    print(f'{B_a=} - {B_num=}')

    # expt(f^2(x)) analytical
    exptf2_a = exptf2_analytical(a0, b0, mu_0, sigma_0)
    print(f'{exptf2_a=} - {exptf2_num=}')

    # error var analytical
    var_eta_analytical = error_var_analytical(a0, b0, mu_0, sigma_0)
    print(f'{var_eta_analytical=} - {var_eta_num=}')
