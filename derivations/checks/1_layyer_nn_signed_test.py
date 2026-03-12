import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def phi(x):
    # pdf N(0, 1)
    return (1/(np.sqrt(2*np.pi))) * np.exp(-0.5*x**2)

def funct(x, a0, a1, b0, b1):
    return max(0, a0*x+b0) * max(0, a1*x+b1) * (1 / np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

def getc(a):
    n = len(a)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = max(a[i], a[j])
    return C

def getalpha(a0, b0):
    n = len(a0)
    alpha = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            index = np.argmax([-b0[i]/a0[i], -b0[j]/a0[j]])
            if index == 0:
                alpha[i, j] = 0.5*np.abs(a0[i])/a0[i]
            if index == 1:
                alpha[i, j] = 0.5*np.abs(a0[j])/a0[j]
    return alpha

def B_analytical(a0, a1, b0):
    u = np.zeros_like(a0)

    # # easy way to compute
    # for i in range(len(u)):
    #     if a0[i] >= 0:
    #         u[i] = a0[i] * qfunc(-b0[i]/a0[i])
    #     else:
    #         u[i] = a0[i] * (1 - qfunc(-b0[i]/a0[i]))
    # Bcheck = a1[:, np.newaxis].T @ u

    # using alpha
    alpha = 0.5 * np.abs(a0) / a0
    u = (0.5 - alpha) * a0 + 2*alpha*a0*qfunc(-b0/a0)
    B =  a1[:, np.newaxis].T @ u

    return B

def get_expz(a0, b0):
    """
    :param a0:
    :param b0:
    :return:
    todo: this is for x~N(0, 1) if x~N(mu, sigma) adjust code
    """
    n = len(a0)
    expz = np.zeros((n, n))
    checkmatrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ai = a0[i]
            aj = a0[j]
            bi = b0[i]
            bj = b0[j]
            checkmatrix[i, j] = integrate.quad(funct, -np.inf, np.inf, args=(ai, aj, bi, bj))[0]
            if ai > 0 and aj > 0:
                # case 1
                c = max(-bi / ai, -bj / aj)
                expz[i, j] = ai*aj* (c*phi(c) + qfunc(c)) + (ai*bj + aj*bi) * phi(c) + bi*bj*qfunc(c)
            elif ai < 0 and aj < 0:
                # case 2
                c = max(-bi / ai, -bj / aj)
                expz[i, j] = ai*aj* (1 - c*phi(c) - qfunc(c)) + (ai*bj + aj*bi) * (0 - phi(c)) + bi*bj*(1 - qfunc(c))
            elif ai < 0 and aj > 0:
                # case 3
                lowerbound = -bi / ai
                upperbound = -bj / aj
                expz[i, j] = (ai*aj * (lowerbound * phi(lowerbound) + qfunc(lowerbound) - upperbound*phi(upperbound) -
                      qfunc(upperbound)) + (ai*bj + aj*bi) * (0 + phi(lowerbound) - phi(upperbound)) +
                      bi*bj*(qfunc(lowerbound) - qfunc(upperbound)))
            elif ai > 0 and aj < 0:
                # case 4
                lowerbound = -bj / aj
                upperbound = -bi / ai
                expz[i, j] = (ai*aj * (lowerbound * phi(lowerbound) + qfunc(lowerbound) - upperbound*phi(upperbound) -
                      qfunc(upperbound)) + (ai*bj + aj*bi) * (0 + phi(lowerbound) - phi(upperbound)) +
                      bi*bj*(qfunc(lowerbound) - qfunc(upperbound)))
    print(f'test: {expz-checkmatrix}')
    return checkmatrix


def error_var_analytical(a0, a1, b0, b1, B, sigma_x, mu_x):
    # compute expt{Z}
    expz = get_expz(a0, a1)

    # compute expt{sigma(a_0x_m+b_0)}
    alpha = 0.5 * np.abs(a0)/a0
    expsigma = a0 * ((0.5-alpha)*mu_x - 2*alpha*phi(-b0/a0)) + b0 * ((0.5-alpha) - 2*alpha*qfunc(-b0/a0))

    # compute expt{f^2(x_m)}
    term1 = a1[:, np.newaxis].T @ expz @ a1[:, np.newaxis]
    term2 = 2*b1*a1[:, np.newaxis].T @ expsigma
    term3 = b1**2
    expf2x = term1 + term2 + term3
    print(f'analytical expt(f^2(x)){expf2x}')

    # compute error variance expt{eta^2} = expt{f^2(x_m)} - B^2 expt{x_m^2}
    var_eta = expf2x - B**2 * sigma_x

    #todo debug

    return var_eta



if __name__ == '__main__':
    mu_x = 0
    sigma_x = 1

    # generate random values for NN parameters
    d = 16
    a0 = np.random.uniform(low=-5, high=5, size=d) # dx1
    a1 = np.random.uniform(low=-5, high=5, size=d) # dx1
    b0 = np.random.uniform(low=-5, high=5,size=d) # dx1
    b1 = np.random.uniform(low=-5, high=5, size=1) # 1

    # generate random input N(0, 1)
    x = np.random.normal(loc=0, scale=1, size=10000)

    # run NN
    z = a0[:, np.newaxis] @ x[np.newaxis, :]
    z1 = z + b0[:, np.newaxis]
    fx = a1[:, np.newaxis].T @ np.maximum(np.zeros_like(z1), z1) + b1

    """------------numerical--------------"""
    # B numerically
    varx = np.mean(x**2)
    B_num = np.mean(fx*x) / varx

    # eta numerically
    eta = fx - B_num*x
    var_eta_num = np.mean(eta**2)

    """------------analytical--------------"""
    # B analytical
    B_a = B_analytical(a0, a1, b0)
    print(f'{B_a=} - {B_num=}')

    var_eta_a = error_var_analytical(a0, a1, b0, b1, B_a, sigma_x, mu_x)
    print(f'{var_eta_a=} - {var_eta_num=}')

    print(f'numerical expt(f^2(x)){np.mean(fx*fx)}')
