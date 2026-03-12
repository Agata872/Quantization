import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def phi(x):
    return (1/(np.sqrt(2*np.pi))) * np.exp(-0.5*x**2)


def construct_matrix(a):
    n = len(a)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = max(a[i], a[j])
    return C

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

    # numerical check
    varx = np.mean(x**2)
    B_num = np.mean(fx*x) / varx
    eta = fx - B_num*x
    var_eta_num = np.mean(eta**2)
    print(f'{B_num=}')
    print(f'{var_eta_num=}')

    # analytical computation
    B_analytical = a1[:, np.newaxis].T @ (a0 * qfunc((-b0)/a0))
    print(f'{B_analytical=}')

    # var eta
    C = construct_matrix(-b0/a0)
    term1 = (a0[:, np.newaxis] @ a0[:, np.newaxis].T * (C*phi(C) + qfunc(C)) +
             2*a0[:, np.newaxis]@b0[:, np.newaxis].T * phi(C) + b0[:, np.newaxis]@b0[:, np.newaxis].T * qfunc(C))
    term2 = a0 * phi(-b0/a0) + b0 * qfunc(-b0/a0)
    var_eta_analytical = (a1[:, np.newaxis].T @ (term1) @ a1[:, np.newaxis] +
                          2*b1*a1[:, np.newaxis].T @ (term2) + b1**2 - B_analytical**2 * sigma_x)
    print(f'{var_eta_analytical=}')

    # todo something wrong with the sign of the lowerbound on the integral,
    #  works well if all a's and b's are possitive


    print(f'done')