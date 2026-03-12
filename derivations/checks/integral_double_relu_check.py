import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def funct(x, a0, a1, b0, b1):
    return max(0, a0*x+b0) * max(0, a1*x+b1) * (1 / np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

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
    ai, aj = -1, 1
    bi, bj = 1, 1
    ai = np.random.uniform(low=-5, high=5)
    aj = np.random.uniform(low=-5, high=5)
    bi = np.random.uniform(low=0, high=5)
    bj = np.random.uniform(low=0, high=5)
    ai, aj = 1, -1
    bi, bj = -1, 1

    print(f'{ai=}, {aj=}, {bi=}, {bj=}')

    # numerical integral
    num_int = integrate.quad(funct, -np.inf, np.inf, args=(ai, aj, bi, bj))[0]
    print(f'{num_int=}')

    # analytical positive ai, aj
    c = max(-bi/ai, -bj/aj)
    analytical_pos = ai*aj* (c*phi(c) + qfunc(c)) + (ai*bj + aj*bi) * phi(c) + bi*bj*qfunc(c)
    print(f'{analytical_pos=}')

    # analytical negative ai, aj
    c = max(-bi/ai, -bj/aj)
    analytical_neg = ai*aj* (1 - c*phi(c) - qfunc(c)) + (ai*bj + aj*bi) * (0 - phi(c)) + bi*bj*(1 - qfunc(c))
    print(f'{analytical_neg=}')

    # # analytical pos ai, neg aj
    # lowerbound = -bi/ai
    # upperbound = -bj/aj
    # analytical_pos = ai*aj * (-bi/ai * phi(-bi/ai) + qfunc(-bi/ai) +bj/aj*phi(-bj/aj) - qfunc(-bj/aj)) + (ai*bj + aj*bi) * (0 + phi(-bi/ai) - phi(-bj/aj)) + bi*bj*(qfunc(-bi/ai) - qfunc(-bj/aj))
    # print(f'{analytical_pos=}')

    # analytical pos ai, neg aj
    lowerbound = -bi/ai
    upperbound = -bj/aj
    analytical_posai_negaj = (ai*aj * (lowerbound * phi(lowerbound) + qfunc(lowerbound) - upperbound*phi(upperbound) -
                      qfunc(upperbound)) + (ai*bj + aj*bi) * (0 + phi(lowerbound) - phi(upperbound)) +
                      bi*bj*(qfunc(lowerbound) - qfunc(upperbound)))
    print(f'{analytical_posai_negaj=}')

    # analytical pos ai, neg aj
    lowerbound = -bj/aj
    upperbound = -bi/ai
    analytical_negai_posaj = (ai*aj * (lowerbound * phi(lowerbound) + qfunc(lowerbound) - upperbound*phi(upperbound) -
                      qfunc(upperbound)) + (ai*bj + aj*bi) * (0 + phi(lowerbound) - phi(upperbound)) +
                      bi*bj*(qfunc(lowerbound) - qfunc(upperbound)))
    print(f'{analytical_negai_posaj=}')
