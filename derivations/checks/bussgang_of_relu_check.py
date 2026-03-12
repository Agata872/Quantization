import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

if __name__ == '__main__':
    a0 = np.random.uniform(low=-5, high=5)
    b0 = np.random.uniform(low=-5, high=5)
    print(f'a0: {a0}, b0: {b0}')
    print(f'a0**2: {a0**2}')

    # generate z
    x = np.random.normal(loc=0, scale=1, size=10000)
    z = a0 * x + b0
    print(f'numerical mean z: {np.mean(z)} - numerical variance z: {np.var(z)}')

    # take relu of z
    fz = np.maximum(np.zeros_like(z), z)

    # Bussgang numerical
    B_num = np.mean(fz * z) / np.mean(z**2)

    # Bussgang analytical
    B_anal = qfunc(-b0 / np.abs(a0))
    print(f'{B_num=} - {B_anal=}')
