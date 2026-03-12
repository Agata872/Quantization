import numpy as np


if __name__ == '__main__':
     a = np.array(32.321)
     b = np.array(2.234)

     aq = a.astype('int8')
     bq = b.astype('int8')

     print(f'{a=}, {aq=}, {b=}, {bq=}')
     res = a * bq
     print(f'{res=}')