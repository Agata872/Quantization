import numpy as np

def ZF_precoding(H, Pt):
    Wzf = H.conj() @ np.linalg.inv(H.T @ H.conj())
    norm = np.sqrt(Pt) / np.linalg.norm(Wzf, ord='fro')
    Wzf *= norm

    # pwr_test = np.linalg.norm(Wzf, ord='fro')**2
    # print(pwr_test)
    return Wzf

def MRT_precoding(H, Pt):
    Wmrt = H.conj()
    norm = np.sqrt(Pt) / np.linalg.norm(Wmrt, ord='fro')
    Wmrt *= norm

    # pwr_test = np.linalg.norm(Wmrt, ord='fro')**2
    # print(pwr_test)
    return Wmrt
