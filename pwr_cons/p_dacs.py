import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # DAC parameters
    Vdd = 3
    Io = 10 * 10**(-6)
    Cp = 1 * 10**(-12)

    # System parameters
    BW = 400 * 10**6 #mmWave #100 * 10**(6)
    oversampling_factor = 4
    fs = oversampling_factor * 2 * BW
    M = 32
    bits = np.arange(1, 11)

    print(f'{fs=}')
    for b in bits:
        # pwr of 1 DAC
        static = 0.5 * Vdd * Io * (2**b - 1)
        dynamic = b * Cp * (fs / 2) * (Vdd**2)
        P_dac = 0.5 * Vdd * Io * (2**b - 1) + b * Cp * (fs / 2) * (Vdd**2)

        # pwr of all DACs
        P_all_dacs = 2 * M * P_dac # times two cause I and Q

        # log
        print(f"{b=} - {P_dac=} - {P_all_dacs=}")
        print(f"{static*2*M=} - {dynamic * 2 * M=}")

    # using FoM_w
    print(f'------------------------- FOM based -------------------------')
    fom = 34.4 * 10**(-15)
    for b in bits:
        p_dac_fom = fom * 2**b * fs
        p_dacs_fom = 2 * M * p_dac_fom

        # log
        print(f"{b=} - {p_dac_fom=} - {p_dacs_fom=}")
