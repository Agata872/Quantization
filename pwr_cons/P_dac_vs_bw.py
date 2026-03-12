import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # DAC parameters
    Vdd = 3
    Io = 10 * 10**(-6)
    Cp = 1 * 10**(-12)

    # System parameters
    BWs = np.arange(1, 100, 10, dtype=np.float32) * 10**6# MHz#mmWave #100 * 10**(6)
    print(f'{BWs=}')
    oversampling_factor = 4

    M = 1#32
    bits = [16, 24]


    for b in bits:
        fs = oversampling_factor * 2 * BWs
        print(f"{fs=}")

        # pwr of 1 DAC
        static = 0.5 * Vdd * Io * (2**b - 1)
        dynamic = b * Cp * (fs / 2) * (Vdd**2)
        print(f'{static=}')
        print(f'{dynamic=}')
        P_dac = 0.5 * Vdd * Io * (2**b - 1) + b * Cp * (fs / 2) * (Vdd**2)

        # pwr of all DACs
        P_all_dacs = 2 * M * P_dac # times two cause I and Q


        # # log
        # print(f"{b=} - {P_dac=} - {P_all_dacs=}")
        # print(f"{static*2*M=} - {dynamic * 2 * M=}")
        plt.plot(BWs/(10**6), P_all_dacs, label=f'b={b}')

    plt.legend()
    plt.xlabel('BW [MHz]')
    plt.ylabel("P DACS [W]")
    plt.show()


    #
    # # using FoM_w
    # print(f'------------------------- FOM based -------------------------')
    # fom = 34.4 * 10**(-15)
    # for b in bits:
    #     p_dac_fom = fom * 2**b * fs
    #     p_dacs_fom = 2 * M * p_dac_fom
    #
    #     # log
    #     print(f"{b=} - {p_dac_fom=} - {p_dacs_fom=}")
