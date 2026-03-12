import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

def pwr_fwd_pass(M, K, b, dl, Nh):
    adds = dl * (11*M*K - 8*K - 8*M) + dl**2 * (12*M*K + 9*M + 9*K) + 2**b * (6*M*K*dl - 4*M +2*M*dl) + 2*M*(2**(2*b))
    muls = dl*(6*M*K + 7*K + 7*M) + dl**2 * (12*M*K + 9*M + 9*K) + 2**b * (6*M*K*dl + 2*M + 2*M*dl) + 4*M*(2**(2*b))
    flops = adds + muls

    # check
    adds2, muls2, flops2 = pwr_fwd_pass_with_nh(M, K, b, dl, Nh)

    return adds, muls, flops

def pwr_fwd_pass_with_nh(M, K, b, dh, Nh): #todo
    adds_in = 5*M*K*dh + K*(M-1)*dh + M*(K-1)*dh + M*dh + M*dh**2 + K*dh + K*dh**2
    adds_hidden = 3*M*K*dh**2 - M*K*dh + K*(M-1)*dh + M*(K-1)*dh + 2*M*dh**2 - M*dh + 2*K*dh**2 - K*dh
    adds_out = 6*M*K*dh*2**b - 2*M*K*2**b + 2*M*(K-1)*2**b + 4*M*2**(2*b) + 2*M*dh*2**b - 2*M*2**b

    muls_in = 6*M*K*dh + K*dh + M*dh + 2*M*dh + M*dh**2 + 2*K*dh + K*dh**2
    muls_hidden = 3*M*K*dh**2 + K*dh + M*dh + 2*M*dh**2 + 2*K*dh**2
    muls_out = 6*M*K*dh*2**b + 2*M*2**b + 4*M*2**(2*b) + 2*M*dh*2**b

    adds = adds_in + Nh * adds_hidden + adds_out
    muls = muls_in + Nh * muls_hidden + muls_out

    flops = adds + muls
    return adds, muls, flops

def flops_per_second(M, K, b, dl, Nh, Rs):
    """
    M : nr antennas
    K: nr users
    b: nr bits
    dl: nr hidden features
    Nh: nr hidden layers
    Rs: symbol rate
    """

    # complexity of 1 fwd pass
    adds, muls, flops = pwr_fwd_pass_with_nh(M, K, b, dl, Nh)#pwr_fwd_pass(M, K, b, dl, Nh)

    # complexity of all fwd passes
    return Rs * flops

def flops_per_second_ZF(M, K, Rs, Tc):

    # precoder is computed once per coherence time!! todo account for this, now just once per second!
    adds = 8*M*K**2 + 2*K**3
    muls = 8*M*K**2 + 2*K**3 + 6*K**2
    flops = adds + muls

    # flops per second
    nr_prec_matrices = int(1 / Tc)
    flops_per_sec_prec_matrix = nr_prec_matrices * flops

    # actual precoding: matmul between W and s (for 1 symbol)
    adds_Ws = 2*K*M + 2*(K-1)*M
    muls_Ws = 4*K*M
    flops_Ws = adds_Ws + muls_Ws

    # times number of symbols per second
    flops_Ws_per_second = flops_Ws * Rs

    return flops_per_sec_prec_matrix + flops_Ws_per_second



def power_gnn(flops_per_sec):
    # todo check if the accelorator meets the speed requirement
    """
    options:
    1) d: RDCIM: 2.82 TOPS/second - 66.3 TOPS/W - INT 4
    2) E: Metis AIPU: 52.4 TOPS/second - 15 TOPS/W - INT 8
    3) F: Esperanto: 139 TOPS/second - 6.95 GOPS/W - INT 8
    4) DIANA (analog): 18.1 TOPS/second - 176 TOPs/W - analog
    from: https://nicsefc.ee.tsinghua.edu.cn/project.html
    5) INT8) on Kirin 9000: 36 TOPS/s - 36 TOPS/W - int8
    6) A 40-nm 646.6TOPS/W Sparsity-Scaling DNN Processor for On-Device Training
    39.8 TOPS/s - 646.6 TOP/s/W - FP 8
    """

    # compute tops
    tops = flops_per_sec / (10**12)

    # sanity check so we can use accelerator
    if (tops < 39.8).all():
        print("ok to use accelerator")
    else:
        print('warning accelerator not fast enough!!!')
        print(f"{tops=}")

    # compute pwr (her for E)
    watts = tops/646.6

    return watts


def pwr_DACs(BW, oversampling_factor, b, M):
    # DAC parameters
    Vdd = 3
    Io = 10 * 10**(-6)
    Cp = 1 * 10**(-12)

    # compute sample freq
    fs = oversampling_factor * BW

    # pwr of 1 DAC
    static = 0.5 * Vdd * Io * (2**b - 1)
    dynamic = b * Cp * (fs / 2) * (Vdd**2)
    P_dac = 0.5 * Vdd * Io * (2**b - 1) + b * Cp * (fs / 2) * (Vdd**2)

    # pwr of all DACs
    P_all_dacs = 2 * M * P_dac # times two cause I and Q

    return P_all_dacs

if __name__ == '__main__':
    # system params
    M = 32
    K = 1
    b = 1
    Tc = 10 * 10**-3 #coherence time 10 ms
    oversampling_factor = 4

    # GNN params
    dl = 128
    Nh = 4

    # symbol rate
    BW = np.arange(1, 1002, 100) * 10**3 #1002
    rolloff = 0.1#0.9
    Rs = BW / (1 + rolloff)

    # complexity GNN
    flops_second = flops_per_second(M, K, b, dl, Nh, Rs)
    plt.plot(BW/1000, flops_second/(10**12))
    plt.xlabel("BW [KHz]")
    plt.ylabel("GFLOPS/second")
    tikzplotlib.save('flop_vs_bw.tex')
    fig = plt.gcf()
    fig.savefig('flops_vs_bw.pdf')
    plt.show()



    # pwr gnn
    pwr_gnn = power_gnn(flops_second)
    plt.plot(BW/1000, pwr_gnn)
    plt.xlabel("BW [KHz]")
    plt.ylabel("Power [W]")
    plt.show()


    # complexity ZF
    flops_second_zf = flops_per_second_ZF(M, K, Rs, Tc)

    # compute pwr cons of DACs
    P_dacs_gnn = pwr_DACs(BW, oversampling_factor, 1, M)
    P_dacs_mrt = pwr_DACs(BW, oversampling_factor, 3, M)
    #P_dacs_mrt_b4 = pwr_DACs(BW, oversampling_factor, 4, M)

    print(f'{P_dacs_gnn=}')
    print(f'{P_dacs_mrt=}')




    #plt.plot(BW / 10**3, pwr_gnn, label='GNN consumption')
    plt.plot(BW / 10**3, 1000 * P_dacs_gnn, label='DACs consumption (GNN b=1)')
    plt.plot(BW / 10**3, 1000 * P_dacs_mrt, label='DACs consumption (MRT b=3)')
    plt.plot(BW / 10**3, (P_dacs_gnn+pwr_gnn) * 1000, label=f'GNN + DACs consumption (b=1), Nh={Nh}, dh={dl}')
    #plt.plot(BW / 10**3, P_dacs_mrt_b4, label='DACs consumption (MRT b=4)')

    # #additional plots with diff NN sizes
    # dl = 32
    # Nh = 8
    # flops_second_2 = flops_per_second(M, K, b, dl, Nh, Rs)
    # pwr_gnn2 = power_gnn(flops_second_2)
    # plt.plot(BW / 10**3, (P_dacs_gnn+pwr_gnn2) * 1000, label=f'GNN + DACs consumption (b=1), Nh={Nh}, dh={dl}')
    #
    # dl = 64
    # Nh = 8
    # flops_second_3 = flops_per_second(M, K, b, dl, Nh, Rs)
    # pwr_gnn3 = power_gnn(flops_second_3)
    # plt.plot(BW / 10**3, (P_dacs_gnn+pwr_gnn3) * 1000, label=f'GNN + DACs consumption (b=1), Nh={Nh}, dh={dl}')



    plt.xlabel("BW [kHz]")
    plt.ylabel('power [mW]')
    plt.legend()
    tikzplotlib.save('pcons_vs_bw.tex')
    fig = plt.gcf()
    fig.savefig('pcons_vs_bw.pdf')
    plt.show()
    #print(f'Tflops per second: {flops_second/10**12}')