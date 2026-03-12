import numpy as np
import matplotlib.pyplot as plt
import time
import cvxpy as cp
from scipy.stats import multivariate_normal





def MRT(s, H):
    # P_t = 1

    # beta
    B = H.shape[-1]
    beta = np.sqrt(np.trace(H.T.conj() @ H)) / B

    # precoding matrix
    P = 1/(B*beta) * H.conj().T

    # precode
    x = P @ s

    return x, beta


def MRT_quant(s, H):
    # P_t = 1

    # beta
    B = H.shape[-1]
    beta = np.sqrt(np.trace(H.T.conj() @ H)) / B

    # precoding matrix
    P = 1 / (B * beta) * H.conj().T

    # precode
    x = P @ s

    # quantize
    x_quant = np.sqrt(1/(2*B)) * (np.sign(x.real) + 1j * np.sign(x.imag))
    #
    # test = np.abs(x_quant)**2
    # test2 = np.sum(np.abs(x_quant)**2)
    #
    # print(f' pwr x quant = {np.sum(np.abs(x_quant)**2)}')

    return x_quant, beta

# Simulation parameters (replace these with your actual parameters)
par = {
    'U': 2,                   # Number of users,
    'B': 8,                  # Number of Tx antennas
    'bps': 2,                 # Bits per symbol for QPSK
    'symbols': np.array([1+1j, -1+1j, -1-1j, 1-1j]),  # Example symbols for QPSK
    'precoder': ['MRT_inf', 'MRT'],  # Only MRT cases to consider
    'quantizer': lambda x: x, # Placeholder for the quantizer function
    'SNRdB_list': [-10, -5, 0, 5, 10, 15],  # SNR values in dB
    'trials': 100             # Number of trials todo increase
}

# Initialize results
res = {
    'TxMaxPower': np.zeros((len(par['precoder']), len(par['SNRdB_list']))),
    'RxMaxPower': np.zeros((len(par['precoder']), len(par['SNRdB_list']))),
    'TxAvgPower': np.zeros((len(par['precoder']), len(par['SNRdB_list']))),
    'RxAvgPower': np.zeros((len(par['precoder']), len(par['SNRdB_list']))),
    'VER': np.zeros((len(par['precoder']), len(par['SNRdB_list']))),
    'SER': np.zeros((len(par['precoder']), len(par['SNRdB_list']))),
    'BER': np.zeros((len(par['precoder']), len(par['SNRdB_list'])))
}

# Start simulation
time_elapsed = 0
start_time = time.time()

for tt in range(par['trials']):
    # log
    print(f'trial {tt}')

    # Generate random bit stream
    b = np.random.randint(0, 2, (par['U'], par['bps']))
    b_flipped = b[:, [1, 0]] # intermediate flip to deal with the np.packbits function
    idx = np.packbits(b_flipped, bitorder='little', axis=-1).flatten()
    s = par['symbols'][idx]

    # Generate iid Gaussian channel matrix & noise vector
    n = np.sqrt(0.5) * (np.random.randn(par['U']) + 1j * np.random.randn(par['U']))
    H = np.sqrt(0.5) * (np.random.randn(par['U'], par['B']) + 1j * np.random.randn(par['U'], par['B']))

    #todo replace by loading own channels from testset

    # Algorithm loop
    for pp, precoder in enumerate(par['precoder']):
        if precoder == 'MRT_inf':
            x, beta = MRT(s, H)

        elif precoder == 'MRT':
            x, beta = MRT_quant(s, H)

        #todo add GNN




        # SNR loop
        for k, SNRdB in enumerate(par['SNRdB_list']):
            N0 = 10 ** (-SNRdB / 10)

            # # noise dependent precoders
            # if precoder == 'SDR':
            #     x, beta, x_random, beta_random = SDR(s, H, N0)

            # Transmit data over noisy channel
            Hx = H @ x
            y = Hx + np.sqrt(N0) * n

            # Track power metrics
            res['TxMaxPower'][pp, k] = max(res['TxMaxPower'][pp, k], np.sum(np.abs(x) ** 2))
            res['RxMaxPower'][pp, k] = max(res['RxMaxPower'][pp, k], np.sum(np.abs(Hx) ** 2) / par['U'])
            res['TxAvgPower'][pp, k] += np.sum(np.abs(x) ** 2)
            res['RxAvgPower'][pp, k] += np.sum(np.abs(Hx) ** 2) / par['U']

            # Scale and detect symbols
            beta = 1
            shat = beta * y # beta=1 for QPSK
            distance = np.abs(shat[:, None] - par['symbols'])**2
            idxhat = np.argmin(distance, axis=1)
            bhat = np.unpackbits(idxhat.astype(np.uint8).reshape(-1, 1), axis=1)[:, -par['bps']:]

            # Error metrics
            err = (idx != idxhat)
            res['VER'][pp, k] += int(np.any(err))
            res['SER'][pp, k] += np.sum(err) / par['U']
            res['BER'][pp, k] += np.sum(b != bhat) / (par['U'] * par['bps'])

    # Time tracking
    if (time.time() - start_time) > 10:
        elapsed = time.time() - start_time
        time_elapsed += elapsed
        print(f"Estimated remaining time: {time_elapsed * (par['trials'] / (tt + 1) - 1) / 60:.2f} min.")
        start_time = time.time()

# Normalize results
for metric in ['VER', 'SER', 'BER', 'TxAvgPower', 'RxAvgPower']:
    res[metric] /= par['trials']
res['time_elapsed'] = time_elapsed

# Results in `res` now contain normalized metrics

# Plot Vector Error Rate (VER)
plt.figure(figsize=(10, 6))
for pp, precoder in enumerate(par['precoder']):
    plt.plot(par['SNRdB_list'], res['VER'][pp, :], marker='o', label=f'{precoder} - VER')
plt.xlabel('SNR (dB)')
plt.ylabel('Vector Error Rate (VER)')
plt.title('Vector Error Rate (VER) vs SNR')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Plot Symbol Error Rate (SER)
plt.figure(figsize=(10, 6))
for pp, precoder in enumerate(par['precoder']):
    plt.plot(par['SNRdB_list'], res['SER'][pp, :], marker='o', label=f'{precoder} - SER')
plt.xlabel('SNR (dB)')
plt.ylabel('Symbol Error Rate (SER)')
plt.title('Symbol Error Rate (SER) vs SNR')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Plot Bit Error Rate (BER)
plt.figure(figsize=(10, 6))
for pp, precoder in enumerate(par['precoder']):
    plt.plot(par['SNRdB_list'], res['BER'][pp, :], marker='o', label=f'{precoder} - BER')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Bit Error Rate (BER) vs SNR')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.show()