import torch
import numpy as np
from utils.utils import rayleigh_channel_MU, getSymbols, create_folder, logparams, los_channel_MU, getSymbols_QPSK
import os

class ChannelSymbolsDataset(torch.utils.data.Dataset):
    def __init__(self, H, s, nr_symbols_per_channel, device='cuda'):
        """
        :param H: nr_channels x M x K
        :param s: K x nr_channels*nr_symbols_per_channel
        """
        self.H = torch.from_numpy(H).to(device)
        self.s = torch.from_numpy(s).to(device)
        self.nr_symbols_per_channel = nr_symbols_per_channel

    def __len__(self):
        return len(self.H)

    def __getitem__(self, idx):
        channel = self.H[idx, :, :]
        symbols = self.s[:, idx * self.nr_symbols_per_channel:(idx + 1) * self.nr_symbols_per_channel]
        return channel, symbols

def getdata_nonlinprec(nr_symbols_per_channel, datapath, M, K, Ntr, Nval, Nte, channel_model='iid'):
    """
    :param nr_channels:
    :param nr_symbols_per_channel:
    :param datapath:
    :param M:
    :param K:
    :param Ntr:
    :param Nval:
    :param Nte:
    :return:
    """
    output_dir = os.path.join(datapath, f'M_{M}_K_{K}_Ntr_{Ntr}_Nval_{Nval}_Nte_{Nte}_SperChannel{nr_symbols_per_channel}')
    print(f'{output_dir=}')
    if os.path.exists(output_dir):
        # load everything
        Htrain = np.load(os.path.join(output_dir, 'Htrain.npy'))
        Hval = np.load(os.path.join(output_dir, 'Hval.npy'))
        Htest = np.load(os.path.join(output_dir, 'Htest.npy'))
        strain = np.load(os.path.join(output_dir, 'strain.npy'))
        sval = np.load(os.path.join(output_dir, 'sval.npy'))
        stest = np.load(os.path.join(output_dir, 'stest.npy'))
        print(f' I loaded data')
    else:
        # generate channels
        if channel_model == 'iid':
            nr_channels = Ntr + Nval + Nte
            H = np.zeros((nr_channels, M, K), dtype=complex)
            for i in range(nr_channels):
                H[i, :, :] = rayleigh_channel_MU(M, K)
        elif channel_model == 'los':
            print('in los part')
            nr_channels = Ntr + Nval + Nte
            H = np.zeros((nr_channels, M, K), dtype=complex)
            for i in range(nr_channels):
                H[i, :, :] = los_channel_MU(M, K)

        # split into test, val and train sets
        Htrain = H[0:Ntr, :, :]
        Hval = H[Ntr:Nval + Ntr, :, :]
        Htest = H[Ntr + Nval:Ntr + Nval + Nte, :, :]

        # generate symbols
        nr_symbols = nr_channels * nr_symbols_per_channel
        s = np.zeros((K, nr_symbols), dtype=complex)
        for k in range(K):
            s[k, :] = getSymbols(nr_symbols, p=1)

        # split symbols into test, val and trainsets
        strain = s[:, 0:Ntr*nr_symbols_per_channel]
        sval = s[:, Ntr*nr_symbols_per_channel: (Ntr+Nval)*nr_symbols_per_channel]
        stest = s[:, (Ntr+Nval)*nr_symbols_per_channel:(Ntr+Nval+Nte)*nr_symbols_per_channel]

        # save the data
        create_folder(output_dir)
        np.save(os.path.join(output_dir, 'Htrain.npy'), Htrain)
        np.save(os.path.join(output_dir, 'Hval.npy'), Hval)
        np.save(os.path.join(output_dir, 'Htest.npy'), Htest)
        np.save(os.path.join(output_dir, 'strain.npy'), strain)
        np.save(os.path.join(output_dir, 'sval.npy'), sval)
        np.save(os.path.join(output_dir, 'stest.npy'), stest)

        print(f' I created data')
    return Htrain, Hval, Htest, strain, sval, stest


def getdata_nonlinprec_QPSK(nr_symbols_per_channel, datapath, M, K, Ntr, Nval, Nte, channel_model='iid'):
    """
    :param nr_channels:
    :param nr_symbols_per_channel:
    :param datapath:
    :param M:
    :param K:
    :param Ntr:
    :param Nval:
    :param Nte:
    :return:
    """
    output_dir = os.path.join(datapath, f'QPSK_M_{M}_K_{K}_Ntr_{Ntr}_Nval_{Nval}_Nte_{Nte}_SperChannel{nr_symbols_per_channel}')
    print(f'{output_dir=}')
    if os.path.exists(output_dir):
        # load everything
        Htrain = np.load(os.path.join(output_dir, 'Htrain.npy'))
        Hval = np.load(os.path.join(output_dir, 'Hval.npy'))
        Htest = np.load(os.path.join(output_dir, 'Htest.npy'))
        strain = np.load(os.path.join(output_dir, 'strain.npy'))
        sval = np.load(os.path.join(output_dir, 'sval.npy'))
        stest = np.load(os.path.join(output_dir, 'stest.npy'))
        btrain = np.load(os.path.join(output_dir, 'btrain.npy'))
        bval = np.load(os.path.join(output_dir, 'bval.npy'))
        btest = np.load(os.path.join(output_dir, 'btest.npy'))
        print(f' I loaded data')

    else:
        # generate channels
        if channel_model == 'iid':
            nr_channels = Ntr + Nval + Nte
            H = np.zeros((nr_channels, M, K), dtype=complex)
            for i in range(nr_channels):
                H[i, :, :] = rayleigh_channel_MU(M, K)
        elif channel_model == 'los':
            print('in los part')
            nr_channels = Ntr + Nval + Nte
            H = np.zeros((nr_channels, M, K), dtype=complex)
            for i in range(nr_channels):
                H[i, :, :] = los_channel_MU(M, K)

        # split into test, val and train sets
        Htrain = H[0:Ntr, :, :]
        Hval = H[Ntr:Nval + Ntr, :, :]
        Htest = H[Ntr + Nval:Ntr + Nval + Nte, :, :]

        # generate symbols
        nr_symbols = nr_channels * nr_symbols_per_channel
        s = np.zeros((K, nr_symbols), dtype=complex)
        b = np.zeros((K, nr_symbols, 2)) # K x nr_symbols x bps
        for k in range(K):
            s[k, :], b[k, :, :] = getSymbols_QPSK(nr_symbols)


        # split symbols into test, val and trainsets
        strain = s[:, 0:Ntr*nr_symbols_per_channel]
        sval = s[:, Ntr*nr_symbols_per_channel: (Ntr+Nval)*nr_symbols_per_channel]
        stest = s[:, (Ntr+Nval)*nr_symbols_per_channel:(Ntr+Nval+Nte)*nr_symbols_per_channel]

        # store bitstreams
        btrain = b[:, 0:Ntr*nr_symbols_per_channel, :]
        bval = b[:, Ntr*nr_symbols_per_channel: (Ntr+Nval)*nr_symbols_per_channel, :]
        btest = b[:, (Ntr+Nval)*nr_symbols_per_channel:(Ntr+Nval+Nte)*nr_symbols_per_channel, :]

        # save the data
        create_folder(output_dir)
        np.save(os.path.join(output_dir, 'Htrain.npy'), Htrain)
        np.save(os.path.join(output_dir, 'Hval.npy'), Hval)
        np.save(os.path.join(output_dir, 'Htest.npy'), Htest)
        np.save(os.path.join(output_dir, 'strain.npy'), strain)
        np.save(os.path.join(output_dir, 'sval.npy'), sval)
        np.save(os.path.join(output_dir, 'stest.npy'), stest)
        np.save(os.path.join(output_dir, 'btrain.npy'), btrain)
        np.save(os.path.join(output_dir, 'bval.npy'), bval)
        np.save(os.path.join(output_dir, 'btest.npy'), btest)

        print(f' I created data')
    return Htrain, Hval, Htest, strain, sval, stest, btrain, bval, btest
