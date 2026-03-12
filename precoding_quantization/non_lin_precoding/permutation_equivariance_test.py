import torch
import numpy as np
from utils.utils import rayleigh_channel_MU, getSymbols, create_folder, logparams
import os
from tqdm import tqdm
from model import MLPmodel, SumRateLoss, MLPmodel_noquant, GNNmodel
import matplotlib.pyplot as plt
from torchsummary import summary
from datetime import datetime
from MIMO_sims.Rsum_all import Rsum_Bussgang_Rx
from data_handling import getdata_nonlinprec, ChannelSymbolsDataset
import tikzplotlib

def permutation_matrix(N):
    rng = np.random.default_rng()
    I = np.identity(N)
    rng.shuffle(I, axis=-1)
    return I

if __name__ == '__main__':
    # for local pc or server
    local = False
    varx = 0.5
    if local:
        root_dir = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub'
        quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
        quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')
    else:
        root_dir = r'D:\thomas.feys'
        quant_params_path = r'D:\thomas.feys\Quantization\precoding_quantization\non-uniform-quant-params'
        quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    bits = 3
    M, K = 32, 16
    Pt = M
    H = np.arange(M * K).reshape((M, K))
    print(f'{H=}')

    # quantizer params
    if bits == 1:
        output_levels = np.sqrt(Pt / (2 * M)) * torch.Tensor([-1, 1])  # only valid for 1 bit case
    else:
        output_levels = torch.from_numpy(np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))).type(
            torch.float32)
        print(f'{output_levels=}')

    # create the model
    tau = 1
    nr_features, nr_hidden_layers = 128, 3
    model = GNNmodel(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels,
                     quantize=True)

    # generate channel and symbol vector
    H = np.arange(M*K, dtype=np.complex64).reshape((M, K))
    s = np.arange(K, dtype=np.complex64)
    x_init = np.zeros(2*M, dtype=np.float32).reshape((M, 2))

    # do permutations

    # add batch dimension
    H_batched = H[None, :, :]
    s_batched = s[None, :]
    x_init_batched = x_init[None, :, :]

    # run model on un permuted data
    y = model(torch.from_numpy(H_batched), torch.from_numpy(s_batched), torch.from_numpy(x_init_batched))
    y_np = y.detach().numpy()
    print(f'{y_np=}')





    P1 = permutation_matrix(M)
    P2 = permutation_matrix(K)
    print(f'{P1=}')
    print(f'{P2=}')

    print(f'P1 H: {P1 @ H}')
    print(f'P2 H: {H @ P2}')
    print(f'P1 H P2: {P1 @ H @ P2}')


    #permute antenna index H and run GNN on it
    Hperm = P1 @ H
    Hperm = Hperm.astype(np.complex64)
    Hperm_batch = Hperm[None, :, :]

    y_perm = model(torch.from_numpy(Hperm_batch), torch.from_numpy(s_batched),
                   torch.from_numpy(x_init_batched)).detach().numpy()

    y_perm_check = P1 @ np.squeeze(y_np)

    print(f'{H=}')
    print(f'{Hperm=}')
    print(f'{y_np=}')
    print(f'{y_perm=}')
    print(f'{y_perm_check=}')
    print(f'diff antenna perm y - y_perm: {y_perm - y_np} normal to be nonzero')
    print(f'diff antenna perm pi y - y_perm: {y_perm - y_perm_check} should be zero')


    # permute user index
    Hperm_k = H @ P2
    sperm = P2 @ s
    print(f'{Hperm_k=}')
    print(f'{sperm=}')

    Hperm_k_batch = Hperm_k[None, :, :].astype(np.complex64)
    sperm_batch = sperm[None, :].astype(np.complex64)

    y_perm_k = model(torch.from_numpy(Hperm_k_batch), torch.from_numpy(sperm_batch),
                   torch.from_numpy(x_init_batched)).detach().numpy()
    print(f'{y_perm_k=}')
    print(f'{y_np=}')

    print(f'diff = {y_perm_k - y_np}')

    # #check if P1 W P2 == GNN(P1 H P2)
    # W_perm = P1 @ output_complex @ P2
    # print(f'{W_perm=}')
    #
    # print(f'should be zero if PE: f(P1 H P2) - P1 W P2: {output_perm_complex - W_perm}')
    # print(f'should not be zero (if zero PI): f(P1 H P2) - W: {output_perm_complex - output_complex}')


