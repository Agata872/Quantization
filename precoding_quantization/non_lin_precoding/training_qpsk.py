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
from data_handling import getdata_nonlinprec, ChannelSymbolsDataset, getdata_nonlinprec_QPSK
import tikzplotlib


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def train(sim_params, train_params):
    # unpack simulation parameters
    M = sim_params['M']
    K = sim_params['K']
    Pt = sim_params['Pt']
    bits = sim_params['bits']
    noise_var = sim_params['noise_var']
    quant_params_path = sim_params['quant_params_path']
    quant = sim_params['quant']
    varx = sim_params['varx']
    root_dir = sim_params['root_dir']

    # unpack training parameters
    channel_model = train_params['channel_model']
    model_type = train_params['model_type']
    output_type = train_params['output_type']
    tau = train_params['tau']
    Ntr = train_params['Nr_train']
    Nval = train_params['Nr_val']
    Nte = train_params['Nr_test']
    nr_symbols_per_channel = train_params['nr_symbols_per_channel']
    batch_size = train_params['batch_size']
    nr_epochs = train_params['epochs']
    lr = train_params['lr']
    nr_hidden_layers = train_params['nr_hidden_layers']
    nr_features = train_params['nr_features']
    model_dir = train_params['stored_model_dir']

    # folder for storing model
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = os.path.join(os.getcwd(), model_dir,
                             f'M_{M}_K_{K}_bs_{batch_size}_layers_{nr_hidden_layers}_dl_{nr_features}_tau_{tau}_Ntr_{Ntr}')

    # quantizer params
    if bits == 1:
        output_levels = np.sqrt(Pt / (2 * M)) * torch.Tensor([-1, 1])  # only valid for 1 bit case
    else:
       print(f'QPSK can only be trained for 1 bit!!!')

    # set GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if GPU is available
    print(f'device: {device}')

    # load/generate the data
    datapath = os.path.join(root_dir, r'Quantization\precoding_quantization\non_lin_precoding\datasets',
                            f'{channel_model}')

    Htrain, Hval, Htest, strain, sval, stest, btrain, bval, btest = getdata_nonlinprec_QPSK(nr_symbols_per_channel, datapath, M, K, Ntr, Nval,
                                                                  Nte, channel_model)
    trainset = ChannelSymbolsDataset(Htrain.astype(np.complex64), strain.astype(np.complex64),
                                     nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    validation_set = ChannelSymbolsDataset(Hval.astype(np.complex64), sval.astype(np.complex64),
                                           nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    test_set = ChannelSymbolsDataset(Htest.astype(np.complex64), stest.astype(np.complex64),
                                     nr_symbols_per_channel=nr_symbols_per_channel, device=device)
    training_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # test
    H, s = next(iter(training_dataloader))  # bs x MK x output_feature_size
    print(f'{H.shape} - {s.shape}')

    # create model
    if model_type == 'GNN':
        model = GNNmodel(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels.to(device),
                         quantize=quant, output_type=output_type).to(device)
        if quant:
            name = f'{bits}_bits_GNN_{output_type}_{timestamp}'
        else:
            name = f'GNN_no_quant_{timestamp}'

        model_path = os.path.join(base_path, name)
        create_folder(model_path)

    elif model_type == 'MLP':
        if quant:
            model = MLPmodel(M, K, bits, tau, output_levels.to(device)).to(device)
            model_path = os.path.join(base_path, f'{bits}_bits_MLP_{timestamp}')
            create_folder(model_path)
        else:  # train without quantization
            model = MLPmodel_noquant(M, K).to(device)  # santity check
            model_path = os.path.join(base_path, f'MLP_no_quant_{timestamp}')
            create_folder(model_path)

    print(model)
    print(f'nr trainable params: {sum([param.nelement() for param in model.parameters()])}')

    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # get loss function
    loss_fn = SumRateLoss()

    # containers to store loss
    loss_history = []
    vloss_history = []
    running_loss = 0
    best_vloss = 0
    last_loss = 0

    x_init = torch.zeros((batch_size, M, 2)).to(device)  # zeros as initial input for antennanode features
    # loop over batches
    for epoch in range(nr_epochs):
        with tqdm(training_dataloader, unit='batch') as tqdmbatch:
            for i, batch in enumerate(tqdmbatch):
                H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel

                # move input data to the GPU
                H, s = H, s

                # set accumulated grads to zero
                optimizer.zero_grad()

                # forward pass
                outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
                for sidx in range(s.shape[-1]):
                    if model_type == 'GNN':
                        outputs[:, :, sidx] = model(H, s[:, :, sidx],
                                                    x_init)  # NN takes 1 channel and 1 symbol as input
                    else:
                        outputs[:, :, sidx] = model(H, s[:, :, sidx])  # NN takes 1 channel and 1 symbol as input

                # normalization accross the symbol dimension (when multiple bits are considered) not needed cause 1 bit and QPSK!
                # l2_norm = torch.linalg.vector_norm(outputs, ord=2, dim=1)  # bs x nr_symbols
                # expt_x2 = torch.mean(l2_norm ** 2, dim=-1)  # bs
                # epsilon = 1e-7  # to avoid NaN
                # alpha = torch.sqrt(Pt / (expt_x2 + epsilon))
                # alpha = alpha[:, None, None]  # add two dimensions for broadcasting
                # normalized_output = alpha * outputs
                # l2_norm_check = torch.linalg.vector_norm(normalized_output, ord=2, dim=1)  # bs x nr_symbols
                # expt_x2_check = torch.mean(l2_norm_check ** 2, dim=-1)
                # # print(f'{outputs=}')

                # compute loss
                loss = loss_fn(outputs.to(device), H.type(torch.complex64), s.type(torch.complex64),
                               noise_var)

                # backprop + gradient descent step
                loss.backward()
                optimizer.step()

                # gather data and report
                running_loss += loss.item()
                if i % 100 == 99:
                    last_loss = running_loss / 100

                    # log some values
                    tqdmbatch.set_postfix(loss=last_loss)
                    running_loss = 0

        # save training loss after each epoch
        loss_history.append(last_loss)

        # validation loss
        model.eval()
        with torch.no_grad():
            running_vloss = 0
            for i, batch in enumerate(validation_dataloader):
                H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel
                bs = H.shape[0]
                x_init = torch.zeros((bs, M, 2)).to(device)  # zeros as initial input for antenna node features

                # move input data to the GPU
                H, s = H.to(device), s.to(device)

                # forward pass
                outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
                for sidx in range(s.shape[-1]):
                    outputs[:, :, sidx] = model(H, s[:, :, sidx], x_init)  # NN takes 1 channel and 1 symbol as input

                # normalization across the symbol dimension (when multiple bits are considered)
                l2_norm = torch.linalg.vector_norm(outputs, ord=2, dim=1)  # bs x nr_symbols
                expt_x2 = torch.mean(l2_norm ** 2, dim=-1)  # bs
                epsilon = 1e-7  # to avoid NaN
                alpha = torch.sqrt(Pt / (expt_x2 + epsilon))
                alpha = alpha[:, None, None]  # add two dimensions for broadcasting
                normalized_output = alpha * outputs

                # compute loss
                vloss = loss_fn(normalized_output.to(device), H.type(torch.complex64), s.type(torch.complex64),
                                noise_var)
                running_vloss += vloss.item()

        # log the validation loss
        avg_vloss = running_vloss / (i + 1)
        print(f'avg vallidation loss: {avg_vloss}')
        vloss_history.append(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            path = os.path.join(model_path, 'model_{}'.format(timestamp))
            torch.save(model.state_dict(), path)

        # print epoch nr
        print(f'epoch: {epoch}')

    # plot training loss
    plt.plot(loss_history, label='training loss')
    plt.plot(vloss_history, label='validation loss')
    plt.legend()
    fig = plt.gcf()
    fig.savefig(os.path.join(model_path, 'loss_history.pdf'))
    plt.show()

    """post training"""

    # load the best model
    saved_model = GNNmodel(M, K, nr_features, nr_hidden_layers, bits, tau, output_levels.to(device), quantize=True).to(
        device)
    saved_model.load_state_dict(torch.load(os.path.join(model_path, 'model_{}'.format(timestamp))))

    # define snr points
    snr_points = np.array([-30, -20, -10, 0.1, 10, 20, 30])

    # test set
    saved_model.eval()
    nr_batches = int(Nte / batch_size)
    Rsum_batches = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf_agc = np.zeros((nr_batches, len(snr_points)))
    Rsum_batches_zf_noquant = np.zeros((nr_batches, len(snr_points)))

    with torch.no_grad():
        running_vloss = 0
        for i, batch in enumerate(test_dataloader):
            print(f'batch of test set {i} / {nr_batches}')
            H, s = batch  # H: bs x M x K, s: bs x K x nr_symbols_per_channel
            bs = H.shape[0]
            x_init = torch.zeros((bs, M, 2)).to(device)  # zeros as initial input for antennanode features

            # move input data to the GPU
            H, s = H.to(device), s.to(device)

            # forward pass
            outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
            for sidx in range(s.shape[-1]):
                outputs[:, :, sidx] = saved_model(H, s[:, :, sidx], x_init)  # NN takes 1 channel and 1 symbol as input

            # normalization accross the symbol dimension (when multiple bits are considered)
            l2_norm = torch.linalg.vector_norm(outputs, ord=2, dim=1)  # bs x nr_symbols
            expt_x2 = torch.mean(l2_norm ** 2, dim=-1)  # bs
            epsilon = 1e-7  # to avoid NaN
            alpha = torch.sqrt(Pt / (expt_x2 + epsilon))
            alpha = alpha[:, None, None]  # add two dimensions for broadcasting
            normalized_output = alpha * outputs

            # compute sumrate
            Rsum_batches[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                  Pt=M, automatic_gain_control=False, precoding='non-linear',
                                                  x_nonlin=normalized_output.numpy(),
                                                  quant_params_path=quant_params_path,
                                                  s_provided=s.cpu().numpy(), normalize_across_symbols=True)
            # zf/mrt benchmark
            Rsum_batches_zf[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                     Pt=M, automatic_gain_control=False, precoding='zf-mrt',
                                                     quant_params_path=quant_params_path, s_provided=s.cpu().numpy(),
                                                     normalize_across_symbols=True)

            # zf/mrt benchmark
            Rsum_batches_zf_agc[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='non-uniform',
                                                         Pt=M, automatic_gain_control=True, precoding='zf-mrt',
                                                         quant_params_path=quant_params_path,
                                                         s_provided=s.cpu().numpy(), normalize_across_symbols=True)
            # zf/mrt no quant
            Rsum_batches_zf_noquant[i, :] = Rsum_Bussgang_Rx(H.cpu().numpy(), snr_points, bits=bits, quant='none',
                                                             Pt=M, precoding='zf-mrt', s_provided=s.cpu().numpy(),
                                                             normalize_across_symbols=True)

    # avg across the batches
    Rsum_avg = np.mean(Rsum_batches, axis=0)
    Rsum_avg_zf = np.mean(Rsum_batches_zf, axis=0)
    Rsum_avg_zf_agc = np.mean(Rsum_batches_zf_agc, axis=0)
    Rsum_avg_zf_no_quant = np.mean(Rsum_batches_zf_noquant, axis=0)

    plt.plot(snr_points, Rsum_avg, label='non lin prec')
    plt.plot(snr_points, Rsum_avg_zf, label='ZF/MRT')
    plt.plot(snr_points, Rsum_avg_zf_agc, label='ZF/MRT - AGC')
    plt.plot(snr_points, Rsum_avg_zf_no_quant, label='ZF/MRT - no quant')
    plt.xlabel('SNR [dB]')
    plt.ylabel('R sum')
    plt.legend()
    fig = plt.gcf()
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(os.path.join(model_path, 'Rsum_testeset.tex'))
    fig.savefig(os.path.join(model_path, 'Rsum_testeset.pdf'))
    plt.show()

    logparams(os.path.join(model_path, 'sim_params.json'), sim_params)
    logparams(os.path.join(model_path, 'train_params.json'), train_params)


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

    # sim params
    M = 8
    K = 2
    Pt = M
    bits = 1
    quant = True  # train with or without quantization

    # train params
    channel_model = 'iid'  # 'los' #'cellfree'
    nr_hidden_layers, nr_features = 4, 128
    model_type = 'GNN'  # MLP
    output_type = 'gumbel_softmax_hard'  # 'softmax_hard', 'softmax', 'gumbel_softmax_hard', 'gumbel_softmax'
    batch_size = 64  # 128
    lr = 0.5 * 10 ** -3
    nr_epochs = 20  # 20 #10
    snr_tx = 20  # in db
    noise_var = Pt / (10 ** (snr_tx / 10))
    tau = 4  # for gumbel softmax
    stored_model_dir = f'stored_models_{channel_model}_QPSK'  # todo set to desired folder!

    # data set params
    Ntr = 200000  # should be multiple of batchsize
    Nval = 1000
    Nte = 10000
    nr_symbols_per_channel = 125  # because QPSK Pt=cte so no normalization needed over the symbols

    # put all the params in a dictionary to store it
    sim_params = {
        'M': M,
        'K': K,
        'Pt': Pt,
        'bits': bits,
        'snr_tx': snr_tx,
        'noise_var': noise_var,
        'quant_params_path': quant_params_path,
        'quant': quant,
        'varx': varx,
        'root_dir': root_dir
    }

    training_params = {
        'channel_model': channel_model,
        'model_type': model_type,
        'output_type': output_type,
        'tau': tau,
        'Nr_train': Ntr,
        'Nr_val': Nval,
        'Nr_test': Nte,
        'nr_symbols_per_channel': nr_symbols_per_channel,
        'batch_size': batch_size,
        'epochs': nr_epochs,
        'lr': lr,
        'nr_hidden_layers': nr_hidden_layers,
        'nr_features': nr_features,
        'stored_model_dir': stored_model_dir
    }

    M = [16]
    K = [2]
    bits = [1]
    output = ['softmax_hard', 'gumbel_softmax_hard', 'softmax_hard', 'softmax', 'gumbel_softmax']  # todo later
    tau_range = [1]  # todo later (+annealing during training)
    for m in M:
        for tau in tau_range:
            for b in bits:
                for k in K:
                    sim_params['K'] = k
                    sim_params['M'] = m
                    sim_params['Pt'] = m
                    sim_params['bits'] = b
                    snr_tx = 20  # in db
                    noise_var = sim_params['Pt'] / (10 ** (snr_tx / 10))
                    sim_params['noise_var'] = noise_var
                    training_params['output_type'] = 'gumbel_softmax_hard'
                    training_params['tau'] = tau
                    print(f'---------------starting training for-------------------')
                    print(f'{sim_params=}')
                    print(f'{training_params=}')
                    train(sim_params, training_params)
                    print(f'--------------------Done training---------------')





