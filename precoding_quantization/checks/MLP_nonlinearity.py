import numpy as np
import matplotlib.pyplot as plt
import os
from utils.quantization import quantize_nonuniform
from utils.utils import create_folder
# import tensorflow as tf
# from tensorflow import keras
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    """ Check quantization for one quantizer analytical exprssion vs numerical """

    # quantizer params path
    varx = 0.5
    quant_params_path = r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')

    # load quantizer params
    bits = 3
    thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
    outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
    beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))

    # construct data
    nr_data = 10000#100000
    alpha = 3
    x_in_re = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in_im = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in = x_in_re + 1j* x_in_im

    # quantize
    y = quantize_nonuniform(x_in, thresholds, outputlevels)

    # only do real part (cause imag is the same)
    y_out_re = torch.tensor(np.real(y)[:, np.newaxis], dtype=torch.float32)
    x_in_re = torch.tensor(x_in_re[:, np.newaxis], dtype=torch.float32)

    # Create a TensorDataset
    dataset = TensorDataset(x_in_re, y_out_re)

    # Create a DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    plt.plot(x_in_re, y_out_re)
    plt.show()

    # create model
    neurons = 64
    model = nn.Sequential(
        nn.Linear(1, neurons),
        nn.ReLU(),
        nn.Linear(neurons, neurons),
        nn.ReLU(),
        nn.Linear(neurons, 1)
    )

    # loss MSE
    loss_fn = nn.MSELoss()

    # select optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    epochs = 10
    loss_value = []
    for i in range(epochs):
        # Iterate over the data
        for inputs, labels in dataloader:
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            print(f'{i}) {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_value.append(loss.item())

    plt.plot(loss_value)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.show()

    # save model params
    #torch.save(model.state_dict(), "my_model.pickle")

    y_test_check = np.zeros(len(x_in_re))
    for x_i in x_in_re:
        print(f'{x_i=}')
        out = model(x_i[:, np.newaxis])
        y_test_check[i]=model(x_i[:, np.newaxis])

    y_test = model(x_in_re).detach().numpy()


    plt.plot(x_in_re.detach().numpy(), y_test, label='nn')
    plt.plot(x_in_re.detach().numpy(), y_out_re.detach().numpy(), label='dac')
    plt.legend()
    plt.show()