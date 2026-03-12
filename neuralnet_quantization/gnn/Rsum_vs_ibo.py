import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils import load_params, C2R, getSymbols, create_folder, get_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from gnn.losses import polynomial_loss
from gnn.model import GNN_layer, Pwr_norm_gnn
import tikzplotlib
import sys
from testing import avg_sum_rate
import re
from tqdm import tqdm

def get_sumrates(Htest, model, Bs, noise_var, pa='poly', Srapp=2.0, Pt=1, M=64, K=2, backoffdpd=-3.0):
    '''
    :param model: trained model for precoding
    :param Bs: poly coefs: [B1, B2, ..., B2N+1]
    :param H: channels (bs x m x k)
    :param pa: pa type
    :param zf: if true add zero forcing as a benchmark
    :param lin: if true add zf with linear pa as a benchmark
    :return: evaluation of the model on the testset, compared to ZF
    '''

    #convert channel to correct shape for the nn
    Htest_real = C2R(Htest)
    Htest_flat = tf.reshape(Htest_real, [Htest.shape[0], -1, 2])

    #run the neural net to get precoding vectors for all channel realizations
    y_preds = model.predict(Htest_flat)

    #containers to store the sumrates
    Rnn = np.zeros((Htest.shape[0]), dtype=complex) #neural net
    Rnn_dpd = np.zeros((Htest.shape[0]), dtype=complex) #neural net
    Rzf = np.zeros((Htest.shape[0]), dtype=complex) #zero forcing
    Rzfdpd = np.zeros((Htest.shape[0]), dtype=complex) #zero forcing linear PA
    Rzf_rapp = np.zeros((Htest.shape[0]), dtype=complex)
    Rzf_rapp_amam = np.zeros((Htest.shape[0]), dtype=complex)
    Rnn_rapp = np.zeros((Htest.shape[0]), dtype=complex)

    #compute sumrate over each channel realization
    for i in tqdm(range(Htest.shape[0])):
        #print(f'channel realization: {i}')
        H = Htest_real[i, :, :]

        # compute precoding coeff with the nn
        #print(f'M: {M}, K:{K}')
        y_pred = y_preds[i, :, :, :] #(1, M, K, 2)
        Wre = y_pred[:, :, 0]
        Wim = y_pred[:, :, 1]
        Wnn = Wre + 1j * Wim  # M x K

        # compute the avg sumrate for the neural network precoder
        #print('nonlin NEURAL NET')
        Rnn[i] = avg_sum_rate(Wnn, Htest[i, :, :], noise_var, Srapp=Srapp, pa=pa, b3=None, Bs=Bs)
        Rnn_dpd[i] = avg_sum_rate(Wnn, Htest[i, :, :], noise_var, Srapp=Srapp, pa='softlim', b3=None, Bs=Bs, backoff=backoffdpd)
        Rnn_rapp[i] = avg_sum_rate(Wnn, Htest[i, :, :], noise_var, Srapp=Srapp, pa='rapp', b3=None, Bs=Bs, backoff=backoffdpd)

        #zero forcing
        #print('nonlin ZF')
        Hcomplex = Htest[i, :, :]#np.squeeze(Htest[i, :, :])
        Wzf = Hcomplex.conj() @ np.linalg.inv(Hcomplex.T @ Hcomplex.conj())  # M x K
        norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
        Wzf *= norm
        Rzf[i] = avg_sum_rate(Wzf, Htest[i, :, :], noise_var, pa=pa, Srapp=Srapp, b3=None, Bs=Bs)
        Rzfdpd[i] = avg_sum_rate(Wzf, Htest[i, :, :], noise_var, pa='softlim', Srapp=Srapp, b3=None, Bs=Bs, backoff=backoffdpd)
        Rzf_rapp[i] = avg_sum_rate(Wzf, Htest[i, :, :], noise_var, Srapp=Srapp, pa='rapp', b3=None, Bs=Bs, backoff=backoffdpd)
        Rzf_rapp_amam[i] = avg_sum_rate(Wzf, Htest[i, :, :], noise_var, Srapp=Srapp, pa='rapp_amam_only', b3=None, Bs=Bs, backoff=backoffdpd)



        # #add linear pa as benchmark
        # print('lin ZF')
        # Hcomplex = Htest[i, :, :]#np.squeeze(Htest[i, :, :])
        # Wzf = Hcomplex.conj() @ np.linalg.inv(Hcomplex.T @ Hcomplex.conj())  # M x K
        # norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
        # Wzf *= norm
        # Rzflin[i] = avg_sum_rate(Wzf, Htest[i, :, :], noise_var,Srapp=Srapp ,pa='lin', b3=None, Bs=Bs)

    return Rnn, Rzf, Rzfdpd, Rnn_dpd, Rnn_rapp, Rzf_rapp, Rzf_rapp_amam


if __name__ == '__main__':
    #define path of the model to be tested
    M, K = 64, 4
    basepath = r'D:\thomas.feys\GNN_precoder_snr_input\gnn\stored_models_diff_IBO'
    basepath = os.path.join(basepath, f'M_64_K_{K}_nvar_0.64')
    modelnames = os.listdir(basepath) #retrieve names of the folder that stores the model
    print(modelnames)

    backoffs = np.array([-9, -7.5, -6, -4.5, -3, -1.5, 0.0])
    Rnn_diff_bo = np.zeros_like(backoffs)
    Rzf_diff_bo = np.zeros_like(backoffs)
    Rzf_dpd = np.zeros_like(backoffs)
    Rnn_dpp_diff_bo = np.zeros_like(backoffs)
    Rnn_rapp_diff_bo = np.zeros_like(backoffs)
    Rzf_rapp_diff_bo = np.zeros_like(backoffs)
    Rzf_rapp_amam_diff_bo = np.zeros_like(backoffs)

    for i, modelname in enumerate(modelnames):
        path = os.path.join(basepath, modelname)

        #get the back-off
        if modelname != 'Rsum_vs_bo' and modelname != 'Rsum_vs_bo_correct':
            str_bo = modelname.split('_IBO_')[0]
            bo = float(str_bo)
            index = np.where(backoffs == bo)[0][0]
            print(f'modelname: {modelname}')
            print(f'ibo: {bo}')
            print(f'index: {index}')

            # load simulation params
            params = load_params(os.path.join(path, 'mimo_params.json'))
            trainingparams = load_params(os.path.join(path, 'train_params.json'))

            # get the params out
            Pt = params['Pt']
            Ntr = trainingparams['Nr_train']
            Nval = trainingparams['Nr_val']
            Nte = trainingparams['Nr_test']

            # load model
            model = keras.models.load_model(
                os.path.join(path, 'model.h5'),
                custom_objects={
                    'GNN_layer': GNN_layer,
                    'Pwr_norm_gnn': Pwr_norm_gnn,
                    'loss': polynomial_loss
                }
            )
            print(model.summary())

            # load poly params
            polyparams = np.load(os.path.join(path, 'Bs.npy'))
            print(f'poly params: {polyparams}')

            # load channels
            datafolder = os.path.join(os.getcwd(), 'datasets')
            Htrain, Hval, Htest = get_data(M, K, Ntr, Nval, Nte, datafolder)

            # set snr point
            snr_tx = 20  # dB
            noise_var = np.array([Pt / (10 ** (snr_tx / 10))])

            #print(f'starting simultaion for: M={M}, K={K}, Pt= {Pt}, b3={b3} snr tx:{snr_tx}')
            print(f'starting simultaion for: {modelname} M={M}, K={K}, Pt= {Pt}, Bs={polyparams} snr tx:{snr_tx}')

            #evaluate(Htest[0:250, :, :], model, b3, path=path, Pt=Pt, M=M, K=K) #when using old third order model
            pa = 'poly'
            nr_channels = 10#1000
            #polyparams = polyparams[0:2] #only take third order distortion
            print(f'________________________________________POLY PARAMS___________________________: {polyparams}')
            Rnn, Rzf, Rzfdpd, Rnn_dpd, Rnn_rapp, Rzf_rapp, Rzf_rapp_amam = get_sumrates(Htest[0:nr_channels, :, :], model, polyparams, noise_var, pa=pa, Srapp=2, Pt=Pt, M=M,
                                            K=K, backoffdpd=bo)

            Rnn_diff_bo[index] = np.mean(Rnn, axis=0)
            Rzf_diff_bo[index] = np.mean(Rzf, axis=0)
            Rzf_dpd[index] = np.mean(Rzfdpd, axis=0)
            Rnn_dpp_diff_bo[index] = np.mean(Rnn_dpd, axis=0)
            Rnn_rapp_diff_bo[index] = np.mean(Rnn_rapp, axis=0)
            Rzf_rapp_diff_bo[index] = np.mean(Rzf_rapp, axis=0)
            Rzf_rapp_amam_diff_bo[index] = np.mean(Rzf_rapp_amam, axis=0)

    # extra_bo = np.array([-20, -15, -12])
    # Rzf_rapp_extra = np.zeros_like(extra_bo)
    # Rzf_dpd_extra = np.zeros_like(extra_bo)
    # for i, bo in enumerate(extra_bo):
    #     Rnn, Rzf, Rzfdpd, Rnn_dpd, Rnn_rapp, Rzf_rapp = get_sumrates(Htest[0:nr_channels, :, :], model, polyparams,
    #                                                                  noise_var, pa=pa, Srapp=2, Pt=Pt, M=M,
    #                                                                  K=K, backoffdpd=bo)
    #     Rzf_rapp_extra[i] = np.mean(Rzf_rapp, axis=0)
    #     Rzf_dpd_extra[i] = np.mean(Rzfdpd, axis=0)
    #
    # Rzf_dpd_long = np.concatenate((Rzf_dpd_extra, Rzf_dpd))
    # Rzf_rapp_long = np.concatenate((Rzf_rapp_extra, Rzf_rapp_diff_bo))
    # ibo_long = np.concatenate((extra_bo, backoffs))

    resultspath = os.path.join(os.getcwd(), basepath)
    create_folder(os.path.join(resultspath, 'Rsum_vs_bo'))
    # plt.plot(ibo_long, Rzf_dpd_long, label='ZF + DPD LONG')
    # plt.plot(ibo_long, Rzf_rapp_long, label='ZF + RAPP LONG')
    plt.plot(backoffs, Rnn_diff_bo, marker='+', label='NN')
    plt.plot(backoffs, Rzf_diff_bo, marker='o', label='ZF')
    plt.plot(backoffs, Rzf_dpd, marker='x', label='ZF+DPD')
    plt.plot(backoffs, Rnn_dpp_diff_bo, marker='d', label='NN+DPD')
    plt.plot(backoffs, Rnn_rapp_diff_bo, marker='d', label='NN+RAPP')
    plt.plot(backoffs, Rzf_rapp_diff_bo, marker='d', label='ZF+RAPP')
    plt.plot(backoffs, Rzf_rapp_amam_diff_bo, label='ZF + RAPP AMAM')

    plt.legend()
    plt.xlabel('backoff [dB]')
    plt.ylabel('Rsum')
    tikzplotlib.clean_figure()
    tikzplotlib.save(os.path.join(resultspath, 'Rsum_vs_bo', f'R_vs_bo_k{K}.tex'))
    fig = plt.gcf()
    fig.savefig(os.path.join(resultspath, 'Rsum_vs_bo', f'R_vs_bo_k{K}.pdf'))
    plt.show()