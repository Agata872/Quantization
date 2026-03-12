import numpy as np
import os
from datetime import datetime
from utils.utils import create_folder, logparams, logmodel, C2R, get_data, load_params
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from model import get_GNN, GNN_layer, Pwr_norm_gnn
from callbacks import monitor_weights_and_grads, Grad_tb_callback
from losses import quant_loss_uncorrelated_Rqq
from MIMO_sims.Rsum_all import Rsum_Bussgang_Rx, Rsum_analytical_wrapper, Rsum_Bussgang_DAC
import tikzplotlib

def evaluate_post_training(eval_params, Hval, bestModel, path=None):
    """
    :param eval_params: dictionary with parameters for evaluation
    :param Hval: validation set bs x M x K (to be converted to correct form to input to GNN)
    :param bestModel: model with best weights
    :param path: path to store results
    :return:
    """

    # unpack parameters
    nr_snr_points = eval_params['nr_snr_points']
    Pt = eval_params['Pt']
    M = eval_params['M']
    K = eval_params['K']
    bits = eval_params['bits']
    zf = eval_params['zf']
    inf_res = eval_params['inf_res']
    title_add = eval_params['title_add']
    quant_params_path = eval_params['quant_params_path']
    mode = eval_params['mode']

    # get snr values to compute Rsum at
    snr_points = np.linspace(-30, 35, nr_snr_points)

    # convert channel to correct shape for the nn
    H_real = C2R(Hval)
    H_flat = tf.reshape(H_real, [Hval.shape[0], -1, 2])

    # run GNN
    y_pred = bestModel.predict(H_flat)

    if mode == 'rsum_bussgang_rx':
        # compute Rsum numerically (including correlation)
        Rsum_gnn = Rsum_Bussgang_Rx(Hval, snr_points, bits, quant='non-uniform', Pt=Pt,
                                    automatic_gain_control=True, quant_params_path=quant_params_path,
                                    precoding='gnn', precoding_weights=y_pred)

        # compute Rsum numerically for zf/mrt (including correlation)
        if zf:
            Rsum_zf = Rsum_Bussgang_Rx(Hval, snr_points, bits, quant='non-uniform', Pt=Pt,
                                    automatic_gain_control=True, quant_params_path=quant_params_path,
                                    precoding='zf-mrt')

        # compute Rsum numerically for zf/mrt (including correlation), without quantization
        if inf_res:
            Rsum_zf_inf = Rsum_Bussgang_Rx(Hval, snr_points, bits, quant='none', Pt=Pt,
                                    automatic_gain_control=True, quant_params_path=quant_params_path,
                                    precoding='zf-mrt')
    elif  mode == 'rsum_analytical':
        #assumes uncorellated distortion
        # compute Rsum numerically (including correlation)
        Rsum_gnn = Rsum_analytical_wrapper(Hval, snr_points, bits, quant='non-uniform', Pt=Pt,
                                    automatic_gain_control=True, quant_params_path=quant_params_path,
                                    precoding='gnn', precoding_weights=y_pred)

        # compute Rsum numerically for zf/mrt (including correlation)
        if zf:
            Rsum_zf = Rsum_analytical_wrapper(Hval, snr_points, bits, quant='non-uniform', Pt=Pt,
                                       automatic_gain_control=True, quant_params_path=quant_params_path,
                                       precoding='zf-mrt')

        # compute Rsum numerically for zf/mrt (including correlation), without quantization
        if inf_res:
            Rsum_zf_inf = Rsum_analytical_wrapper(Hval, snr_points, bits, quant='none', Pt=Pt,
                                           automatic_gain_control=True, quant_params_path=quant_params_path,
                                           precoding='zf-mrt')

    elif mode == 'rsum_bussgang_dac_uncorrelated' or mode == 'rsum_bussgang_dac_correlated':
        if mode == 'rsum_bussgang_dac_uncorrelated':
            correlated = False
        else: correlated = True


        # compute Rsum numerically (including correlation)
        Rsum_gnn = Rsum_Bussgang_DAC(Hval, snr_points, bits, quant='non-uniform', Pt=Pt, correlated_dist=correlated,
                                           automatic_gain_control=True, quant_params_path=quant_params_path,
                                           precoding='gnn', precoding_weights=y_pred)

        # compute Rsum numerically for zf/mrt (including correlation)
        if zf:
            Rsum_zf = Rsum_Bussgang_DAC(Hval, snr_points, bits, quant='non-uniform', Pt=Pt, correlated_dist=correlated,
                                              automatic_gain_control=True, quant_params_path=quant_params_path,
                                              precoding='zf-mrt')

        # compute Rsum numerically for zf/mrt (including correlation), without quantization
        if inf_res:
            Rsum_zf_inf = Rsum_Bussgang_DAC(Hval, snr_points, bits, quant='none', Pt=Pt, correlated_dist=correlated,
                                                  automatic_gain_control=True, quant_params_path=quant_params_path,
                                                  precoding='zf-mrt')




    # plot results
    create_folder(os.path.join(path, 'evaluation'))
    plt.plot(snr_points, Rsum_gnn, label='GNN')
    if zf:
        plt.plot(snr_points, Rsum_zf, label='ZF/MRT')
    if inf_res:
        plt.plot(snr_points, Rsum_zf_inf, label='ZF/MRT no quantization')
    plt.title(f'M: {M} - K: {K} - bits: {bits} {title_add}')
    plt.legend()
    plt.xlabel(r'$P_T / \sigma_{v}^2$')
    plt.ylabel(r'$R_{\mathrm{sum}}$')
    tikzplotlib.save(os.path.join(path, 'evaluation', 'Rsum_vs_snr.tex'))
    fig = plt.gcf()
    fig.savefig(os.path.join(path, 'evaluation', 'R_vs_snr.pdf'))
    plt.show()


if __name__ == '__main__':


    dir = r'D:\thomas.feys\Quantization\precoding_quantization\GNN\
    stored_models_larger_models\M_32_K_2_nvar_0.32\GNN_bits_1_M_32_K_2_20-03-2024--10-44-50'
    dir = r'D:\thomas.feys\Quantization\precoding_quantization\GNN\stored_models_larger_models\M_32_K_1_nvar_0.32\GNN_bits_1_M_32_K_1_20-03-2024--08-43-39'
    dir=r'D:\thomas.feys\Quantization\precoding_quantization\GNN\stored_models_larger_models\M_32_K_1_nvar_0.32\GNN_bits_2_M_32_K_1_20-03-2024--09-07-53'
    dir=r'D:\thomas.feys\Quantization\precoding_quantization\GNN\stored_models_larger_models\M_32_K_1_nvar_0.32\GNN_bits_3_M_32_K_1_20-03-2024--09-31-48'
    dir=r'D:\thomas.feys\Quantization\precoding_quantization\GNN\stored_models_larger_models\M_32_K_1_nvar_0.32\GNN_bits_4_M_32_K_1_20-03-2024--09-56-04'
    # load and unpack params
    mimo_params = load_params(os.path.join(dir, 'mimo_params.json'))
    training_params = load_params(os.path.join(dir, 'train_params.json'))
    M = mimo_params['M']
    K = mimo_params['K']
    bits = mimo_params['bits']
    quant_params_path = mimo_params['quant_params_path']
    channelmodel = mimo_params['channelmodel']
    nr_train = training_params['Nr_train']
    nr_val = training_params['Nr_val']
    nr_test = training_params['Nr_test']

    # generate or load channels
    datafolder = os.path.join(os.getcwd(), 'datasets')
    Htrain, Hval, Htest = get_data(M, K, nr_train, nr_val, nr_test, datafolder, channelmodel=channelmodel)

    # load model (to get best weights)
    bestModel = keras.models.load_model(
        os.path.join(dir, 'model.h5'),
        custom_objects={
            'GNN_layer': GNN_layer,
            'Pwr_norm_gnn': Pwr_norm_gnn,
            'loss': quant_loss_uncorrelated_Rqq
        }
    )

    # evaluate the neural nets performance
    mode = 'rsum_analytical' #'rsum_analytical', 'rsum_bussgang_dac_correlated',
    # 'rsum_bussgang_dac_uncorrelated', 'rsum_bussgang_rx'
    eval_params = {
        'nr_snr_points': 24,
        'Pt': M,
        'M': M,
        'K': K,
        'bits': bits,
        'zf': True,
        'inf_res': True,
        'title_add': f'independent_test_{mode}',
        'quant_params_path': quant_params_path,
        'mode': mode
    }

    # evaluate Rsum on the validation set
    output_dir = os.path.join(dir, 'testing')
    create_folder(output_dir)
    evaluate_post_training(eval_params, Hval, bestModel, path=output_dir)
    print(f'eval done')

