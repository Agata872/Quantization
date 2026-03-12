import numpy as np
import os
from datetime import datetime
from utils.utils import create_folder, logparams, logmodel, C2R, get_data
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from model import get_GNN, GNN_layer, Pwr_norm_gnn
from callbacks import monitor_weights_and_grads, Grad_tb_callback
from losses import quant_loss_uncorrelated_Rqq, polynomial_loss, quant_numerical, quant_numerical_MLP_DAC
from MIMO_sims.Rsum_all import Rsum_Bussgang_Rx
import tikzplotlib

def evaluate(eval_params, Hval, bestModel, path=None):
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

    # get snr values to compute Rsum at
    snr_points = np.linspace(-30, 35, nr_snr_points)

    # convert channel to correct shape for the nn
    H_real = C2R(Hval)
    H_flat = tf.reshape(H_real, [Hval.shape[0], -1, 2])

    # run GNN
    y_pred = bestModel.predict(H_flat)

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








def train(training_params, sim_params, mainfolder='stored_models'):
    # unpack simulation parameters
    M = sim_params['M']
    K = sim_params['K']
    Pt = sim_params['Pt']
    bits = sim_params['bits']
    noise_var = sim_params['noise_var']
    ch_model = sim_params['channelmodel']
    quant_params_path = sim_params['quant_params_path']

    # unpack training parameters
    loss_type = training_params['loss']
    polyparams = training_params['polyparams']
    layertype = training_params['layer_type']
    Ntr = training_params['Nr_train']
    Nval = training_params['Nr_val']
    Nte = training_params['Nr_test']
    lr = training_params['lr']
    layers = training_params['layers']
    eager_mode = training_params['eager_mode']
    earlystopping = training_params['early_stop']
    reduce_lr = training_params['reduce_lr']
    feature_size = training_params['dl']
    activation = training_params['activation']
    monitor_weights_and_grads_callback = training_params['monitor_weights_and_grads']
    tensorboard_gradmon = training_params['tensorboard']
    aggregation = training_params['aggregation']

    # create results folder
    subfolder = f"M_{M}_K_{K}_nvar_{noise_var}"
    output_dir = os.path.join(
        os.getcwd(),
        mainfolder,
        subfolder,
        f"{training_params['note']}bits_{bits}_M_{M}_K_{K}_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}"
    )

    # store the simulation params
    create_folder(output_dir)
    logparams(os.path.join(output_dir, 'mimo_params.json'), sim_params, output_dir=output_dir)
    logparams(os.path.join(output_dir, 'train_params.json'), training_params)

    # generate or load channels
    datafolder = os.path.join(os.getcwd(), 'datasets')
    Htrain, Hval, Htest = get_data(M, K, Ntr, Nval, Nte, datafolder, channelmodel=ch_model)

    # convert complex numbers to real numbers bs x M x K => bs x M x K x 2
    Htrainre = C2R(Htrain)
    Htestre = C2R(Htest)
    Hvalre = C2R(Hval)

    # flatten for GNN (not needed for CCNN) bs x M x K x 2 => bs x MK x 2
    Htrain_flat = tf.reshape(Htrainre, [Ntr, -1, 2])
    Hval_flat = tf.reshape(Hvalre, [Nval, -1, 2])

    # make tf datasets
    # #todo: note that the input to the nn is flat while the label aka input to the lossfunction ground truth is not flat!
    train_dataset = tf.data.Dataset.from_tensor_slices((Htrain_flat, Htrainre))
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((Hval_flat, Hvalre))
    val_dataset = val_dataset.batch(batch_size)

    # get the model
    model = get_GNN(M, K, feature_size, Pt, layers, activation, aggregation=aggregation)
    print(model.summary())

    # select optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # compile the model
    if loss_type == 'quant_loss_uncorrelated_Rqq':
        beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))
        alpha = 1 - beta
        loss = quant_loss_uncorrelated_Rqq(alpha, noise_var)
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=eager_mode)
        print(f'model loss function: quant loss uncorrelated Rqq')
    elif loss_type == 'polynomial_loss':
        path = os.path.dirname(quant_params_path)
        path = os.path.join(path, polyparams)
        path = os.path.join(path, f'betas_{bits}_bits.npy')
        Bs = np.load(path)
        print(f'loading poly params for {bits} bits')
        print(f'{Bs=}')
        loss = polynomial_loss(Bs, noise_var, Gw=True)
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=eager_mode)
        print(f'model loss function: {model.loss.__name__}')
    elif loss_type == 'quant_numerical':
        beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))
        alpha = 1 - beta
        thresholds = np.load(os.path.join(quant_params_path, f'{bits}bits_thresholds.npy'))
        outputlevels = np.load(os.path.join(quant_params_path, f'{bits}bits_outputlevels.npy'))
        loss = quant_numerical(alpha,
                               noise_var,
                               tf.convert_to_tensor(thresholds, dtype=tf.float32),
                               tf.convert_to_tensor(outputlevels, dtype=tf.float32)
                               )
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=eager_mode)
        print(f'model loss function: {model.loss.__name__}')
        print(f'model loss function: poly loss')
    elif loss_type == 'quant_numerical_MLP_DAC':
        path = r'D:\thomas.feys\Quantization\precoding_quantization\checks\MLP_quantizer\4_hidden_layers_16_neurons'
        #r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\checks\MLP_quantizer\1_hidden_layer_64_neurons_20epochs_200ktrain'
        quant_model = keras.models.load_model(os.path.join(path, 'model.h5'))
        beta = np.load(os.path.join(quant_params_path, f'{bits}bits_nmse.npy'))
        alpha = 1 - beta
        loss = quant_numerical_MLP_DAC(alpha, noise_var, quant_model)
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=eager_mode)
        print(f'model loss function: {model.loss.__name__}')
        print(f'model loss function: quant numerical MLP DAC loss')

    # add callback to save checkpoints of the model
    save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "model.h5"),
        monitor='val_loss',
        verbose=0,
        save_best_only=True
    )

    # add callback to save the loss of the model
    log = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training.log'))

    callbacks = [save, log]

    # add optional callbacks
    if earlystopping:
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        callbacks.append(earlystop)
    if reduce_lr:
        reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_delta=0.001,
                                                        verbose=1)
        callbacks.append(reducelr)

    if tensorboard_gradmon:
        # create logs folder
        create_folder(os.path.join(output_dir, 'tb_logs'))
        # Define the TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(output_dir, 'tb_logs'))
        # Define the custom gradients callback
        gradients_callback = Grad_tb_callback(log_dir=os.path.join(output_dir, 'tb_logs'), histogram_freq=1,
                                              val_dataset=val_dataset)
        callbacks.append(tensorboard_callback)
        callbacks.append(gradients_callback)

    # log the model summary
    logmodel(os.path.join(output_dir, 'model.json'), model)

    # start training
    print(f'*******************************************START TRAINING FOR:*******************************************')
    print(f'{sim_params} - {training_params}')
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    # plot loss progress
    plt.plot(-np.array(history.history['loss']))
    plt.plot(-np.array(history.history['val_loss']))
    plt.title(f'sum rate M: {M} K: {K}')
    plt.ylabel('R')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, 'model_loss.pdf'))
    plt.show()

    print(f'training done for M:{M}, K:{K}')
    print(f'evaluating on Hval for {bits} bits')

    # load model (to get best weights)
    if loss_type == 'quant_loss_uncorrelated_Rqq':
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'GNN_layer': GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': quant_loss_uncorrelated_Rqq
            }
        )
    elif loss_type == 'polynomial_loss':
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'GNN_layer': GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': polynomial_loss
            }
        )
    elif loss_type == 'quant_numerical':
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'GNN_layer': GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': quant_numerical
            }
        )
    elif loss_type == 'quant_numerical_MLP_DAC':
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'GNN_layer': GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': quant_numerical_MLP_DAC
            }
        )

    # evaluate the neural nets performance
    eval_params = {
        'nr_snr_points': 24,
        'Pt': Pt,
        'M': M,
        'K': K,
        'bits': bits,
        'zf': True,
        'inf_res': True,
        'title_add': 'val set after training',
        'quant_params_path': quant_params_path
    }

    # evaluate Rsum on the validation set
    evaluate(eval_params, Hval, bestModel, path=output_dir)
    print(f'eval done')


if __name__ == '__main__':
    # simulation setup
    channel_model = 'iid' #'los'
    M = 32  # nr tx antennas
    K = 1  # nr users
    Pt = M  # total power
    bits = 2 # nr of bits
    snr_tx = 20  # in db
    noise_var = Pt / (10 ** (snr_tx / 10))

    # path to quantizer parameters
    varx = 0.5
    quant_params_path = r'D:\thomas.feys\Quantization\precoding_quantization\non-uniform-quant-params'
    #r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non-uniform-quant-params'
    #r'D:\thomas.feys\Quantization\precoding_quantization\non-uniform-quant-params'
    quant_params_path = os.path.join(quant_params_path, f'Gaussian_var_{varx}', 'numerical')


    # training params
    loss = 'quant_numerical_MLP_DAC' #'quant_numerical' #'quant_loss_uncorrelated_Rqq' 'polynomial_loss' 'quant_numerical'
    polyparms = 'poly_params_fitrange_4sigma'
    layer_type = 'gnn'
    Ntr = 100000 #500 000 nr training data (for final test)
    Nval = 1000
    Nte = 10000 #10000 nr test data
    nr_layers = 6
    hidden_features = 64
    batch_size = 64
    epochs = 10
    lr = 5e-3
    activation = 'lrelu'
    aggregation_string = 'mean' #'sum'
    note = f'GNN_more_symbols_'

    # training settings
    eager_mode = True #todo set to false
    earlystopping = False
    reduce_lr = True
    monitor_weight_and_grad_manually = False
    tensorboard = False

    # todo replace with params json file that is loaded
    # put all the params in a dictionary to store it
    sim_params = {
        'M': M,
        'K': K,
        'Pt': Pt,
        'bits': bits,
        'snr_tx': snr_tx,
        'noise_var': noise_var,
        'channelmodel': channel_model,
        'quant_params_path': quant_params_path
    }

    training_params = {
        'loss': loss,
        'polyparams': polyparms,
        'layer_type': layer_type,
        'Nr_train': Ntr,
        'Nr_val': Nval,
        'Nr_test': Nte,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'reduce_lr': reduce_lr,
        'early_stop': earlystopping,
        'eager_mode': eager_mode,
        'monitor_weights_and_grads': monitor_weight_and_grad_manually,
        'tensorboard': tensorboard,
        'layers': nr_layers,
        'dl': hidden_features,
        'activation': activation,
        'aggregation': aggregation_string,
        'note': note
    }

    if loss == 'polynomial_loss':
        results_folder = f'stored_models_{loss}_{polyparms}'
    else:
        results_folder = f'stored_models_{loss}'


    bits = np.arange(1, 3, 1).tolist()
    users = [1, 2, 4]
    for k in users:
        for b in bits:
            sim_params['bits'] = b
            sim_params['K'] = k
            print(f'training with bits {b} and {k} users')
            train(training_params, sim_params, mainfolder=results_folder)
