import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from utils.utils import C2R, logparams, create_folder, logmodel, get_data
from gnn.losses import polynomial_loss
from datetime import datetime
from gnn.model import GNN_layer, Pwr_norm_gnn, Efficient_GNN_layer
from gnn.testing import evaluate
from gnn.naming import get_name
from gnn.activations import get_activation
from gnn.callbacks import Grad_tb_callback, monitor_weights_and_grads
import sys
import matplotlib as mpl
# mpl.use('Qt5Agg') #interactive plot



def get_efficient_GNN(M, K, feature_size, Pt, layers, activation='lrelu', batchnorm=False, skip_connection=False, aggregation='sum'):
    input_feature_size = 2

    # construct model
    model = keras.Sequential()
    model.add(keras.Input(shape=(M * K, 2)))
    layer_name = get_name('efficient_gnn_layer', nr=0, activation_string=activation)
    model.add(
        Efficient_GNN_layer(input_feature_size, feature_size, M, K, nr=0, act=activation, name=layer_name,
                  aggregation=aggregation))  # add first layer
    # model.add(keras.layers.LeakyReLU())
    for l in range(layers - 2):
        layer_name = get_name('efficient_gnn_layer', nr=l + 1, activation_string=activation)
        model.add(Efficient_GNN_layer(feature_size, feature_size, M, K, nr=l + 1, act=activation,
                            name=layer_name, aggregation=aggregation))
        # model.add(keras.layers.LeakyReLU())
        if batchnorm:
            model.add(keras.layers.BatchNormalization())
    layer_name = get_name('efficient_gnn_layer', nr=layers)
    model.add(Efficient_GNN_layer(feature_size, 2, M, K, nr=layers, name=layer_name, aggregation=aggregation))
    model.add(Pwr_norm_gnn(Pt, M, K))

    return model


def get_GNN(M, K, feature_size, Pt, layers, activation='lrelu', batchnorm=False, skip_connection=False, aggregation='sum'):
    input_feature_size = 2

    if skip_connection == 'sum':
        #todo check nr layers is even otherwise resnet doesnt work (unless we suport it)
        input = keras.Input((M * K, 2))
        # add first layer
        layer_name = get_name('gnn_layer', nr=0, activation_string=activation)
        x = GNN_layer(input_feature_size, feature_size, M, K, nr=0, act=activation, name=layer_name,
                      aggregation=aggregation)(input)

        res_blocks = int((layers - 2) / 2)
        for i in range(res_blocks):
            #copy x to later add it as a skip connectino
            x_skip = x


            #layer 1 of resblock
            layer_name = get_name('gnn_layer', nr=(i*2) + 1, activation_string=activation, skip=None)
            x = GNN_layer(feature_size, feature_size, M, K, nr=(i*2) + 1, act=activation, skip=None,
                      name=layer_name, aggregation=aggregation)(x)

            # layer 2 of resblock
            layer_name = get_name('gnn_layer', nr=(i * 2 + 1) + 1, skip=None)
            x = GNN_layer(feature_size, feature_size, M, K, nr=(i * 2+1) + 1, skip=None,
                      name=layer_name, aggregation=aggregation)(x)

            #add residual
            x = keras.layers.Add()([x, x_skip])
            x = get_activation(activation)(x)

        #output layer
        layer_name = get_name('gnn_layer', nr=layers-1)
        x = GNN_layer(feature_size, 2, M, K, nr=layers-1, name=layer_name, aggregation=aggregation)(x)

        #pwr norm layer
        out = Pwr_norm_gnn(Pt, M, K)(x)

        #create model
        model = keras.models.Model(input, out)

    else:
        # construct model
        model = keras.Sequential()
        model.add(keras.Input(shape=(M * K, 2)))
        layer_name = get_name('gnn_layer', nr=0, activation_string=activation)
        model.add(
            GNN_layer(input_feature_size, feature_size, M, K, nr=0, act=activation, name=layer_name,
                      aggregation=aggregation))  # add first layer
        # model.add(keras.layers.LeakyReLU())
        for l in range(layers - 2):
            layer_name = get_name('gnn_layer', nr=l + 1, activation_string=activation, skip=skip_connection)
            model.add(GNN_layer(feature_size, feature_size, M, K, nr=l + 1, act=activation, skip=skip_connection,
                                name=layer_name, aggregation=aggregation))
            # model.add(keras.layers.LeakyReLU())
            if batchnorm:
                model.add(keras.layers.BatchNormalization())
        layer_name = get_name('gnn_layer', nr=layers)
        model.add(GNN_layer(feature_size, 2, M, K, nr=layers, name=layer_name, aggregation=aggregation))
        model.add(Pwr_norm_gnn(Pt, M, K))

    return model


def train(training_params, sim_params, mainfolder='stored_models'):
    """
    :param training_params:
    :param sim_params:
    :return: train for a certain simulation and training set up
    """
    # unpack simulation parameters
    M = sim_params['M']
    K = sim_params['K']
    Pt = sim_params['Pt']
    noise_var = sim_params['noise_var']
    Bs = sim_params['Bs']
    ch_model = sim_params['channelmodel']
    order = 2 * (Bs.shape[0] - 1) + 1
    print(f'order: {order}')

    # unpack training parameters
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
    batchnorm = training_params['batchnorm']
    skipconnections = training_params['skips']
    monitor_weights_and_grads_callback = training_params['monitor_weights_and_grads']
    tensorboard_gradmon = training_params['tensorboard']
    aggregation = training_params['aggregation']

    # create results folder
    subfolder = f"M_{M}_K_{K}_nvar_{noise_var}"
    output_dir = os.path.join(
        os.getcwd(),
        mainfolder,
        subfolder,
        f"{training_params['note']}M_{M}_K_{K}_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}"
    )

    # store the simulation params
    create_folder(output_dir)
    logparams(os.path.join(output_dir, 'mimo_params.json'), sim_params, output_dir=output_dir)
    logparams(os.path.join(output_dir, 'train_params.json'), training_params)

    # generate channels
    datafolder = os.path.join(os.getcwd(), 'datasets')
    Htrain, Hval, Htest = get_data(M, K, Ntr, Nval, Nte, datafolder, channelmodel=ch_model)

    #plot one channel of first user to see the type
    plt.plot(np.abs(Htest[0, :, 0]), label='|h_k|')
    plt.plot(np.unwrap(np.angle(Htest[0, :, 0])), label='arg h_k')
    plt.show()

    # convert complex numbers to real numbers bs x M x K => bs x M x K x 2
    Htrainre = C2R(Htrain)
    Htestre = C2R(Htest)
    Hvalre = C2R(Hval)

    # todo: dont need this for CCNN
    # flatten for GNN (not needed for CCNN) bs x M x K x 2 => bs x MK x 2
    Htrain_flat = tf.reshape(Htrainre, [Ntr, -1, 2])
    Hval_flat = tf.reshape(Hvalre, [Nval, -1, 2])

    # make tf datasets
    # #todo: note that the input to the nn is flat while the label aka input to the lossfunction ground truth is not flat! check what is needed when using CCNN
    train_dataset = tf.data.Dataset.from_tensor_slices((Htrain_flat, Htrainre))
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((Hval_flat, Hvalre))
    val_dataset = val_dataset.batch(batch_size)

    # define the GNN model
    if layertype == 'gnn':
        model = get_GNN(M, K, feature_size, Pt, layers, activation, batchnorm=batchnorm, skip_connection=skipconnections,
                        aggregation=aggregation)
        print(model.summary())
    elif layertype == 'efficient_gnn':
        model = get_efficient_GNN(M, K, feature_size, Pt, layers, activation, batchnorm=batchnorm,
                        skip_connection=skipconnections,
                        aggregation=aggregation)
        print(model.summary())
    else:
        assert False, f'invalid layer type: {layertype}'

    # select optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # compile the model
    loss = polynomial_loss(Bs, noise_var, Gw=True)
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=eager_mode)
    print(f'model loss function: {model.loss.__name__}')

    # add callbacks todo check if we can use this, see unfolding model
    save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "model.h5"),
        monitor='val_loss',
        verbose=0,
        save_best_only=True
    )

    log = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training.log'))

    callbacks = [save, log]

    if earlystopping:
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        callbacks.append(earlystop)
    if reduce_lr:
        reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_delta=0.001,
                                                        verbose=1)
        callbacks.append(reducelr)
    if monitor_weights_and_grads_callback and skipconnections is None: #doesnt support skip connections
        # create the folder to store things in
        create_folder(os.path.join(output_dir, 'weight_grad_log'))
        print_weights = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: monitor_weights_and_grads(model, val_dataset, epoch,
                                                                       os.path.join(output_dir, 'weight_grad_log')))
        callbacks.append(print_weights)
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
    print(f'evaluating on Hval for poly coeffs: {Bs}')

    if layertype == 'gnn':
        # load model (to get best weights)
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'GNN_layer': GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': polynomial_loss
            }
        )
    elif layertype == 'efficient_gnn':
        # load model
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'Efficient_GNN_layer': Efficient_GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': polynomial_loss
            }
        )

    print(bestModel.summary())

    # evaluate the neural nets performance
    eval_params = {
        'nr_snr_points': 24,
        'Pt': Pt,
        'M': M,
        'K': K,
        'Bs': Bs,
        'zf': True,
        'lin': True,
        'title_add': 'val set after training',
        'pa': 'poly',
        'dpd': False
    }
    evaluate(eval_params, Hval, bestModel, path=output_dir)

    print(f'done evaluating on Hval for poly coeffs: {Bs}')
    print(f'*******************************************DONE TRAINING FOR:*******************************************')
    print(f'{sim_params} - {training_params}')


if __name__ == '__main__':
    # simulation setup
    channel_model = 'los' # 'iid'
    M = 64  # nr tx antennas
    K = 2  # nr users
    Pt = M  # total power
    snr_tx = 20  # in db
    noise_var = Pt / (10 ** (snr_tx / 10))
    Bs = np.array([0.93952922, -0.2295791 - 0.16381764 * 1j,
                   0.01606119 + 0.01539195 * 1j])  # np.array([1, -0.14251214]) #set poly coeffs to determinethe order [b1, b3, b5,..., b2N+1]

    # training params
    layer_type = 'gnn' #'gnn'
    Ntr = 200000 # nr training data todo set to 200000
    Nval = 2000
    Nte = 10000 # nr test data
    nr_layers = 4
    hidden_features = 128
    batch_size = 64
    epochs = 50  # 50
    lr = 5e-3
    activation = 'lrelu'
    bn = False
    aggregation_string = 'mean' #'sum'
    note = f'GNN'

    # training settings
    eager_mode = False
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
        'snr_tx': snr_tx,
        'noise_var': noise_var,
        'Bs': Bs,
        'channelmodel': channel_model
    }

    training_params = {
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
        'batchnorm': bn,
        'aggregation': aggregation_string,
        'note': note
    }

    # pa params for -9 , -7.5, -6, -4.5, -3, -1.5, 0 db backof for a third order model
    backoffs = [-9, -7.5, -6, -4.5, -3, -1.5, 0]
    different_backoffs = [np.array([1, -0.01993004 - 0.01079656 * 1j]), np.array([1, -0.03068619 - 0.01884522 * 1j]),
                          np.array([1, -0.04226415 - 0.02490772 * 1j]), np.array([1, -0.05612979 - 0.03005297 * 1j]),
                          np.array([1, -0.07781605 - 0.0401193 * 1j]), np.array([1, -0.11728841 - 0.0650147 * 1j]),
                          np.array([1, -0.15905536 - 0.07924839 * 1j])
                          ]

    # pa params 9th order ibo = -3db
    backoff = -3
    # pa_params = np.array([1, -6.61430363e-02 - 2.06358303e-02 * 1j,  1.87106273e-03 + 7.78243704e-04 * 1j,
    #  -2.20630955e-05 - 1.03064329e-05 * 1j,  9.12461695e-08 + 4.53884349e-08 * 1j])

    # 11th order -3db IBO range 0-sqrt(pt)
    pa_params = np.array([1, -1.11143930e-01-6.30816977e-02*1j, 6.60156653e-03+5.47141526e-03*1j,
                -1.91451680e-04-1.86610370e-04*1j,  2.62822435e-06+2.80380833e-06*1j,
                -1.36811147e-08-1.54579691e-08*1j])

    # 11th order parameters for different backoffs  for -9 , -7.5, -6, -4.5, -3, -1.5, 0 db backof
    pa_params_diff_ibo = [
        np.array([1, -4.38184836e-02 - 1.01466832e-01 *1j  , 1.50490437e-03 + 8.42208488e-03 *1j,
         - 3.13452827e-05 - 2.81868627e-04*1j,  3.49967293e-07 + 4.20633310e-06*1j,
         - 1.59432984e-09 - 2.31868139e-08*1j]) ,
        np.array([1, -5.79334438e-02-9.36769411e-02*1j,  2.39315994e-03+7.94859107e-03*1j,
         -5.57663136e-05-2.69264129e-04*1j,  6.65066314e-07+4.04837957e-06*1j,
         -3.14808144e-09-2.24280442e-08*1j]),
        np.array([1, -7.50994886e-02-8.42352484e-02*1j,  3.66782506e-03+7.26453523e-03*1j,
         -9.54049052e-05-2.48371067e-04*1j,  1.22703316e-06+3.75613932e-06*1j,
         -6.13183499e-09-2.08924283e-08*1j]),
        np.array([1, -9.35828409e-02-7.41305601e-02*1j,  5.16172165e-03+6.46522185e-03*1j,
         -1.44481282e-04-2.22483069e-04*1j,  1.94963213e-06+3.37874265e-06*1j,
         -1.00752209e-08-1.88479147e-08*1j]),
        np.array([1, -1.11143930e-01 - 6.30816977e-02 * 1j, 6.60156653e-03 + 5.47141526e-03 * 1j,
          -1.91451680e-04 - 1.86610370e-04 * 1j, 2.62822435e-06 + 2.80380833e-06 * 1j,
          -1.36811147e-08 - 1.54579691e-08 * 1j]),
        np.array([1, -1.29033190e-01-5.49758824e-02*1j,  8.21176444e-03+4.85204392e-03*1j,
         -2.48588087e-04-1.68144990e-04*1j,  3.52215545e-06+2.56527492e-06*1j,
         -1.88139985e-08-1.43562319e-08*1j]),
        np.array([1, -1.44473655e-01-4.67375592e-02*1j,  9.58442261e-03+4.13617338e-03*1j,
         -2.96362436e-04-1.43570171e-04*1j,  4.25309097e-06+2.19271142e-06*1j,
         -2.29128062e-08-1.22805850e-08*1j])
    ]
    ibo = [-9, -7.5, -6, -4.5, -3, -1.5, 0]#[-9, -7.5, -6, -4.5, -3, -1.5, 0]

    # snr values at witch we train P_t/noisevar
    snr_tx_set = np.array([20])  # np.array([-10, 0, 10, 20, 30]) #in db
    noise_var_set = Pt / (10 ** (snr_tx_set / 10))
    print(f'noisevarset: {noise_var_set}')

    # number of users and antennas
    Ms = [64]  # [64]#[64]
    Ks = [4] #[1, 6, 8]

    # model params
    layers = [8]#[8]  # [3, 8, 16]
    features = [256]  #[128] # [32, 64, 128, 256, 512]
    activations = ['lrelu']
    skips = [None] #['sum', learned_per_layer', 'mlp', 'gnn', None]  # , 'mlp', 'gnn', 'learned_per_layer']
    # folder to store results
    folder_for_test = f'stored_models_los_channel'
    aggregation_string = 'mean'

    # start training for different configurations
    for l in layers:
        for f in features:
            for act in activations:
                for k in range(len(Ks)):
                    for m in range(len(Ms)):
                        for skip in skips:
                            #for i, pacoefs in enumerate(pa_params_diff_ibo):
                            print(f'----------------------OUTER LOOP GNN k {Ks[k]} , m {Ms[m]}---------------------')
                            sim_params['Pt'] = Ms[m]  # m changes => total transmit power changes P_t = M
                            noise_var_set = sim_params['Pt'] / (
                                        10 ** (snr_tx_set / 10))  # update noise vars with new Pt
                            print(
                                f"    - new Pt: {sim_params['Pt']} - snr set: {10 * np.log10(sim_params['Pt'] / noise_var_set)}")

                            # loop over different SNRs to train at
                            for nvar in noise_var_set:
                                print(f'training at an IBO of {ibo[4]} with PA: {pa_params_diff_ibo[4]}')

                                # update the simulation params
                                sim_params['Bs'] = pa_params_diff_ibo[4]
                                sim_params['K'] = Ks[k]
                                sim_params['M'] = Ms[m]
                                sim_params['noise_var'] = nvar
                                sim_params['snr_tx'] = 10 * np.log10(sim_params['Pt'] / sim_params['noise_var'])

                                training_params['layers'] = l
                                training_params['dl'] = f
                                training_params['activation'] = act
                                training_params['skips'] = skip
                                training_params['aggregation'] = aggregation_string
                                training_params['note'] = f'{channel_model}_{ibo[4]}_IBO_{layer_type}_{aggregation_string}_aggregation_skips_{skip}_{l}' \
                                                          f'_layers_{f}_features_{act}'


                                #set seed the same initializations when comparing eg activations, etc
                                tf.random.set_seed(42)
                                # start training
                                train(training_params, sim_params, mainfolder=folder_for_test)
"""
Higher order tests: 

- test 1: 9th order pa -3db IBO - mean aggregation - accidentally trained with the efficient GNN layer!!
layers: [4, 8] - features: [64, 128, 256] 
=> works well for k=2


- test 2: 9th order pa -3db IBO - mean aggregation - this time with normal layer!!
layers: [4, 8] - features: [64, 128, 256] 
=> works well for k=2

- test 3: train with 500000 instead of 200k

-test 4: 500k train + different snrs
all previous tests SNR=20dB
now: SNR = [0, 10, 30]

"""




"""
test1: M=64 K=[1, 2, 4, 6] layers=[4, 6, 8, 16] features=[32, 64, 128, 256, 512] activations=['lrelu', 'elu', 'tanh']
results: 
- LReLu otherwise NaNs because of the final normalization layer <0 => relu gives 0 => Nan 
- more than 6 layers => NaN (probably exploding or vanashing gradient problem)

test 1.5 added batch norm
- doesnt help for vanishing gradient

test 2: added skips
=> still no gradients => when checking tensorboard the gradients seem to be all very small for deeper nets (8 layers)
=> but doesnt seem to be 'vanishing' gradients as they are small in every layer!!

test 3: normal skips instead of linear interpolation skips

test 4:  normal skips instead of linear interpolation skips but explude k=1 as there seems to be a problem there
=> maybe sum for aggregation is not properly defined for K = 1?
16 layers, features : [64, 128, 256] 

test 5: mean aggregation instead of sum
 layers = [4, 6, 8, 16] - features = [64, 128, 256] 
 => mean aggregation fixes instability issues and performs better than mean aggregation
  but for 16 layers we still get some NaN values (=> add skips?)
  
test 6: mean aggregation + skip connections?
=> sum skips and gnn gating skips
=> slightly better than no skips
best: 
k=1 =>8 layers 256 features (128 features very close) sum skip
k=2 =>8 layers 256 featuers sum skip | slightly better: 8 layers 64 features gnn skip
k=4 =>8 layers 128 features gnn skip | close: 8 layers 256 features sum skip 


test 7: mean aggregation + skips
=> learned gating per layer vs mlp gating


test 8: efficient gnn layer
=> new layer based on taking the mean of the full feature matrix as a 'message' instead of looking at the neighbors
(the edge itself is in this mean now, maybe remove it later.)
result: 
- only works for k=1, sucks for the rest

"""