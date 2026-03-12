import keras.losses
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.quantization import quantize_nonuniform
from utils.utils import create_folder
import tensorflow as tf
from tensorflow.keras import Sequential, layers

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
import numpy as np


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]  # check dimension

        # np.random.randint returns ints from [low, high) !
        idx = np.random.randint(self.X.shape[0], size=shape[0])

        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.

    # Example

    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```


    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas

    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):

        self.output_dim = output_dim

        # betas is either initializer object or float
        if isinstance(betas, Initializer):
            self.betas_initializer = betas
        else:
            self.betas_initializer = Constant(value=betas)

        self.initializer = initializer if initializer else RandomUniform(
            0.0, 1.0)

        super().__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=self.betas_initializer,
                                     # initializer='ones',
                                     trainable=True)

        super().build(input_shape)

    def call(self, x):

        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
        H = tf.transpose(C-tf.transpose(x))  # matrix of differences
        return tf.exp(-self.betas * tf.math.reduce_sum(H**2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':
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
    nr_data = 200000
    alpha = 3
    x_in_re = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in_im = np.linspace(-alpha*np.sqrt(varx), alpha*np.sqrt(varx), nr_data)
    x_in = x_in_re + 1j* x_in_im

    # quantize
    y = quantize_nonuniform(x_in, thresholds, outputlevels)

    # only do real part (cause imag is the same)
    y_out_re = np.real(y)
    x_in_re = np.real(x_in)

    # # create dataset
    # dataset = tf.data.Dataset.from_tensor_slices((x_in_re[:, np.newaxis],
    #                                               y_out_re[:, np.newaxis]))
    x_data = x_in_re[:, np.newaxis]
    y_data = y_out_re[:, np.newaxis]


    # create model
    neurons = 8
    model = Sequential()
    model.add(keras.Input(shape=(1)))
    model.add(RBFLayer(256, initializer=InitCentersRandom(x_data), betas=2.0))
    model.add(layers.Dense(1, use_bias=False))
    print(model.summary())

    # select optimizer
    optimizer = tf.keras.optimizers.Adam()

    # loss
    loss = keras.losses.MeanSquaredError()

    # compile the model
    model.compile(loss=loss, optimizer=optimizer)

    # add callback to save checkpoints of the model
    output_dir = os.path.join(os.getcwd(), 'MLP_quantizer')
    create_folder(output_dir)
    save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "model.h5"),
        monitor='val_loss',
        verbose=0,
        save_best_only=True
    )
    callbacks = [save]

    reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_delta=0.0002,
                                                    verbose=1)
    callbacks.append(reducelr)

    # start training
    history = model.fit(
        x=x_data,
        y=y_data,
        validation_data=(x_data[::100], y_data[::100]),
        batch_size=32,
        epochs=10,
        callbacks=callbacks
    )


    # plot history
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('MSE')
    plt.show()

    # test the model
    y_test = model(x_data)
    plt.plot(x_in_re, y_test.numpy(), label='nn')
    plt.plot(x_in_re, y_out_re, label='dac')
    plt.title('tf')
    plt.legend()
    plt.show()



    print('done')

