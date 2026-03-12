import keras.losses
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils import create_folder
import tensorflow as tf
from tensorflow.keras import Sequential, layers

if __name__ == '__main__':


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
    neurons = 256
    model = Sequential()
    model.add(keras.Input(shape=(1)))
    model.add(layers.Dense(neurons, activation='relu', use_bias=False))
    model.add(layers.Dense(1, use_bias=True))
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

