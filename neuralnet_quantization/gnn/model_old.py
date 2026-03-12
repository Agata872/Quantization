import tensorflow as tf
from tensorflow import keras
import numpy as np
from math import comb

"""---layers---"""
class Gnn_layer(keras.layers.Layer):
    def __init__(self, hidden_features, input_dim, **kwargs):
        """
        :param hidden_features: nr of hidden features in this layer, also known as d_l
        :param input_dim: nr of hidden features in the previous layer, also known as d_{l-1}
        """
        super(Gnn_layer, self).__init__(**kwargs)

        #store some constants
        self.hidden_features = hidden_features
        self.input_dim = input_dim
        shape = (self.hidden_features, self.input_dim)

        #create trainable weights for the layer
        self.Wself =self.add_weight(
            shape=shape,
            initializer= tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        self.Wm = self.add_weight(
            shape=shape,
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        self.Wk = self.add_weight(
            shape=shape,
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )

    def get_config(self):
        config = super(Gnn_layer, self).get_config()
        config.update({
            'hidden_features': self.hidden_features,
            'input_dim': self.input_dim,
        })
        return config

    def aggregate_antennas(self, inputs):
        """
        :param inputs: bs x M x K x nr_features
        :return: message_m (bs x M x nr_features) message passing of all edges connected to each antenna node
        """
        message_m = tf.math.reduce_sum(inputs, axis=2) #message passing of neighbors of m for all m (bs x M x nr_features)
        return message_m

    def aggregate_users(self, inputs):
        """
        :param inputs: bs x M x K x nr_features
        :return: message_k (bs x K x nr_features) message passing of all edges connected to each user node
        """
        message_k = tf.math.reduce_sum(inputs, axis=1) #message passing of neighbors of k for all k (bs x K x nr_features)
        return message_k

    def call(self, inputs):
        """
        :param inputs: shape: (bs x M x K x nr_features)
        :return:
        """
        """todo
        - make seperate message functions
        - check list and reshape stuff 
        - try to find a vectorized version and write a test for it 
        - 
        """

        #aggregate step (maybe change to mean for proper normalization)
        message_m = self.aggregate_antennas(inputs) #(bs x M x nr_features)
        message_k = self.aggregate_users(inputs) #(bs x K x nr_features)

        #update step
        #zl = tf.Variable(tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.hidden_features)))
        #zl_list = []
        zl_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        idx = 0
        for m in range(tf.shape(inputs)[1]):
            for k in range(tf.shape(inputs)[2]):
                x = tf.matmul(self.Wself, inputs[:, m, k, :], transpose_b=True) + \
                    tf.matmul(self.Wm, message_m[:, m, :], transpose_b=True) + \
                    tf.matmul(self.Wk, message_k[:, k, :], transpose_b=True)
                x = tf.transpose(x) #bx x d_l
                #zl_list.append(x) #store all intermediates in a list
                zl_ta = zl_ta.write(idx, x)
                idx += 1


                #desired behaviour, but asingment not available in tensorflow => use list as workaround
                #zl[:, m, k, :] = tf.matmul(self.Wself, inputs[:, m, k, :], transpose_b=True) #+ self.Wm @ message_m[m] + self.Wk @ message_k[k]

        debug = False
        #create tensor from tensor array
        Zl_old = zl_ta.stack()

        if debug:
            print('tensor array shape: ')
            print(Zl_old.shape)
            #create tensor from list (M*K x bs x d_l)
            #Zl_old = tf.stack(zl_list)
            print('before transposed shape')
            print(Zl_old.shape)

        Zl = tf.transpose(Zl_old, perm=[1, 0, 2]) # change chape to (bs x M*K x d_l)

        if debug:
            print('transposed shape')
            print(Zl.shape)
            for i in range(8):
                assert tf.math.reduce_all(tf.equal(Zl_old[i, :, :], Zl[:, i, :])), 'transpose was incorrect'

        #reshape tensor into the correct shape of (bs x M x K x d_l)
        Zl = tf.reshape(Zl, (tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.hidden_features)) #bs x M x K x d_l

        if debug:
            print('reshaped')
            print(Zl.shape)

            #test if the reshape was correct
            cnt = 0
            for m in range(tf.shape(inputs)[1]):
                for k in range(tf.shape(inputs)[2]):
                    assert tf.math.reduce_all(tf.equal(Zl[:, m, k, :], Zl_old[cnt])), f'reshaped was not correct: {tf.equal(Zl[:, m, k, :], Zl_old[cnt])}'
                    #print(tf.equal(Zl[:, m, k, :], zl_list[cnt]))
                    cnt += 1
            print('reshape assert succesfull')
        return Zl

class Pwr_norm(keras.layers.Layer):
    def __init__(self, Pt, **kwargs):
        super(Pwr_norm, self).__init__(**kwargs)
        self.Pt = Pt

    def get_config(self):
        config = super().get_config()
        config.update({
            'Pt': self.Pt
        })
        return config

    def call(self, inputs):
        #cast precoding matrix back to complex numbers
        Wre = inputs[:, :, :, 0]
        Wim = inputs[:, :, :, 1]
        W = tf.complex(Wre, Wim)  # bs x K x M
        print('shape W in pwr norm')
        print(tf.shape(W))
        print(f"input power: {tf.math.real(tf.norm(W, ord='fro', axis=(1, 2)) ** 2)}")

        #compute pwr normalization cte
        alpha = tf.math.sqrt(self.Pt / tf.math.real(tf.norm(W, ord='fro', axis=(1, 2)) ** 2))
        alpha = tf.reshape(alpha, (tf.shape(alpha)[0], 1, 1, 1))
        # print(f'alpha: {alpha}')

        # scale the output with alpha
        output = alpha * inputs
        Wreo = output[:, :, :, 0]
        Wimo = output[:, :, :, 1]
        Wo = tf.complex(Wreo, Wimo)
        print(f"output power: {tf.math.real(tf.norm(Wo, ord='fro', axis=(1, 2)) ** 2)}")
        # print(f'ouput: {output}')
        return output

"""---models---"""
class Precoding_gnn(tf.keras.Model):
    def __init__(self, M=4, K=2, nr_features=128, pt=1, name='precoding_GNN', **kwargs):
        super(Precoding_gnn, self).__init__(name=name, **kwargs)
        #store nr of antennas, users, features
        self.M = M
        self.K = K
        self.nr_features = nr_features
        self.pt = pt

        #generate the layers
        self.layer1 = Gnn_layer(self.nr_features, 2) #input layer (bs x M x K x 2)
        self.layer2 = Gnn_layer(self.nr_features, self.nr_features) #(bs x M x K x nr_features)
        self.layer3 = Gnn_layer(self.nr_features, self.nr_features) #(bs x M x K x nr_features)
        self.layer4 = Gnn_layer(self.nr_features, self.nr_features) #(bs x M x K x nr_features)
        self.layer5 = Gnn_layer(2, self.nr_features) # output layer (bs x M x K x 2)
        self.pwr_norm = Pwr_norm(self.pt)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        x = tf.nn.relu(x)
        x = self.layer3(x)
        x = tf.nn.relu(x)
        x = self.layer4(x)
        x = tf.nn.relu(x)
        x = self.layer5(x)
        out = self.pwr_norm(x)
        return out

    def get_config(self):
        config = super(Precoding_gnn, self).get_config()
        config.update({
            'M': self.M,
            'K': self.K,
            'nr_features': self.nr_features,
            'pt': self.pt
        })
        return config


"""---loss functions---"""
def polynomial_loss(Bs, noisevar, Gw=True):
    """
    :param Bs:  polynomialcoefficients [B1, B3, B5, ..., B2N+1]
    :param noisevar:
    :param Gw: if False use the approximation G(w)= I
    :return:
    """
    def loss(H_batch_train, y_pred):
        """
        :param H_batch_train: batch of training channels (bs x M x K x 2)
        :param y_pred: output of the neuralnet (bs x M x K x 2)
        :return:
        """
        #(2N+1)th order polynomial
        N = Bs.shape[0] - 1 #order of the polynomial [B1, B3, B5, ..., B2N+1]
        print('order: ')
        print(2*N + 1)
        print('Bs: ')
        print(Bs)

        #reconstruct the channel and cast to complex
        Hre = H_batch_train[:, :, :, 0]
        Him = H_batch_train[:, :, :, 1]
        H = tf.complex(Hre, Him)

        #reconstruct the precoding matrix and cast to complex
        Wre = y_pred[:, :, :, 0]
        Wim = y_pred[:, :, :, 1]
        W = tf.complex(Wre, Wim) #bs x M x K

        #compute the Bussgang gain matrix G(W)
        Cx = W @ tf.transpose(W, conjugate=True, perm=(0, 2, 1))

        #compute H^T W
        Htrans = tf.transpose(H, perm=(0, 2, 1))

        #compute G(w) if we dont neglect it
        if Gw:
            G = compute_Gw(Bs, Cx)
            HGW = Htrans @ G @ W
        else: #simplify G = I
            HGW = Htrans @ W

        #get desired signal
        desiredsignal = tf.math.abs(tf.linalg.diag_part(HGW)) ** 2

        #get user interference: sum_k |HW|^2 - desiredsignal
        userInterference = tf.reduce_sum(tf.math.abs(HGW) ** 2, axis=2) - desiredsignal

        #compute the disotrtion power
        Hconj = tf.math.conj(H)
        Ce = compute_ce(N, Bs, Cx)
        distortion = tf.cast(tf.linalg.diag_part(Htrans @ Ce @ Hconj), dtype=tf.float32)

        #compute snidr per user
        snidr = desiredsignal / (userInterference + distortion + noisevar)

        #rate per user
        R = tf.math.log(1 + snidr) / tf.math.log(2.0)

        #sumrate
        Rsum = tf.reduce_sum(R, axis=1)

        return -Rsum
    return loss

def compute_ce(N, Bs, Cx):
    Ce = 0
    for n in range(1, N+1):
        Ln = compute_Ln(n, N, Bs, Cx)
        Ce += Ln @ Cx * tf.cast(tf.abs(Cx)**(2*n), dtype=tf.complex64) @ tf.transpose(Ln, perm=(0, 2, 1), conjugate=True)
    return Ce

def compute_Ln(n, N, Bs, Cx):
    """
    :param n:
    :param N:
    :param Bs: array of polynomial coeffs [B1, B3, B5, ..., B2N+1]
    :param M:
    :return:
    """
    M = Cx.shape[1]
    diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
    Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])

    #compute Ln =
    Ln = 0
    for l in range(n, N+1):
        Ln += tf.cast(comb(l, n) * np.math.factorial(l+1), dtype=tf.complex64) * Bs[l] * \
              tf.cast(Im, dtype=tf.complex64) * diagCx**(l-n)
    Ln *= (1 / np.sqrt(n+1))
    return Ln

def compute_Gw(Bs, Cx):
    """
    :param Bs: poly coefs [b1, b3, b5, ..., b2N+1]
    :param Cx: input covariance matrix
    :return: bussgang gain matrix
    """
    M = Cx.shape[1]
    diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
    Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])

    #compute G(W) = B1 Im + (n+1)! B3 Im diag(Cx) + ... + (N+1)! B2N+1 diag(Cx)^N
    Gw = tf.cast(np.math.factorial(1), dtype=tf.complex64) * Bs[0] * tf.cast(Im, dtype=tf.complex64) #1e order term
    for n in range(1, Bs.shape[0]):#higher order terms
        Gw += tf.cast(np.math.factorial(n+1), dtype=tf.complex64) * Bs[n] * tf.cast(Im, dtype=tf.complex64) @ diagCx**n
    return Gw


#code for debugging
"""
if __name__ == '__main__':
    #create dummy data
    M, K = 4, 2
    Hsingle = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Hsingle2 = np.array([[5, 5], [6, 6], [7, 7], [8, 8]])
    H = np.stack([Hsingle, Hsingle2], axis=-1)
    H = H[np.newaxis, ...]
    H = tf.convert_to_tensor(H, dtype=float)
    bs = 3
    H = tf.reshape(tf.range(M*K*2*bs, dtype=float), (bs, M, K, 2))
    dl = 128
    print(f'H shape: {H.shape}')

    # #create a gnn layer for debugging
    # gnn = Gnn_layer(dl, 2)
    # print(gnn.get_config())
    #
    # #run the layer on the dummy data
    # y = gnn(H)

    #create model
    model = Precoding_gnn(M=M, K=K, nr_features=32)
    print(model.get_config())

    #add optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    #add loss todo change to custom loss
    my_loss = tf.keras.losses.MeanSquaredError()

    #compile model
    model.compile(loss=my_loss, optimizer=optimizer, run_eagerly=False)

    #fit
    history = model.fit(H, H, batch_size=1, epochs=1)
    model.summary()

    print('debug')
"""