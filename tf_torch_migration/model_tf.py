import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utils import rayleigh_channel_MU
from activations import get_activation


class GNN_layer(layers.Layer):
    def __init__(self, input_feature_size, feature_size, M, K, nr=0, skip=None, act=None, aggregation='sum', **kwargs):
        #layer_name = get_name('gnn_layer', nr, act, skip)
        super(GNN_layer, self).__init__(**kwargs)
        self.M = M
        self.K = K
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.nr = nr
        self.skip = skip
        self.activation_string = act
        self.activation = get_activation(self.activation_string)
        self.aggregation = aggregation

        self.edge_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wedge_{self.nr}'
        )

        self.m_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wm_{self.nr}'
        )

        self.k_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wk_{self.nr}'
        )

        if skip == 'learned_per_layer':
            assert self.feature_size == self.input_feature_size, f'could not add a skip connection as the feature size'\
                                                                 f' of the two consecutive layers was not equal. Got: '\
                                                                f'{self.input_feature_size} and {self.feature_size} '
            self.alpha1 = self.add_weight(
                shape=(self.input_feature_size, ),
                initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.2),
                trainable=True,
                name=f'alpha_{self.nr}'
            )
        elif skip == 'gnn':
            assert self.feature_size == self.input_feature_size, f'could not add a skip connection as the feature size'\
                                                                 f' of the two consecutive layers was not equal. Got: '\
                                                                f'{self.input_feature_size} and {self.feature_size} '
            self.alpha_weight = self.add_weight(
                shape=(self.feature_size, self.input_feature_size),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
                name=f'Walpha_{self.nr}'
            )

            self.alpha_m_weight = self.add_weight(
                shape=(self.feature_size, self.input_feature_size),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
                name=f'Walpha_m_{self.nr}'
            )

            self.alpha_k_weight = self.add_weight(
                shape=(self.feature_size, self.input_feature_size),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
                name=f'Walpha_k_{self.nr}'
            )

        elif skip == 'mlp':
            assert self.feature_size == self.input_feature_size, f'could not add a skip connection as the feature size'\
                                                                 f' of the two consecutive layers was not equal. Got: '\
                                                                f'{self.input_feature_size} and {self.feature_size} '
            self.alpha_dense = tf.keras.layers.Dense(self.feature_size, activation='sigmoid')

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_feature_size': self.input_feature_size,
            'feature_size': self.feature_size,
            'M': self.M,
            'K': self.K,
            'skip': self.skip,
            'act': self.activation_string,
            'nr': self.nr,
            'aggregation': self.aggregation
            #'name': self.layer_name
        })
        return config

    def call(self, inputs):
        """
        :param inputs: bs x MK x input_feature_size
        :return: outputs: bs x MK x feature_size
        """
        #todo maybe switch reduce_sum aggregation to reduce_mean aggregation?

        batch_size = tf.shape(inputs)[0]

        # update edge features: bs x feature_size x input_feature_size @ bs x input_feature_size x MK
        Z_l_edge = self.edge_weights @ tf.transpose(inputs, perm=[0, 2, 1])

        #transpose bs x featur_size x MK to bs x MK x feature_size
        Z_l_edge = tf.transpose(Z_l_edge, perm=[0, 2, 1])

        #aggregation
        #bs x MK x input_feature_size to bs x M x K x input_feature_size
        edges = tf.reshape(inputs, (batch_size, self.M, self.K, self.input_feature_size))

        if self.aggregation == 'sum':
            #aggregate to the antenna nodes aka for each antenna sum the edges connected to it
            m_message = tf.reduce_sum(edges, axis=2) #sum accross K dimension => bs x M x input_feature_size

            #aggregate to the user nodes aka for each user sum the edges connected to it
            k_message = tf.reduce_sum(edges, axis=1) #sum accross the M dimension => bs x K x input_feature_size
        elif self.aggregation == 'mean':
            # aggregate to the antenna nodes aka for each antenna sum the edges connected to it
            m_message = tf.reduce_mean(edges, axis=2)  # sum accross K dimension => bs x M x input_feature_size

            # aggregate to the user nodes aka for each user sum the edges connected to it
            k_message = tf.reduce_mean(edges, axis=1)  # sum accross the M dimension => bs x K x input_feature_size
        else:
            tf.Assert(False, [f'invalid aggregation operation: {self.aggregation}'])

        #multiply with a learned weight matrix
        Wm_m_message = self.m_weights @ tf.transpose(m_message, perm=[0, 2, 1]) #bs x feature_size x M
        Wk_k_message = self.k_weights @ tf.transpose(k_message, perm=[0, 2, 1]) #bs x feature_size x K
        Wm_m_message = tf.transpose(Wm_m_message, perm=[0, 2, 1]) #bs x M x feature_size
        Wk_k_message = tf.transpose(Wk_k_message, perm=[0, 2, 1]) #bs x K x feature size

        #resturcture m_message as
        # m_message[:, 0, :], ... ,  m_message[:, 0, :], m_message[:, 1, :] , ... m_message[:, M-1, :]
        m_message_expanded = tf.repeat(Wm_m_message, repeats=self.K, axis=1) #bs x KM x feature_size

        #resturcture k_message as
        # k_message[:, 0, :], k_message[:, 1, :], ..., k_message[:, K-1, :] ,k_message[:, 0, :], ... k_message[:, K-1, :]
        k_message_expanded = tf.tile(Wk_k_message, multiples=[1, self.M, 1]) #bs x KM x feature_size

        #sum all three parts
        z_l = Z_l_edge + m_message_expanded + k_message_expanded

        #activation
        z_l = self.activation(z_l)

        #add skip connection if need
        if self.skip == 'learned_per_layer':
            alpha1 = tf.math.sigmoid(self.alpha1)
            alpha2 = 1 - alpha1
            z_l = alpha1 * z_l + alpha2 * inputs

        if self.skip == 'gnn': #compute alpha1 as the output of a 1 layer GNN:
            #todo normalization might be an issue => if we are in the tails of the sigmoid no gradient to learn with
            #Walpha @ z^{(l-1)
            Walpha_z_l_1 = self.alpha_weight @ tf.transpose(inputs, perm=[0, 2, 1])

            #transpose bs x featur_size x MK to bs x MK x feature_size
            Walpha_z_l_1 = tf.transpose(Walpha_z_l_1, perm=[0, 2, 1])

            # W_alpha,m @ M_N(m) and W_alpha,k @ M_N(k)
            Walpha_m_message = self.alpha_m_weight @ tf.transpose(m_message, perm=[0, 2, 1])  # bs x feature_size x M
            Walpha_k_message = self.alpha_k_weight @ tf.transpose(k_message, perm=[0, 2, 1])  # bs x feature_size x K
            Walpha_m_message = tf.transpose(Walpha_m_message, perm=[0, 2, 1])  # bs x M x feature_size
            Walpha_k_message = tf.transpose(Walpha_k_message, perm=[0, 2, 1])  # bs x K x feature size

            # resturcture m_message as
            # m_message[:, 0, :], ... ,  m_message[:, 0, :], m_message[:, 1, :] , ... m_message[:, M-1, :]
            alpha_m_message_expanded = tf.repeat(Walpha_m_message, repeats=self.K, axis=1)  # bs x KM x feature_size

            # resturcture k_message as
            # k_message[:, 0, :], k_message[:, 1, :], ..., k_message[:, K-1, :] ,k_message[:, 0, :], ... k_message[:, K-1, :]
            alpha_k_message_expanded = tf.tile(Walpha_k_message, multiples=[1, self.M, 1])  # bs x KM x feature_size

            #input alpha gnn: Walpha @ z^{(l-1) +  W_alpha,m @ M_N(m) + W_alpha,k @ M_N(k)
            input_alpha_gnn = Walpha_z_l_1 + alpha_m_message_expanded + alpha_k_message_expanded
            alpha1 = tf.math.sigmoid(input_alpha_gnn)
            alpha2 = 1 - alpha1

            #interpolation update
            z_l = alpha1 * z_l + alpha2 * inputs


        if self.skip == 'mlp': #compute alpha based on a single dense layer
            alpha1 = self.alpha_dense(inputs)
            alpha2 = 1 - alpha1

            #interpolation update
            z_l = alpha1 * z_l + alpha2 * inputs

        return z_l #, self.edge_weights, self.k_weights, self.m_weights #extra outputs for debugging

class Pwr_norm_gnn(layers.Layer):
    def __init__(self, Pt, M, K, **kwargs):
        super(Pwr_norm_gnn, self).__init__()
        self.Pt = Pt
        self.M = M
        self.K = K

    def get_config(self):
        config = super().get_config()
        config.update({
            'Pt': self.Pt,
            'M': self.M,
            'K': self.K
        })
        return config

    def call(self, inputs):
        """
        :param input: bs x MK x 2 (Real)
        :return: bs x M x K x 2 (Real)
        """
        batch_size = tf.shape(inputs)[0]

        #reshape to bs x M x K x 2
        input_reshaped = tf.reshape(inputs, (batch_size, self.M, self.K, 2))

        #cast back to complex numbers
        Wre = input_reshaped[:, :, :, 0]
        Wim = input_reshaped[:, :, :, 1]
        W = tf.complex(Wre, Wim)  # bs x K x M
        #print(f"input power: {tf.math.real(tf.norm(W, ord='fro', axis=(1, 2)) ** 2)}")

        #compute alpha
        # Wh = tf.transpose(W, perm=(0, 2, 1), conjugate=True)
        # WhW = tf.matmul(Wh, W)
        alpha = tf.math.sqrt(self.Pt / tf.math.real(tf.norm(W, ord='fro', axis=(1, 2)) ** 2))
        #alpha = tf.math.sqrt(tf.cast(self.Pt, dtype=tf.float32)) / tf.math.sqrt(tf.math.real(tf.linalg.trace(WhW)))
        alpha = tf.reshape(alpha, (tf.shape(alpha)[0], 1, 1, 1))
        #print(f'alpha: {alpha}')

        #scale the output with alpha
        output = alpha * input_reshaped #todo this does not broadcast properly fix it (see previous code)
        # Wreo = output[:, :, :, 0]
        # Wimo = output[:, :, :, 1]
        # Wo = tf.complex(Wreo, Wimo)
        # print(f"output power: {tf.math.real(tf.norm(Wo, ord='fro', axis=(1, 2)) ** 2)}")
        #print(f'ouput shape : {tf.shape(output)}')
        return output

#unit test for debugging
if __name__ == '__main__':
    """ check multiple layers """
    #create dummy data
    M, K = 64, 2
    bs = 3

    H = np.zeros((bs, M, K), dtype=complex)
    for i in range(bs):
        H[i, :, :] = rayleigh_channel_MU(M, K)
    H = tf.reshape(tf.range(M*K*2*bs, dtype=float), (bs, M, K, 2))
    print(f'H shape: {H.shape}, represents bs x M x K x 2')

    dl = 129
    gnn1 = GNN_layer(2, dl, M, K)
    gnn2 = GNN_layer(dl, dl, M, K, skip='mlp')
    gnn3 = GNN_layer(dl, dl, M, K)
    gnn4 = GNN_layer(dl, 2, M, K)
    pwrnorm = Pwr_norm_gnn(M, M, K)

    H_flat = tf.reshape(H, [bs, -1, 2])
    y = H_flat
    for i in range(50):
        y = tf.reshape(y, [bs, -1, 2])
        y = gnn1(y)
        y1 = gnn2(y)
        y2 = gnn3(y1)
        y3 = gnn4(y2)
        y = pwrnorm(y3)
        print(f"--------------------------------------------------------- iter {i} ---------------------------------------------")
        print(f'{y=}')


    """ check one layer """
    # #create dummy data
    # M, K = 5, 4
    # bs = 3
    # H = tf.reshape(tf.range(M*K*2*bs, dtype=float), (bs, M, K, 2))
    # print(f'H shape: {H.shape}, represents bs x M x K x 2')
    #
    # #create a gnn layer for debugging
    # dl = 64
    # gnn = GNN_layer(2, dl, M, K)
    #
    # #run the layer on the dummy data
    # H_flat = tf.reshape(H, [bs, -1, 2])
    # y, Wedge, Wk, Wm = gnn(H_flat)
    # testbs = tf.shape(y)[0]
    # #w = tf.reshape(y, (testbs, M, K, dl))
    #
    # #todo sanity check in numpy using for loops
    # H = H.numpy()
    # zl = np.zeros((bs, M, K, dl))
    # Wedge = Wedge.numpy()
    # Wk = Wk.numpy()
    # Wm = Wm.numpy()
    # #Wedge = np.ones((dl, 2)) #np.eye(dl, 2)
    # # Wm = np.ones((dl, 2)) #np.eye(dl, 2)
    # # Wk = np.ones((dl, 2)) #np.eye(dl, 2)
    #
    # for i, Hi in enumerate(H): #loop over batches Hi M x K x 2
    #     print(f'Hi shape: {Hi.shape}')
    #     m_message = np.sum(Hi, axis=1)
    #     k_message = np.sum(Hi, axis=0)
    #     for m in range(M):
    #         for k in range(K):
    #             zl[i, m, k, :] = Wedge @ H[i, m, k, :] + Wm @ m_message[m, :] + Wk @ k_message[k, :]
    #
    # diff = zl - w.numpy()
    # print(f'{diff=}')
    # print(f'sum: {np.sum(diff)}')
    print('debug')
