import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#from training import getdata_nonlinprec, ChannelSymbolsDataset

class GNN_layer_fast(nn.Module):
    def __init__(self, input_feature_size, output_feature_size, M, K, outputlayer=False):
        super().__init__()
        self.M = M
        self.K = K
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.outputlayer = outputlayer

        #define trainable weights
        self.Wedge = nn.Parameter(torch.zeros(self.input_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wedge)

        self.Wm = nn.Parameter(torch.zeros(self.input_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wm)

        self.Wk = nn.Parameter(torch.zeros(self.input_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wk)

        self.Wself_m = nn.Parameter(torch.zeros(self.input_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wself_m)

        self.Wself_k = nn.Parameter(torch.zeros(self.input_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wself_k)

        self.Wneigh_m = nn.Parameter(torch.zeros(self.output_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wneigh_m)

        self.Wneigh_k = nn.Parameter(torch.zeros(self.output_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wneigh_k)

    def forward(self, z_mk, z_m, z_k):
        """
        :param z_mk: input edge features (bs x MK x input feature size)
        :param z_m: input antenna node featuers (bs x M x input feature size)
        :param z_k: input user node features (bs x K x input feature size)
        :return:- z_mk_updated (bs x MK x output feature size)
                - z_m_updated (bs x M x output feature size)
                - z_k_updated (bs x K x output feature size)
        """
        bs = z_mk.shape[0]


        """update edge features (bs x output_feature_size x input_feature_size @ bs x input_feature_size x MK)"""
        # multiply weight with edge features
        Wz_mk = z_mk @ self.Wedge # bs x MK x output_feature_size

        # multiply weight with antenna features (OK checked!)
        Wz_m = z_m @ self.Wm  # bs x M x output_feature_size
        Wz_m_expanded = torch.repeat_interleave(Wz_m, repeats=self.K, dim=1) # expand to bs x M x K x output_feature_size

        # multiply weight with user features (OK checked!)
        Wz_k = z_k @ self.Wk # bs x K x output_feature_size
        Wz_k_expanded = torch.tile(Wz_k, dims=(1, 1, self.M, 1)) # expand to bs x MK x output_feature_size
        Wz_k_expaned_reshaped = torch.reshape(Wz_k_expanded, (bs, self.M*self.K, self.output_feature_size))
        # todo replace by squeeze

        # sum and take nonlinear activation
        z_mk_updated = F.leaky_relu(Wz_mk + Wz_m_expanded + Wz_k_expaned_reshaped)

        """message passing"""
        # bs x MK x input_feature_size to bs x M x K x input_feature_size
        edges = torch.reshape(z_mk_updated, (bs, self.M, self.K, self.output_feature_size))

        # message passing from antennas to users
        message_nk = torch.mean(edges, dim=1) # sum across M dimension => bs x K x output_feature_size

        # message passing from users to antennas
        message_nm = torch.mean(edges, dim=2) # sum across K dimension => bs x M x output_feature_size

        """ update node features"""
        # update antenna node features
        Wz_m_2 = z_m @ self.Wself_m # bs x M x output_feature_size
        W_M_m = message_nm @ self.Wneigh_m # bs x M x output_feature_size
        if self.outputlayer:
            z_m_updated = Wz_m_2 + W_M_m
        else:
            z_m_updated = F.leaky_relu(Wz_m_2 + W_M_m) # bs x M x output_feature_size

        # update user node features
        Wz_k_2 = z_k @ self.Wself_k # bs x K x output_feature_size
        W_M_k = message_nk @ self.Wneigh_k # bs x K x output_feature_size
        z_k_updated = F.leaky_relu(Wz_k_2 + W_M_k) # bs x K x output_feature_size

        return z_mk_updated, z_m_updated, z_k_updated

class GNN_layer(nn.Module):
    def __init__(self, input_feature_size, output_feature_size, M, K, outputlayer=False):
        super().__init__()
        self.M = M
        self.K = K
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.outputlayer = outputlayer

        #define trainable weights
        self.Wedge = nn.Parameter(torch.zeros(self.output_feature_size, self.input_feature_size))
        nn.init.xavier_uniform_(self.Wedge)

        self.Wm = nn.Parameter(torch.zeros(self.output_feature_size, self.input_feature_size))
        nn.init.xavier_uniform_(self.Wm)

        self.Wk = nn.Parameter(torch.zeros(self.output_feature_size, self.input_feature_size))
        nn.init.xavier_uniform_(self.Wk)

        self.Wself_m = nn.Parameter(torch.zeros(self.output_feature_size, self.input_feature_size))
        nn.init.xavier_uniform_(self.Wself_m)

        self.Wself_k = nn.Parameter(torch.zeros(self.output_feature_size, self.input_feature_size))
        nn.init.xavier_uniform_(self.Wself_k)

        self.Wneigh_m = nn.Parameter(torch.zeros(self.output_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wneigh_m)

        self.Wneigh_k = nn.Parameter(torch.zeros(self.output_feature_size, self.output_feature_size))
        nn.init.xavier_uniform_(self.Wneigh_k)

    def forward(self, z_mk, z_m, z_k):
        """
        :param z_mk: input edge features (bs x MK x input feature size)
        :param z_m: input antenna node featuers (bs x M x input feature size)
        :param z_k: input user node features (bs x K x input feature size)
        :return:- z_mk_updated (bs x MK x output feature size)
                - z_m_updated (bs x M x output feature size)
                - z_k_updated (bs x K x output feature size)
        """
        bs = z_mk.shape[0]


        """update edge features (bs x output_feature_size x input_feature_size @ bs x input_feature_size x MK)"""
        # multiply weight with edge features
        Wz_mk = self.Wedge @ torch.transpose(z_mk, 1, 2) # bs x output_feature_size x MK
        Wz_mk = torch.transpose(Wz_mk, 1, 2) # back to bs x MK x output_feature_size

        # multiply weight with antenna features (OK checked!)
        Wz_m = self.Wm @ torch.transpose(z_m, 1, 2) # bs x output_feature_size x M
        Wz_m = torch.transpose(Wz_m, 1, 2) # bs x M x output_feature_size
        Wz_m_expanded = torch.repeat_interleave(Wz_m, repeats=self.K, dim=1) # expand to bs x M x K x output_feature_size

        # multiply weight with user features (OK checked!)
        Wz_k = self.Wk @ torch.transpose(z_k, 1, 2) # bs x output_feature_size x K
        Wz_k = torch.transpose(Wz_k, 1, 2) # bs x K x output_feature_size
        Wz_k_expanded = torch.tile(Wz_k, dims=(1, 1, self.M, 1)) # expand to bs x MK x output_feature_size
        Wz_k_expaned_reshaped = torch.reshape(Wz_k_expanded, (bs, self.M*self.K, self.output_feature_size))

        # sum and take nonlinear activation
        z_mk_updated = F.leaky_relu(Wz_mk + Wz_m_expanded + Wz_k_expaned_reshaped)

        """message passing"""
        # bs x MK x input_feature_size to bs x M x K x input_feature_size
        edges = torch.reshape(z_mk_updated, (bs, self.M, self.K, self.output_feature_size))

        # message passing from antennas to users
        message_nk = torch.mean(edges, dim=1) # sum across M dimension => bs x K x input_feature_size

        # message passing from users to antennas
        message_nm = torch.mean(edges, dim=2) # sum across K dimension => bs x M x input_feature_size

        """ update node features"""
        # update antenna node features
        Wz_m_2 = self.Wself_m @ torch.transpose(z_m, 1, 2)  # bs x output_feature_size x M
        Wz_m_2 = torch.transpose(Wz_m_2, 1, 2) # back to bs x M x output_feature_size
        W_M_m = self.Wneigh_m @ torch.transpose(message_nm, 1, 2) # bs x output_feature_size x M
        W_M_m = torch.transpose(W_M_m, 1, 2) # back to bs x M x output_feature_size
        if self.outputlayer:
            z_m_updated = Wz_m_2 + W_M_m
        else:
            z_m_updated = F.leaky_relu(Wz_m_2 + W_M_m) # bs x M x output_feature_size

        # update user node features
        Wz_k_2 = self.Wself_k @ torch.transpose(z_k, 1, 2) # bs x output_feature_size x K
        Wz_k_2 = torch.transpose(Wz_k_2, 1, 2) # back to bs x K x output_feature_size
        W_M_k = self.Wneigh_k @ torch.transpose(message_nk, 1, 2) # bs x output_feature_size x K
        W_M_k = torch.transpose(W_M_k, 1, 2) # back to bs x K x output_feature_size
        z_k_updated = F.leaky_relu(Wz_k_2 + W_M_k) # bs x K x output_feature_size

        return z_mk_updated, z_m_updated, z_k_updated


class GNNmodel(torch.nn.Module):
    def __init__(self, M, K, nr_features, nr_hidden_layers, bits, tau, quantzation_levels, quantize=True, output_type='softmax_hard'):
        torch.nn.Module.__init__(self)
        self.M = M
        self.K = K
        self.dl = nr_features
        self.nr_hidden_layers = nr_hidden_layers
        self.bits = bits
        self.nr_out_levels = 2**(self.bits)
        self.tau = tau # todo adjust this value (potentially decay it during training)
        self.quantize = quantize
        self.quantzation_levels = quantzation_levels # size 2^bits
        self.output_type = output_type

        # define layers of GNN
        self.input_layer = GNN_layer_fast(2, self.dl, M, K)
        self.hidden_layers = nn.ModuleList()
        for l in range(nr_hidden_layers):
            self.hidden_layers.append(GNN_layer_fast(self.dl, self.dl, M, K))

        if self.quantize:
            self.output_layer = GNN_layer_fast(self.dl, 2*self.nr_out_levels, M, K, outputlayer=True)
        else:
            self.output_layer = GNN_layer_fast(self.dl, 2, M, K, outputlayer=True)


    def forward(self, H, s, x_init):
        """
        :param H: bs x M x K (complex)
        :param s: bs x K (complex)
        :param x_init: bs x M x 2 (real)
        :return: bs x M (complex)
        """
        bs = H.shape[0]

        # construct correct input shapes
        H_re = H.real
        H_imag = H.imag
        H_reshaped = torch.stack((H_re, H_imag), dim=-1)# bs x M x K x 2
        H_flat = torch.reshape(H_reshaped, (bs, self.M*self.K, 2))
        s_re = s.real
        s_imag = s.imag
        s_flat = torch.stack((s_re, s_imag), dim=-1)# bs x K x 2

        # forward pass
        z_mk, z_m, z_k = self.input_layer(H_flat, x_init, s_flat) # input layer
        for layer in self.hidden_layers: # hidden layers
            z_mk, z_m, z_k = layer(z_mk, z_m, z_k)
        z_mk, z_m, z_k = self.output_layer(z_mk, z_m, z_k) # output layer

        # quantize or not
        if self.quantize == False:
            # output shape: bs x M x 2
            # reshape to complex nr
            z_m_re = z_m[:, :, 0]
            z_m_imag = z_m[:, :, 1]
            z_m_complex = z_m_re + 1j * z_m_imag

        else:
            # output shape: bs x M x 2*nr_output_levels

            if self.output_type == 'gumbel_softmax_hard': # gumble softmax to sample the categorical distribution
                logits = torch.reshape(z_m, (-1, self.M, 2, self.nr_out_levels)) # bs x M x 2 x nr_out_levels
                softmax_out = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

            elif self.output_type == 'gumbel_softmax': # gumble softmax to sample the categorical distribution
                logits = torch.reshape(z_m, (-1, self.M, 2, self.nr_out_levels)) # bs x M x 2 x nr_out_levels
                softmax_out = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)

            elif self.output_type == 'softmax_hard':
                logits = torch.reshape(z_m, (-1, self.M, 2, self.nr_out_levels)) # bs x M x 2 x nr_out_levels
                # argmax in fwd pass, softmax in backwd pass
                y_soft = F.softmax(logits, dim=-1)
                #print(f'ysoft = {y_soft[0, :, :, :]}')
                index = torch.argmax(logits, dim=-1, keepdim=True)
                y_hard = torch.zeros_like(y_soft).scatter(-1, index, 1.0)
                #print(f'yhard = {y_hard[0, :, :, :]}')
                softmax_out = y_hard - y_soft.detach() + y_soft # only keeps the value of y_hard, and the gradient of y_soft

            elif self.output_type == 'softmax':
                logits = torch.reshape(z_m, (-1, self.M, 2, self.nr_out_levels)) # bs x M x 2 x nr_out_levels
                softmax_out = F.softmax(logits, dim=-1)

            # multiply with outputlevel vector to get true precoder outputs (desired out shape: bs x M x 2)
            output_levels = softmax_out * self.quantzation_levels
            output_levels_squeezed = torch.sum(output_levels,
                                               dim=-1)  # sum over last dim to remove it (only one 1 anyway)

            # cast to complex numbers (desired outshape: bs x M)
            re_parts = output_levels_squeezed[:, :, 0]
            im_parts = output_levels_squeezed[:, :, 1]
            z_m_complex = re_parts + 1j * im_parts

        return z_m_complex


class MLPmodel_noquant(torch.nn.Module):
    def __init__(self, M, K):
        torch.nn.Module.__init__(self)
        self.nr_neurons = 64
        self.M = M
        self.K = K
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * (self.M * self.K) + 2 * self.K,  self.nr_neurons ),
            nn.ReLU(),
            nn.Linear( self.nr_neurons,  self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, 2 * self.M)# an output for each antenna per re and im part of each antenna
        )

    def forward(self, H, s):
        """
        :param H: bs x M x K (complex)
        :param s: bs x K (complex)
        :return: bs x M (complex)
        """

        # flatten the input data
        Hflat = torch.flatten(H, start_dim=1)
        sflat = torch.flatten(s, start_dim=1)

        # get the real and imag parts
        Hflat_re = Hflat.real
        Hflat_imag = Hflat.imag
        sflat_re = sflat.real
        sflat_imag = sflat.imag

        # concat everything to get the input
        input = torch.cat((Hflat_re, Hflat_imag, sflat_re, sflat_imag), dim=-1).float()

        # pass throug network
        raw_output = self.layers(input)

        # construct complex vector
        re_parts = raw_output[:, 0:self.M]
        im_parts = raw_output[:, self.M:2 * self.M]
        output = re_parts + 1j * im_parts

        return output

class MLPmodel(torch.nn.Module):
    def __init__(self, M, K, bits, tau, quantzation_levels):
        torch.nn.Module.__init__(self)
        self.nr_neurons = 1024
        self.M = M
        self.K = K
        self.bits = bits
        self.nr_out_levels = 2**(self.bits)
        self.tau = tau # todo adjust this value (potentially decay it during training)
        self.quantzation_levels = quantzation_levels # size 2^bits
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * (self.M * self.K) + 2 * self.K,  self.nr_neurons ),
            nn.ReLU(),
            nn.Linear( self.nr_neurons,  self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, self.nr_neurons),
            nn.ReLU(),
            nn.Linear(self.nr_neurons, 2 * self.M * self.nr_out_levels), # an output for each possible outputlevel per re and im part of each antenna
        )

    def forward(self, H, s):
        """
        :param H: bs x M x K (complex)
        :param s: bs x K (complex)
        :return: bs x M (complex)
        """
        # flatten the input data
        Hflat = torch.flatten(H, start_dim=1)
        sflat = torch.flatten(s, start_dim=1)

        # get the real and imag parts
        Hflat_re = Hflat.real
        Hflat_imag = Hflat.imag
        sflat_re = sflat.real
        sflat_imag = sflat.imag

        # concat everything to get the input
        input = torch.cat((Hflat_re, Hflat_imag, sflat_re, sflat_imag), dim=-1).float()

        # pass throug network
        raw_output = self.layers(input)

        # reshape output from 2*M*nr_out_level to (bs x 2*M x nr_out_levels)
        logits = torch.reshape(raw_output, (-1, 2 * self.M, self.nr_out_levels))

        # take gumbel softmax
        #softmax_out = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        #softmax_out = torch.nn.functional.softmax(logits, dim=-1)


        # # alternative (argmax in fwd pass, softmax in backwd pass)
        y_soft = torch.nn.functional.softmax(logits, dim=-1)
        index = torch.argmax(logits, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter(-1, index, 1.0)
        softmax_out = y_hard - y_soft.detach() + y_soft # only keeps the value of y_hard, and the gradient of y_soft

        # todo alternative: sigmoid activation, per DAC bit => value>0.5 => bit = 1?


        # # alternative (argmax in fwd pass, softmax in backwd pass)
        # y_soft = torch.nn.functional.softmax(logits, dim=-1)
        # index = torch.argmax(logits, dim=-1, keepdim=True)
        # y_hard = torch.zeros_like(y_soft).scatter(-1, index, 1.0)
        # softmax_out = y_hard - y_soft.detach() + y_soft # only keeps the value of y_hard, and the gradient of y_soft

        # todo alternative: sigmoid activation, per DAC bit => value>0.5 => bit = 1?


        # multiply with outputlevel vector to get true precoder outputs (desired out shape: bs x 2M)
        output_levels = softmax_out * self.quantzation_levels
        output_levels_squeezed = torch.sum(output_levels, dim=-1) # sum over last dim to remove it (1 of the 2 is zero anyways)

        # cast to complex numbers (desired outshape: bs x M)
        re_parts = output_levels_squeezed[:, 0:self.M]
        im_parts = output_levels_squeezed[:, self.M:2*self.M]
        output = re_parts + 1j * im_parts

        return output


class SumRateLoss_generalized_Bussgang(nn.Module):
    def __init__(self):
        super(SumRateLoss_generalized_Bussgang, self).__init__()

    def forward(self, nn_outputs, H, s, noise_var):
        """
        :param nn_outputs: bs x M x nr_symbols (NN is ran once per symbol output is collected in this tensor)
        :param H: bs x M x K (channel matrix is kept constant across the symbols (for one computation of the loss))
        :param s: bs x K x nr_symbols
        :return: sumrate: bs
        """
        nr_symb = nn_outputs.shape[-1]

        # decompose y = Gs + q (G: bs x M x K)
        G_full = (nn_outputs @ torch.transpose(torch.conj(s), 1, 2)) / nr_symb  # = E(y s^H)
        #cov_s = (s @ torch.transpose(torch.conj(s), 1, 2)) / nr_symb # = E(s s^H) = diag(1)
        #G_full = G @ torch.linalg.inv(cov_s)
        q = nn_outputs - G_full @ s

        # some stuff we will reuse later
        HT = torch.transpose(H, 1, 2)
        HTG = HT @ G_full

        # intended sig
        inteded_sig = torch.abs(torch.diagonal(HTG, dim1=1, dim2=2))**2

        # user interference
        user_interference = torch.sum(torch.abs(HTG)**2, dim=-1) - inteded_sig

        # distortion
        Cq = (q @ torch.transpose(torch.conj(q), 1, 2)) / nr_symb # = E(qq^H)
        dist = torch.real(torch.diagonal(HT @ Cq @ torch.conj(H), dim1=1, dim2=2))

        # compute sindr
        sindr_general = inteded_sig / (user_interference + dist + noise_var)

        # compute rate per user
        rate_general = torch.log2(1 + sindr_general)  # bs x K

        # compute sum rate
        sumrate_general = torch.sum(rate_general, dim=-1)  # bs

        # take mean over all batches as the loss
        avg_sumrate_general = torch.mean(sumrate_general)





        # # todo remove: keep for now as sanity check
        # # compute r = H^T x
        # r = torch.transpose(H, 1, 2) @ nn_outputs # bs x K x nr_symbols
        #
        # # compute bussgang gain per user Bk
        # B = torch.mean(r * torch.conj(s), dim=-1) / torch.mean(torch.abs(s)**2, dim=-1) # bs x K
        #
        # # check how close to one E(|s|**2) is add or leave out the term depending on it
        # #check = torch.mean(torch.abs(s)**2, dim=-1) #=> ok if nr symbols per channel is large enough
        #
        # # compute distortion ter per user: E(|d_k|^2) = E(|rk|^2) - |Bk|^2 E(|sk|^2)
        # D = torch.mean(torch.abs(r)**2, dim=-1) - torch.abs(B)**2 * torch.mean(torch.abs(s)**2, dim=-1) # bs x K
        #
        # # compute SINDR_k
        # #sindr = (torch.abs(B)**2) / (D + noise_var) # bs x K
        # sindr = (torch.abs(B) ** 2 * torch.mean(torch.abs(s) ** 2, dim=-1)) / (D + noise_var)  # bs x K
        #
        # # compute rate per user
        # rate = torch.log2(1 + sindr) # bs x K
        #
        # # compute sum rate
        # sumrate = torch.sum(rate, dim=-1) # bs
        #
        # # take mean over all batches as the loss
        # avg_sumrate = torch.mean(sumrate)

        return -avg_sumrate_general

class SumRateLoss(nn.Module):
    def __init__(self):
        super(SumRateLoss, self).__init__()

    def forward(self, nn_outputs, H, s, noise_var):
        """
        :param nn_outputs: bs x M x nr_symbols (NN is ran once per symbol output is collected in this tensor)
        :param H: bs x M x K (channel matrix is kept constant across the symbols (for one computation of the loss))
        :param s: bs x K x nr_symbols
        :return: sumrate: bs
        """

        # compute r = H^T x
        r = torch.transpose(H, 1, 2) @ nn_outputs # bs x K x nr_symbols

        # compute bussgang gain per user Bk
        B = torch.mean(r * torch.conj(s), dim=-1) / torch.mean(torch.abs(s)**2, dim=-1) # bs x K

        # check how close to one E(|s|**2) is add or leave out the term depending on it
        #check = torch.mean(torch.abs(s)**2, dim=-1) #=> ok if nr symbols per channel is large enough

        # compute distortion ter per user: E(|d_k|^2) = E(|rk|^2) - |Bk|^2 E(|sk|^2)
        D = torch.mean(torch.abs(r)**2, dim=-1) - torch.abs(B)**2 * torch.mean(torch.abs(s)**2, dim=-1) # bs x K

        # compute SINDR_k
        #sindr = (torch.abs(B)**2) / (D + noise_var) # bs x K
        sindr = (torch.abs(B) ** 2 * torch.mean(torch.abs(s) ** 2, dim=-1)) / (D + noise_var)  # bs x K

        # compute rate per user
        rate = torch.log2(1 + sindr) # bs x K

        # compute sum rate
        sumrate = torch.sum(rate, dim=-1) # bs

        # take mean over all batches as the loss
        avg_sumrate = torch.mean(sumrate)

        return -avg_sumrate


# # testing
# M, K = 4, 2
# bits = 1
# Pt = M
# output_levels = np.sqrt(Pt/(2*M)) * torch.Tensor([-1, 1])
# tau = 1
# model = MLPmodel(M, K, bits, tau, output_levels)
# nr_features = 64
# nr_hidden_layers = 3
# GNN_model = GNNmodel(M, K, nr_hidden_layers, nr_hidden_layers, bits, tau, output_levels, quantize=True)
# snr_tx = 20  # in db
# noise_var = Pt / (10 ** (snr_tx / 10))
#
# # sim params
# Ntr = 1000
# Nval = 1000
# Nte = 1000
#
# # todo load/generate data
# nr_symbols_per_channel = 100
# datapath = r'D:\thomas.feys\Quantization\precoding_quantization\non_lin_precoding\datasets' #r'C:\Users\Thomas\OneDrive - KU Leuven\Documents\GitHub\Quantization\precoding_quantization\non_lin_precoding\datasets'
# Htrain, Hval, Htest, strain, sval, stest = getdata_nonlinprec(nr_symbols_per_channel, datapath, M, K, Ntr, Nval, Nte)
# trainset = ChannelSymbolsDataset(Htrain, strain, nr_symbols_per_channel=nr_symbols_per_channel)
# batch_size = 3
# training_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
#
# # test model
# outputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
# H, s = next(iter(training_dataloader))
# print(f'{H.shape} - {s.shape}')
#
# for sidx in range(s.shape[-1]):
#     outputs[:, :, sidx] = model(H, s[:, :, sidx])
#
# gnnoutputs = torch.zeros((batch_size, M, nr_symbols_per_channel), dtype=torch.complex64)
# for sidx in range(s.shape[-1]):
#     gnnoutputs[:, :, sidx] = GNN_model(H.type(torch.complex64), s[:, :, sidx].type(torch.complex64))
#
# print(outputs.shape)
# print(outputs)
#
# # test loss
# loss_fn = SumRateLoss()
# loss = loss_fn(outputs, H.type(torch.complex64), s.type(torch.complex64), noise_var)
# print(f'loss hape: {loss.shape}')
# print(f'{loss=}')
# #
# #
# # # test loss with zf precoding
# # H = H[0, :, :].numpy()
# # s = s[0, :, :].numpy()
# # Wzf = H.conj() @ np.linalg.inv(H.T @ H.conj())
# # norm = np.sqrt(Pt) / np.linalg.norm(Wzf, ord='fro')
# # Wzf *= norm
# #
# # # precode
# # x = Wzf @ s
# #
# # snr_txs = np.arange(-30, 30, 5.1)  # in db
# # print(f'{snr_txs=}')
# # noise_vars = Pt / (10 ** (snr_txs / 10))
# # print(f'{noise_vars=}')
# #
# # rsums = np.zeros_like(snr_txs)
# # for i, noise_var in enumerate(noise_vars):
# #     rsums[i] = -loss_fn(torch.from_numpy(x[None, :]).type(torch.complex64), torch.from_numpy(H[None, :, :]).type(torch.complex64),
# #                        torch.from_numpy(s[None, :]).type(torch.complex64), noise_var)
# # print(rsums)
# # plt.plot(snr_txs, rsums)
# # plt.show()
#
#
