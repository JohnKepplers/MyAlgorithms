__author__ = 'Rybkin & Kravchenko'

import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX
print(floatX)
from keras.layers.recurrent import Recurrent, LSTM
from keras import backend


def update_controller(self, inp, h_tm1, M):
    x = T.concatenate([inp, M], axis=-1)
    if len(h_tm1) == 2:
        if hasattr(self.lstm, "get_constants"):
            BW, BU = self.lstm.get_constants(x)
            h_tm1 += (BW, BU)
    _, h = self.lstm.step(x, h_tm1)

    return h


def circulant(leng, n_shifts):
    # wicked tensor
    eye = np.eye(leng)
    shifts = range(n_shifts // 2, -n_shifts // 2, -1)
    C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    return theano.shared(C.astype(theano.config.floatX))


def re_norm(x):
    return x / (x.sum(axis=1, keepdims=True))


def soft_max(x):
    wt = x.flatten(ndim=2)
    w = T.nnet.softmax(wt)
    return w.reshape(x.shape)


def cosine_similarity(M, k):
    dot = (M * k[:, None, :]).sum(axis=-1)
    nM = T.sqrt((M ** 2).sum(axis=-1))
    nk = T.sqrt((k ** 2).sum(axis=-1, keepdims=True))
    return dot / (nM * nk)


class NeuralTuringMachine(Recurrent):
    def __init__(self, output_dim, memory_size, shift_range=3,
                 init='glorot_uniform', inner_init='orthogonal',
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.n_slots = memory_size[1]
        self.m_length = memory_size[0]
        self.shift_range = shift_range
        self.init = init
        self.inner_init = inner_init

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(NeuralTuringMachine, self).__init__(**kwargs)

    def build(self, input_shape):
        input_leng, input_dim = input_shape[1:]
       # self.input = T.tensor3()

        self.lstm = LSTM(
            input_dim=input_dim + self.m_length,
            input_length=input_leng,
            output_dim=self.output_dim, init=self.init,
            forget_bias_init='zero',
            inner_init=self.inner_init)

        self.lstm.build(input_shape)

        # initial memory, state, read and write vecotrs
        self.M = theano.shared((.001 * np.ones((1,)).astype(floatX)))
        self.init_h = backend.zeros((self.output_dim))
        self.init_wr = self.lstm.init((self.n_slots,))
        self.init_ww = self.lstm.init((self.n_slots,))

        # write
        self.W_e = self.lstm.init((self.output_dim, self.m_length))  # erase
        self.b_e = backend.zeros((self.m_length))
        self.W_a = self.lstm.init((self.output_dim, self.m_length))  # add
        self.b_a = backend.zeros((self.m_length))

        # get_w  parameters for reading operation
        self.W_k_read = self.lstm.init((self.output_dim, self.m_length))
        self.b_k_read = self.lstm.init((self.m_length,))
        self.W_c_read = self.lstm.init((self.output_dim, 3))
        self.b_c_read = backend.zeros((3))
        self.W_s_read = self.lstm.init((self.output_dim, self.shift_range))
        self.b_s_read = backend.zeros((self.shift_range))  # b_s lol! not intentional

        # get_w  parameters for writing operation
        self.W_k_write = self.lstm.init((self.output_dim, self.m_length))
        self.b_k_write = self.lstm.init((self.m_length,))
        self.W_c_write = self.lstm.init((self.output_dim, 3))  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c_write = backend.zeros((3))
        self.W_s_write = self.lstm.init((self.output_dim, self.shift_range))
        self.b_s_write = backend.zeros((self.shift_range))

        self.C = circulant(self.n_slots, self.shift_range)

        self.trainable_weights = self.lstm.trainable_weights + [
            self.W_e, self.b_e,
            self.W_a, self.b_a,
            self.W_k_read, self.b_k_read,
            self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read,
            self.W_k_write, self.b_k_write,
            self.W_s_write, self.b_s_write,
            self.W_c_write, self.b_c_write,
            self.M,
            self.init_h, self.init_wr, self.init_ww]

        self.init_c = backend.zeros((self.output_dim))
        self.trainable_weights = self.trainable_weights + [self.init_c, ]

    def read(self, w, M):
        return (w[:, :, None] * M).sum(axis=1)

    def write(self, w, e, a, M):
        Mtilda = M * (1 - w[:, :, None] * e[:, None, :])
        Mout = Mtilda + w[:, :, None] * a[:, None, :]
        return Mout

    def get_content_w(self, beta, k, M):
        num = beta[:, None] * cosine_similarity(M, k)
        return soft_max(num)

    def get_location_w(self, g, s, C, gamma, wc, w_tm1):
        wg = g[:, None] * wc + (1 - g[:, None]) * w_tm1
        Cs = (C[None, :, :, :] * wg[:, None, None, :]).sum(axis=3)
        wtilda = (Cs * s[:, :, None]).sum(axis=1)
        wout = re_norm(wtilda ** gamma[:, None])
        return wout

    def get_controller_output(self, h, W_k, b_k, W_c, b_c, W_s, b_s):
        k = T.tanh(T.dot(h, W_k) + b_k)  # + 1e-6
        c = T.dot(h, W_c) + b_c
        beta = T.nnet.relu(c[:, 0]) + 1e-4
        g = T.nnet.sigmoid(c[:, 1])
        gamma = T.nnet.relu(c[:, 2]) + 1.0001
        s = T.nnet.softmax(T.dot(h, W_s) + b_s)
        return k, beta, g, gamma, s

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim

    def call(self, x, mask = None):
        M_tm1, wr_tm1, ww_tm1 = mask[:3]
        # reshape
        M_tm1 = M_tm1.reshape((x.shape[0], self.n_slots, self.m_length))
        # read
        h_tm1 = mask[3:]
        k_read, beta_read, g_read, gamma_read, s_read = self.get_controller_output(
            h_tm1[0], self.W_k_read, self.b_k_read, self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read)
        wc_read = self.get_content_w(beta_read, k_read, M_tm1)
        wr_t = self.get_location_w(g_read, s_read, self.C, gamma_read,
                                   wc_read, wr_tm1)
        M_read = self.read(wr_t, M_tm1)

        # update controller
        h_t = update_controller(self, x, h_tm1, M_read)

        # write
        k_write, beta_write, g_write, gamma_write, s_write = self.get_controller_output(
            h_t[0], self.W_k_write, self.b_k_write, self.W_c_write,
            self.b_c_write, self.W_s_write, self.b_s_write)
        wc_write = self.get_content_w(beta_write, k_write, M_tm1)
        ww_t = self.get_location_w(g_write, s_write, self.C, gamma_write,
                                   wc_write, ww_tm1)
        e = T.nnet.sigmoid(T.dot(h_t[0], self.W_e) + self.b_e)
        a = T.tanh(T.dot(h_t[0], self.W_a) + self.b_a)
        M_t = self.write(ww_t, e, a, M_tm1)

        M_t = M_t.flatten(ndim=2)

        return h_t[0], [M_t, wr_t, ww_t] + h_t
