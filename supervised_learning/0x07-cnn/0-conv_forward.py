#!/usr/bin/env python3
""" a function  that performs forward propagation over a pooling layer """
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """ Function Convolution Forward"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2)
        ph = int(ph)
        pw = np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2)
        pw = int(pw)
        outputH = int(((h_prev - kh + (2 * ph)) / sh) + 1)
        outputW = int(((w_prev - kw + (2 * pw)) / sw) + 1)
        convol = np.zeros((m, outputH, outputW, c_new))
        padImages = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                           mode='constant', constant_values=0)
        padImagesVector = np.arange(0, m)
        for i in range(outputH):
            for j in range(outputW):
                for k in range(c_new):
                    i_s = i * sh
                    j_s = j * sw
                    convol[padImagesVector, i, j, k] = activation((
                        np.sum(np.multiply(padImages[
                            padImagesVector,
                            i_s: i_s + kh, j_s: j_s + kw],
                                           W[:, :, :, k]),
                               axis=(1, 2, 3))) + b[0, 0, 0, k])
    return convol
