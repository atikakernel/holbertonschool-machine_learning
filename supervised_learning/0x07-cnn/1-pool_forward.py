#!/usr/bin/env python3
""" a function  that performs forward propagation over a pooling layer """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Function Convolution Forward"""
    m, h, w, c = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    out_h = int(((h-kh)/sh) + 1)
    out_w = int(((w-kw)/sw) + 1)
    convol = np.zeros((m, out_h, out_w, c))
    img = np.arange(m)

    for x in range(out_h):
        for y in range(out_w):
            if mode == 'max':
                convol[img, x, y] = (np.max(A_prev[img,
                                                   x*sh:(kh+(x*sh)),
                                                   y*sw:(kw+(y*sw))],
                                            axis=(1, 2)))
            if mode == 'avg':
                convol[img, x, y] = (np.mean(A_prev[img,
                                                    x*sh:(kh+(x*sh)),
                                                    y*sw:(kw+(y*sw))],
                                             axis=(1, 2)))
    return convol
