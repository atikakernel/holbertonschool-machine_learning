#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs a valid convolurion on grayscale image"""
    m, h, w, c = images.shape
    kh, kw = kernel.shape

    output_h = int(((h - kh + (2 * padding_h)) / stride[0]) + 1)
    output_w = int(((w - kw + (2 * padding_w)) / stride[1]) + 1)
    output = np.zeros((m, output_h, output_w, c))

    images_arr = np.arange(0, m)

    for x in range(output_h):
        for y in range(output_w):
            if mode == 'max':
                output[images_arr, x, y] = np.max(images[images_arr,
                                                         x*stride[0]:x*stride[0]+kh, 
                                                         y*stride[1]:y*stride[1]+kw,],
                                                  axis=(1, 2))
            else:
                output[images_arr, x, y] = np.average(images[images_arr,
                                                             x*stride[0]:stride[0]+kh, 
                                                             y*stride[1]:y*stride[1]+kw,],
                                                      axis=(1, 2))
    return output
