#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a valid convolurion on grayscale image"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
    elif padding == 'same':
        ph, pw = int((kh-1)/2), int((kw-1)/2)
    else:
        ph, pw = 0, 0

    output_h = int(((h - kh + (2 * ph)) / stride[0]) + 1)
    output_w = int(((w - kw + (2 * pw)) / stride[1]) + 1)
    output = np.zeros((m, output_h, output_w, nc))

    images_arr = np.arange(0, m)
    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode="symmetric")

    for x in range(output_h):
        for y in range(output_w):
            for color in range(nc):
                y1 = kw + (y * stride[1])
                x1 = kh + (x * stride[0])
                output[images_arr, x, y, color] = np.sum(np.multiply(
                    pad_img[images_arr, x * stride[0]: x1, y * stride[1]: y1],
                    kernels[color]), axis=(1, 2, 3))

    return output
