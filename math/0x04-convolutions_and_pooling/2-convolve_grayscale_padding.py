#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a valid convolurion on grayscale image"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = padding[0]
    pw = padding[1]

    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1
    output = np.zeros((m, output_h, output_w))

    images_arr = np.arange(0, m)
    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode="symmetric")

    for x in range(output_h):
        for y in range(output_w):
            y1 = y + kw
            x1 = x + kh
            output[images_arr, x, y] = np.sum(np.multiply(
                pad_img[images_arr, x: x1, y: y1], kernel), axis=(1, 2))

    return output
