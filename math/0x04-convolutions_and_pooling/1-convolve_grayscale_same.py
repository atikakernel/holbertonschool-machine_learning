#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a valid convolurion on grayscale image"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    padding_h = int((kh - 1) / 2)
    padding_w = int((kw - 1) / 2)

    output_h = h + (2 * padding_h) - kh + 1
    output_w = w + (2 * padding_w) - kw + 1
    output = np.zeros((m, output_h, output_w))

    images_arr = np.arange(0, m)
    pad_img = np.pad(images, 1, mode="symmetric")

    for x in range(output_h):
        for y in range(output_w):
            y1 = y + kw
            x1 = x + kh
            output[images_arr, x, y] = np.sum(np.multiply(
                pad_img[images_arr, x: x1, y: y1], kernel), axis=(1, 2))

    return output
