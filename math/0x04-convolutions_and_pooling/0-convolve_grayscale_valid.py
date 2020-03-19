#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolurion on grayscale image"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))

    images_arr = np.arange(0, m)

    for x in range(output_h):
        for y in range(output_w):
            y1 = y + kw
            x1 = x + kh
            output[images_arr, x, y] = np.sum(np.multiply(
                images[images_arr, x: x1, y: y1], kernel), axis=(1, 2))

    return output
