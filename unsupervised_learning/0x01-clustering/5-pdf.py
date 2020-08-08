#!/usr/bin/env python3
"""PDF function """

import numpy as np


def pdf(X, m, S):
    """
    Probability Density Function of gaussian distributions
    P, or None on failure
             P: numpy.ndarray of shape (n,) containing the PDF values for each
                data point.
                All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape
    mean = m
    x_m = X - mean
    x_mT = x_m.T
    det_S = np.linalg.det(S)
    inv_S = np.linalg.inv(S)
    part_1 = 1 / np.sqrt((2 * np.pi) ** d * det_S)
    part_2 = np.matmul((-x_m / 2), inv_S)
    part_2_1 = np.matmul(part_2, x_mT).diagonal()
    part_2_2 = np.exp(part_2_1)
    pdf = part_1 * part_2_2
    P = np.where(pdf < 1e-300, 1e-300, pdf)
    return P
