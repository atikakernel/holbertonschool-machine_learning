#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def P_init(X, perplexity):
    """Returns: (D, P, betas, H)
    D: a numpy.ndarray of shape (n, n) that
    calculates the pairwise distance between two data points
    P: a numpy.ndarray of shape (n, n) initialized to
    all 0‘s that will contain the P affinities
    betas: a numpy.ndarray of shape (n, 1) initialized to
    all 1’s that will contain all of the beta values
    """
    n, d = X.shape
    x_square = np.sum(np.square(X), axis=1)
    y_square = np.sum(np.square(X), axis=1)
    xy = np.dot(X, X.T)
    D = np.add(np.add((-2 * xy), x_square).T, y_square)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return (D, P, betas, H)
