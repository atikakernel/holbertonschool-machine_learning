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
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)
    betas = np.ones((n, 1))
    P = np.zeros((n, n))
    H = np.log2(perplexity)

    return D, P, betas, H
