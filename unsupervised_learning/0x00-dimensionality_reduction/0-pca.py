#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def pca(X, var=0.95):
    """weights matrix, W, that maintains var fraction of X‘s
             original variance. W is a ndarray (d, nd)
             nd is the new dimensionality o the transformed X
    """
    u, Sigma, vh = np.linalg.svd(X, full_matrices=False)
    cumulative_var = np.cumsum(Sigma) / np.sum(Sigma)
    r = (np.argwhere(cumulative_var >= var))[0, 0]
    w = vh.T
    wr = w[:, :r + 1]
    return wr
