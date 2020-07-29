#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def pca(X, var=0.95):
    """weights matrix, W, that maintains var fraction of X‘s
             original variance. W is a ndarray (d, nd)
             nd is the new dimensionality o the transformed X
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T
    return np.matmul(X_m, W)
