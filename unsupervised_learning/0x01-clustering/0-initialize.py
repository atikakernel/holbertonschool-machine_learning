#!/usr/bin/env python3
"""
Kmean
"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.
        np.ndarray of shape (k, d) containing the initialized centroids for
        each cluster, or None on failure.
    """
    if type(k) is not int or k < 1:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    n, d = X.shape
    mini = np.amin(X, axis=0)
    maxi = np.amax(X, axis=0)
    return np.random.uniform(mini, maxi, size=(k, d))
