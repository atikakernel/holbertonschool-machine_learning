#!/usr/bin/env python3
"""
Variance
"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    var, or None on failure
        var is the total variance
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(C, np.ndarray) or len(C.shape) != 2:
            return None
        D = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        min_dist = np.min(D, axis=0)
        return np.sum(min_dist ** 2)

    except Exception:
        return None
