#!/usr/bin/env python3
"""
Mean and Covariance
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    :param X: numpy.ndarray of shape (n, d) containing the data set
    :return: mean, cov
    """
    if not isinstance(X, np.ndarray)or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    X_mean = X - mean
    cov = np.dot(X_mean.T, X_mean) / (n - 1)

    return (mean, cov)
