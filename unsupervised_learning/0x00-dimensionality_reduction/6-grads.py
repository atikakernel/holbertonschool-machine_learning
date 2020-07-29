#!/usr/bin/env python3
"""
Calculates the gradients
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y
    :param Y: numpy.ndarray of shape (n, ndim) containing the low dimensional
    transformation of X
    :param P: numpy.ndarray of shape (n, n) containing the P affinities of X
    :return: (dY, Q)
        dY is a numpy.ndarray of shape (n, ndim) containing the gradients of Y
        Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    equation1 = ((P - Q) * num)
    dY = np.zeros((n, ndim))
    for i in range(n):
        aux = np.tile(equation1[:, i].reshape(-1, 1), ndim)
        dY[i] = (aux * (Y[i] - Y)).sum(axis=0)
    return (dY, Q)
