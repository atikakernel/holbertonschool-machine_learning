#!/usr/bin/env python3
"""
Calculates the cost
"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation
    :param P: numpy.ndarray of shape (n, n) containing the P affinities
    :param Q: numpy.ndarray of shape (n, n) containing the Q affinities
    :return: C, the cost of the transformation
    """
    Q_n = np.where(Q == 0, 1e-12, Q)
    P_n = np.where(P == 0, 1e-12, P)
    return(np.sum(P * np.log(P_n / Q_n)))
