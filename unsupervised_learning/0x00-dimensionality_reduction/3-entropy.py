#!/usr/bin/env python3
"""
Calculates the entropy
"""
import numpy as np


def HP(Di, beta):
    """return: (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities of
        the points
    """
    num = (np.exp(- Di.copy() * beta))
    den = (np.sum(np.exp(-Di.copy() * beta)))
    P = num / den
    Hi = - np.sum(P * np.log2(P))
    return Hi, P
