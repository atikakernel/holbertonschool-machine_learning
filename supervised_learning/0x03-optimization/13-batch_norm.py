#!/usr/bin/env python3
""" creates the training operation for a neural network in tensorflow
 using the Adam optimization algorithm"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    std = np.sqrt(var + epsilon)
    return gamma * ((Z - mean) / (std)) + beta
