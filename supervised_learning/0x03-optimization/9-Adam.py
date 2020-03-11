#!/usr/bin/env python3
""" creates the training operation for a neural network in tensorflow
 using the Adam optimization algorithm"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm
    """
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    Sd = (beta2 * s) + ((1 - beta2) * grad * grad)

    n_p_corr = Vd / (1 - beta1 ** t)
    n_s_corr = Sd / (1 - beta2 ** t)

    new_v = var - alpha * (n_p_corr / ((n_s_corr ** (0.5)) + epsilon))
    return (new_v, Vd, Sd)
