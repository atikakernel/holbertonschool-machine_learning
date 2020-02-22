#!/usr/bin/env python3
""" converts a numeric label vector into a one-hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
    """ Returns: a one-hot encoding of Y with shape (classes, m),
    or None on failure """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None

    if len(Y) == 0:
        return None

    if not isinstance(classes, int):
        return None

    if classes <= np.amax(Y):
        return None

    onehot = np.zeros((classes, Y.shape[0]))
    for exp, lab in enumerate(Y):
        onehot[lab][exp] = 1
    return onehot
