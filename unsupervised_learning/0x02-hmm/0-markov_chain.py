#!/usr/bin/env python3
"""Markov chain probability"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain in a particular
    state after a specified number of iterations
    """
    try:

        if len(P.shape) != 2:
            return None
        n1, n2 = P.shape
        if (n1 != n2) or type(P) is not np.ndarray or not isinstance(t, int):
            return None
        if t < 0:
            return None
        if n1 != s.shape[1] or s.shape[0] != 1:
            return None
        for i in range(t):
            s = np.dot(s, P)
        return s

    except Exception:
        return None
