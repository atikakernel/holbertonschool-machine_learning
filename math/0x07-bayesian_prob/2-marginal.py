#!/usr/bin/env python3
"""
marginal prob
"""

import numpy as np


def marginal(x, n, P, Pr):
    """
    Returns: the marginal probability of obtaining x and n
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')

    if type(x) is not int or x < 0:
        m = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(m)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        m = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(m)

    for i in range(len(P)):
        if not (P[i] >= 0 and P[i] <= 1):
            a = 'All values in P must be in the range [0, 1]'
            raise ValueError(a)

        if not (Pr[i] >= 0 and Pr[i] <= 1):
            a = 'All values in Pr must be in the range [0, 1]'
            raise ValueError(a)

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    combination = fact_n / (fact_x * fact_nx)
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))
    prior = Pr
    intersection = prior * likelihood
    marginal = np.sum(intersection)
    return marginal
