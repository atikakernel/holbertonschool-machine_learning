#!/usr/bin/env python3
"""
Regular Chains
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    """
    try:
        n = P.shape[0]
        ans = np.ones((1, n))
        eq = np.vstack([P.T - np.identity(n), ans])
        results = np.zeros((n, 1))
        results = np.vstack([results, np.array([1])])
        stationary = np.linalg.solve(eq.T.dot(eq), eq.T.dot(results)).T

        if len(np.argwhere(stationary < 0)) > 0:
            return None
        return stationary

    except Exception:
        return None
