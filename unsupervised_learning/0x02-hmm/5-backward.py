#!/usr/bin/env python3
"""
The Backward Algorithm
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """
    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))

    for i in range(T - 2, -1, -1):
        for j in range(N):
            aux = B[:, i + 1] * Transition[j, :] *\
                  Emission[:, Observation[i + 1]]
            B[j, i] = np.sum(aux)
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
