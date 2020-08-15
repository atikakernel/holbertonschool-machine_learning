#!/usr/bin/env python3
"""
The Viretbi Algorithm
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    markov model
    """
    T = Observation.shape[0]
    N, M = Emission.shape
    bp = np.zeros((N, T))
    v = np.zeros((N, T))
    v[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for s in range(N):
            aux = (v[:, t - 1] * Transition[:, s]) * \
                   Emission[s, Observation[t]]
            v[s, t] = np.max(aux)
            bp[s, t] = np.argmax((v[:, t - 1] * Transition[:, s]) *
                                 Emission[s, Observation[t]])
    P = np.max(v[:, -1])
    S = np.argmax(v[:, -1])
    path = [S]

    for t in range(T - 1, 0, -1):
        S = int(bp[S, t])
        path.append(S)
    return path[::-1], P
