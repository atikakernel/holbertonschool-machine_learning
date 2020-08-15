#!/usr/bin/env python3
""" Forward Markov chain"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm of a HMM
    """
    try:
        N = Transition.shape[0]
        T = Observation.shape[0]
        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.sum(Transitions * F[:, t - 1]
                                 * Emissions)

        P = np.sum(F[:, -1])
        return P, F

    except Exception:
        None, None
