#!/usr/bin/env python3
""" calculates the cost of a neural network with L2 regularization: """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization: """
    for i in range(L):
        k = "W{}".format(i + 1)
        sum1 = np.linalg.norm(weights[k])
        L2_cost = cost + lambtha * sum1 / (2 * m)

    return L2_cost
