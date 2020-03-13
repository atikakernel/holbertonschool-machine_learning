#!/usr/bin/env python3
""" calculates the cost of a neural network with L2 regularization: """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization: """
    sum1 = 0
    for i in range(L):
        l2 = np.linalg.norm(weights["W"+str(i + 1)], keepdims=True)
        sum1 = np.sum(l2) + sum1
    return cost + (lambtha / (2 * m)) * (sum1)
