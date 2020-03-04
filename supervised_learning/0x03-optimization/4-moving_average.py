#!/usr/bin/env python3
"""calculates the weighted moving average of a data set:"""

import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set:"""
    a = 0
    bias_corrected = []
    for i in range(len(data)):
        a = (beta * a + (1 - beta) * data[i])
        bias_corrected.append(a / (1 - beta ** (i + 1)))
    return bias_corrected
