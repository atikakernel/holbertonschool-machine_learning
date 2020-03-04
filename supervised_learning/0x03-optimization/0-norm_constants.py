#!/usr/bin/env python3
"""calculates the normalization (standardization) constants of a matrix"""

import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix"""
    return X.mean(axis=0), X.std(axis=0)
