#!/usr/bin/env python3
""" creates a confusion matrix:"""
import numpy as np


def sensitivity(confusion):
    """creates a confusion matrix:"""
    return np.diagonal(confusion)/np.sum(confusion, axis=1)
