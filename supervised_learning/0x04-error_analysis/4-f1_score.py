#!/usr/bin/env python3
""" creates a confusion matrix:"""
import numpy as np


def f1_score(confusion):
    """creates a confusion matrix:"""
    precision = np.diagonal(confusion)/np.sum(confusion, axis=0)
    sensibility = np.diagonal(confusion)/np.sum(confusion, axis=1)
    return (2 * (precision * sensibility) / (precision + sensibility))
