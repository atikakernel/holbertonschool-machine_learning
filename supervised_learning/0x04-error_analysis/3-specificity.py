#!/usr/bin/env python3
""" creates a confusion matrix:"""
import numpy as np


def specifity(confusion):
    """creates a confusion matrix:"""
    return (np.sum(confusion) - np.sum(confusion, axis=0) -
            np.sum(confusion, axis=1) + np.diagonal(confusion))
