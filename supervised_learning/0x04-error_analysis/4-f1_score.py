#!/usr/bin/env python3
""" creates a confusion matrix:"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """creates a confusion matrix:"""
    sensitivity1 = sensitivity(confusion)
    precision1 = precision(confusion)
    return 2 * sensitivity1 * precision1 / (sensitivity1 + precision1)
