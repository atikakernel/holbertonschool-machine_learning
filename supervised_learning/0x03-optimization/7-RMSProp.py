#!/usr/bin/env python3
"""updates a variable using the RMSProp optimization algorithm:"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm:"""
    Sdvar = beta2 * s + (1 - beta2) * (grad**2)
    new_var = var - alpha*(grad / (Sdvar**(1/2) + epsilon))
    return new_var, Sdvar
