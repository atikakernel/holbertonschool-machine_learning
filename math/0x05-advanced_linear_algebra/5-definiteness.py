#!/usr/bin/env python3
"""
Determinant
"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix:
    Args:
        matrix (np.ndarray): shape (n, n) whose definiteness should be
        calculated.
    Returns:
        matrix definiteness
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.all(matrix.T == matrix):
        return None
    eigenvalues, eigenvector = np.linalg.eig(matrix)

    positive = np.where(eigenvalues > 0)
    ceros = np.where(eigenvalues == 0)
    negative = np.where(eigenvalues < 0)

    pos = eigenvalues[positive]
    cer = eigenvalues[ceros]
    neg = eigenvalues[negative]
    if pos.size and not cer.size and not neg.size:
        return ('Positive definite')
    elif pos.size and cer.size and not neg.size:
        return ('Positive semi-definite')
    elif not pos.size and not cer.size and neg.size:
        return ('Negative definite')
    elif not pos.size and cer.size and neg.size:
        return ('Negative semi-definite')
    elif pos.size and not cer.size and neg.size:
        return ('Indefinite')
    else:
        return None
