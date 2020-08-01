#!/usr/bin/env python3
"""
Determinant
"""


def minor_m(m, row, col):
    """
    Returns:
        the matrix with the omited row, column.
    """
    return [[m[i][j] for j in range(len(m[i])) if j != col]
            for i in range(len(m)) if i != row]


def determinant(matrix):
    """
    Returns:
        the determinant.
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if all([type(i) is list for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j in range(len(matrix[0])):
        omited_matrix = minor_m(matrix, 0, j)
        det += matrix[0][j] * ((-1) ** j) * determinant(omited_matrix)

    return det


def getMatrixMinor(m, i, j):
    """
    :param m: matrix
    :param i: x coord.
    :param j: y coord.
    :return: minor of the position
    """
    return determinant([row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])])


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    :param matrix: list of lists whose minor matrix should be calculated
    :return: minor matrix of matrix
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    if not all(len(matrix) == col for col in [len(row) for row in matrix]):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) is 1 and len(matrix[0]) is 1:
        return [[1]]
    minors = [x[:] for x in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            minors[i][j] = getMatrixMinor(matrix, i, j)
    return minors
