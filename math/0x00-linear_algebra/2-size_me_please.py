#!/usr/bin/env python3
"""Add a function to calculate the shape of a matrix"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
