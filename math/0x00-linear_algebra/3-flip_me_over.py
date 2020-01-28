#!/usr/bin/env python3
"""Add a function to calculate the transpose of a matrix"""


def matrix_transpose(matrix):
    """calculates the transpose of a matrix"""
    return[[matrix[col][row] for col in range(len(matrix))]
           for row in range(len(matrix[0]))]
