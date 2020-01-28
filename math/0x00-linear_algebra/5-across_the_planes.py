#!/usr/bin/env python3
"""Add a function to calculate the element wise sum of a 2D matrix"""


def add_matrices2D(mat1, mat2):
    """calculate element wise sum of a 2D matrix"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[sum(element_wise) for element_wise in zip(arr1, arr2)]
            for arr1, arr2 in zip(mat1, mat2)]
