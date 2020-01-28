#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis """
    copy1 = [row[:] for row in mat1]
    copy2 = mat2.copy()
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return copy1 + copy2

    elif axis == 1 and len(mat1) == len(mat2):
        for arr1, arr2 in zip(copy1, copy2):
            arr1.extend(arr2)
        return copy1
