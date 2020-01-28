#!/usr/bin/env python3
"""Add a function to calculate the element wise sum"""


def add_arrays(arr1, arr2):
    """calculate element wise sum"""
    if len(arr1) != len(arr2):
        return None
    return [sum(element_wise) for element_wise in zip(arr1, arr2)]
