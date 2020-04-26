#!/usr/bin/env python3
"""Derivative"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if poly == [] or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [0]
    result = []
    for i in range(1, len(poly)):
        result.append(i * poly[i])

    if result == [0] * len(result):
        return [0]

    return result
