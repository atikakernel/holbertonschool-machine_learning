#!/usr/bin/env python3
"""Return the integer value of the sum"""


def summation_i_squared(n):
    """ Function summation """
    if n == 0:
        return 0
    else:
        return summation_i_squared(n - 1) + n * n
