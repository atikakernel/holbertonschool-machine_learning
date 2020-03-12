#!/usr/bin/env python3
""" calculates the cost of a neural network with L2 regularization: """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ calculates the cost of a neural network with L2 regularization: """
    
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return (count >= patience, count)
