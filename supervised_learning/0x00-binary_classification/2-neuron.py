#!/usr/bin/env python3
"""Script to create a Neuron in a ANN"""

import numpy as np


class Neuron:
    """ Class Neuron """

    def __init__(self, nx):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """property to retrieve it"""
        return self.__W

    @property
    def b(self):
        """property to retrieve it"""
        return self.__b

    @property
    def A(self):
        """property to retrieve it"""
        return self.__A

    def forward_prop(self, X):
        """ Calculates neuron output."""
        Z = np.matmul(self.__W, X)
        z = Z + self.__b
        sigmoid_z = 1 / (1 + np.exp(-z))
        self.__A = sigmoid_z
        return self.__A
