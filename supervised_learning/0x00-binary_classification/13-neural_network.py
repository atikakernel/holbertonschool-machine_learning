#!/usr/bin/env python3
"""Contains the NeuralNetwork"""

import numpy as np


class NeuralNetwork():
    """ neural network"""

    def __init__(self, nx, nodes):
        """ constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.b1 = np.zeros(nodes).reshape(nodes, 1)
        self.A1 = 0
        self.W2 = np.random.randn(nodes).reshape(1, nodes)
        self.b2 = 0
        self.A2 = 0

        @property
        def W1(self):
                """property to retrieve W1"""
                return self.__W1

        @property
        def b1(self):
            """property to retrieve b1"""
            return self.__b1

        @property
        def A1(self):
            """property to retrieve A1"""
            return self.__A1

        @property
        def W2(self):
            """property to retrieve W2"""
            return self.__W2

        @property
        def b2(self):
            """property to retrieve b2"""
            return self.__b2

        @property
        def A2(self):
            """property to retrieve A2"""
            return self.__A2

        def forward_prop(self, X):
            """Calculates forward propagation of the neural network"""
            Z1 = np.matmul(self.__W1, X) + self.__b1
            self.__A1 = 1 / (1 + np.exp(-Z1))
            Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
            self.__A2 = 1 / (1 + np.exp(-Z2))
            return self.__A1, self.__A2

        def cost(self, Y, A):
            """Calculates the cost of the model using logistic regression"""
            cost = -np.sum((Y * np.log(A)) +
                           ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
            return cost

        def evaluate(self, X, Y):
            """Evaluates the neural networks predictions"""
            self.forward_prop(X)
            A2 = np.where(self.__A2 >= 0.5, 1, 0)
            cost = self.cost(Y, self.__A2)
            return A2, cost

        def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
            """ gradient descent bias + adjusted weights"""

            m = Y.shape[1]
            dz2 = A2 - Y
            dW2 = np.matmul(A1, dz2.T) / m

            db2 = np.sum(dz2, axis=1, keepdims=True) / m
            dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
            dW1 = np.matmul(dz1, X.T) / m
            db1 = np.sum(dz1, axis=1, keepdims=True) / m

            self.__W2 -= (alpha * dW2).T
            self.__b2 -= alpha * db2
            self.__W1 -= alpha * dW1
            self.__b1 -= alpha * db1
