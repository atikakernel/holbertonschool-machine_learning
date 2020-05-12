#!/usr/bin/env python3
"""Script to create a Neuron in a ANN"""

import numpy as np
import matplotlib.pyplot as plt


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

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neurons predictions"""
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Returns: gradient descent bias + adjusted weights"""

        m = Y.shape[1]
        dz = A - Y
        dW = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W -= (alpha * dW).T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ optimized and cost of training"""

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost_list = []
        steps_list = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if i % step == 0 or i == iterations:
                cost_list.append(self.cost(Y, self.__A))
                steps_list.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".
                          format(i, self.cost(Y, self.__A)))
        if graph is True:
            plt.plot(steps_list, cost_list, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
