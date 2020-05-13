#!/usr/bin/env python3
"""Deep net"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """ Deep Neural Network"""

    def __init__(self, nx, layers):
        """const"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for lay in range(self.L):
            if layers[lay] < 1 or type(layers[lay]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))
            if lay == 0:
                He_val = np.random.randn(layers[lay], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(lay + 1)] = He_val
            if lay > 0:
                He_val1 = np.random.randn(layers[lay], layers[lay - 1])
                He_val2 = np.sqrt(2 / layers[lay - 1])
                He_val3 = He_val1 * He_val2
                self.__weights["W" + str(lay + 1)] = He_val3

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def _sigmoid(z):
        """sigmoid function"""
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(A):
        """_sigmoid_derivative for the backpropagation"""
        return A * (1 - A)

    def forward_prop(self, X):
        """Forward propagation function"""
        self.__cache["A0"] = X
        for la in range(self.__L):
            key_W = "W{}".format(la + 1)
            key_b = "b{}".format(la + 1)
            key_A = "A{}".format(la)
            key_newA = "A{}".format(la + 1)

            W = self.__weights[key_W]
            A = self.__cache[key_A]
            b = self.__weights[key_b]
            z = np.matmul(W, A) + b

            if la < self.__L - 1:
                activation = 1 / (1 + np.exp(-z))
            else:
                t = np.exp(z)
                activation = t / np.sum(t, axis=0, keepdims=True)

            self.__cache[key_newA] = activation
        return activation, self.__cache

    def cost(Y, A):
        """cost function for the network"""
        m = Y.shape[1]
        log_likelihood = -(Y * np.log(A))
        return np.sum(log_likelihood) / m

    def evaluate(self, X, Y):
        """evaluates network for one pass"""
        A, _ = self.forward_prop(X)
        A_binary = np.where(np.amax(A, axis=0), 1, 0)
        J = self.cost(Y, A)
        return A_binary, J

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ gradient descent bias + adjusted weights"""

        m = Y.shape[1]
        weights = self.__weights.copy()
        A2 = self.__cache["A" + str(self.__L - 1)]
        A3 = self.__cache["A" + str(self.__L)]
        W3 = weights["W" + str(self.__L)]
        b3 = weights["b" + str(self.__L)]
        dZ_input = {}
        dZ3 = A3 - Y
        dZ_input["dz" + str(self.__L)] = dZ3

        dW3 = (1 / m) * np.matmul(A2, dZ3.T)
        db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

        self.__weights["W" + str(self.__L)] = W3 - (alpha * dW3).T
        self.__weights["b" + str(self.__L)] = b3 - (alpha * db3)

        for lay in range(self.__L - 1, 0, -1):
            cache = self.__cache
            Aa = cache["A" + str(lay)]
            Ap = cache["A" + str(lay - 1)]
            Wa = weights["W" + str(lay)]
            Wn = weights["W" + str(lay + 1)]
            ba = weights["b" + str(lay)]
            dZ1 = np.matmul(Wn.T, dZ_input["dz" + str(lay + 1)])
            dZ2 = Aa * (1 - Aa)
            dZ = dZ1 * dZ2
            dW = (1 / m) * np.matmul(Ap, dZ.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZ_input["dz" + str(lay)] = dZ
            self.__weights["W" + str(lay)] = Wa - (alpha * dW).T
            self.__weights["b" + str(lay)] = ba - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron, performing gradient_descent iterations times."""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        print_step = 0
        cost_axis = np.zeros(iterations + 1, )

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)
        if step and (i == print_step or i == iterations):
            if verbose is True:
                print("Cost after {} iterations: {}".format(i, cost))
                print_step += step

        if graph is True:
            cost_axis[i] = cost

        if i < iterations:
            self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            plt.plot(np.arange(0, iterations + 1), cost_axis)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """saved object"""
        if '.pkl' not in filename:
            filename += '.pkl'

        fileObject = open(filename, 'wb')
        pickle.dump(self, fileObject)
        fileObject.close()

    @staticmethod
    def load(filename):
        """ Objects loaded"""
        try:
            with open(filename, 'rb') as f:
                fileOpen = pickle.load(f)
            return fileOpen
        except FileNotFoundError:
            return None
