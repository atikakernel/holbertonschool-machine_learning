#!/usr/bin/env python3
"""DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network with one hidden layer"""

    def __init__(self, nx, layers):
        """constructor"""
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) != int or layers[i] < 0:
                raise TypeError('layers must be a list of positive integers')
            kb = "b" + str(i + 1)
            kW = "W" + str(i + 1)

            self.weights[kb] = np.zeros(layers[i]).reshape(layers[i], 1)
            if i > 0:
                val = layers[i-1]
            else:
                val = nx
            self.weights[kW] = np.random.randn(layers[i], val) * np.sqrt(2/val)

    @property
    def L(self):
        """property to retrieve L"""
        return self.__L

    @property
    def cache(self):
        """property to retrieve b1"""
        return self.__cache

    @property
    def weights(self):
        """property to retrieve A1"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X

        for l in range(self.__L):
            W_key = "W{}".format(l + 1)
            b_key = "b{}".format(l + 1)
            A_key_prev = "A{}".format(l)
            A_key_forw = "A{}".format(l + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural networks predictions"""
        Ev_f = self.forward_prop(X)[0]
        Ev_a = np.where(Ev_f >= 0.5, 1, 0)
        cost = self.cost(Y, Ev_f)
        return Ev_a, cost

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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        steps = 0
        c_ax = np.zeros(iterations + 1)

        temp_cost = []
        temp_iterations = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__cache["A" + str(self.__L)])
            if i % step == 0 or i == iterations:
                temp_cost.append(cost)
                temp_iterations.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(temp_iterations, temp_cost)
            plt.show()
        return self.evaluate(X, Y)
