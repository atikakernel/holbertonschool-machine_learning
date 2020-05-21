#!/usr/bin/env python3
"""  creates the training operation for the network: """


import tensorflow as tf


def calculate_loss(y, y_pred):
    """  creates the training operation for the network: """

    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
