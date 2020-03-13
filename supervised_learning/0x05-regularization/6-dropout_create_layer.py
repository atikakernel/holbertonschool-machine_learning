#!/usr/bin/env python3
""" calculates the cost of a neural network with L2 regularization: """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ calculates the cost of a neural network with L2 regularization: """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.dense(prev, units=n, activation=activation,
                            kernel_initializer=init)

    return tf.layers.dropout(layer, keep_prob)
