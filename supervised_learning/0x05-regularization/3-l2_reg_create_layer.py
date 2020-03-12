#!/usr/bin/env python3
""" calculates the cost of a neural network with L2 regularization: """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ calculates the cost of a neural network with L2 regularization: """
    regul = tf.contrib.layers.l2_regularizer(lambtha)
    raw  = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    o_tensor = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=raw,
                            kernel_regularizer=regul)
    return(o_tensor(prev))
