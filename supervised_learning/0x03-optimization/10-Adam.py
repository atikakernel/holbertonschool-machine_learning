#!/usr/bin/env python3
""" creates the training operation for a neural network in tensorflow
 using the Adam optimization algorithm"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm
    """
    Adam = tf.train.AdamOptimizer(alpha, beta1, beta2,
                                     epsilon).minimize(loss)
    return Adam
