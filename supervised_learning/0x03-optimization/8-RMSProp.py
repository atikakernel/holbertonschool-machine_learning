#!/usr/bin/env python3
""" creates the training operation for a neural network in tensorflow
 using the RMSProp optimization algorithm"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm
    """
    RMSProp = tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                        epsilon=epsilon).minimize(loss)
    return RMSProp
