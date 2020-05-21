#!/usr/bin/env python3
""" Layers """


import tensorflow as tf


def create_layer(prev, n, activation):
    """ Returns: the tensor output of the layer """

    ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    ten_l = (tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=ini, name="layer"))
    return ten_l(prev)
