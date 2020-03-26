#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5 architecture """
    init = tf.contrib.layers.variance_scaling_initializer()
    act = tf.nn.relu

    layer_1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                               padding='same',
                               activation=act,
                               kernel_initializer=init)(x)

    pool_1 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                    strides=2)(layer_1)

    layer_2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                               padding='valid',
                               activation=act,
                               kernel_initializer=init)(pool_1)

    pool_2 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                    strides=2)(layer_2)

    flat_pool = tf.layers.Flatten()(pool_2)

    layer_3 = tf.layers.Dense(units=120, activation=act,
                              kernel_initializer=init)(flat_pool)
    layer_4 = tf.layers.Dense(units=84, activation=act,
                              kernel_initializer=init)(layer_3)
    output_layer = tf.layers.Dense(units=10,
                                   kernel_initializer=init)(layer_4)
    Smax_pred = tf.nn.softmax(output_layer)
    loss = tf.losses.softmax_cross_entropy(y, output_layer)
    trainOp = tf.train.AdamOptimizer().minimize(loss)
    equal = tf.equal(tf.argmax(y, axis=1), tf.argmax(output_layer, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return Smax_pred, trainOp, loss, accuracy
