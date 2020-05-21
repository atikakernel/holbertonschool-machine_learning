#!/usr/bin/env python3
""" calculates the accuracy of a prediction: """


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction: """

    pred = tf.argmax(y_pred, 1)
    equality = tf.equal(tf.argmax(y, 1), pred)
    acc = tf.reduce_mean(tf.cast(equality, tf.float32))
    return acc
