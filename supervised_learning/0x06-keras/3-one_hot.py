#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix"""

    return K.utils.to_categorical(y=labels, num_classes=classes)
