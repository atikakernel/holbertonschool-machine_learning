#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model:"""

    network.save(filename)
    return None


def load_model(filename):
    """loads an entire model"""

    return K.models.load_model(filename)
