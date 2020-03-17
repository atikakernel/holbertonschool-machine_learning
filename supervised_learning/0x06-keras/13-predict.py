#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """tests a neural network:"""
    return network.predict(data, verbose=verbose)
