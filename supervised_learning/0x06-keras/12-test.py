#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network:"""
    return network.evaluate(data, labels, verbose=verbose)
