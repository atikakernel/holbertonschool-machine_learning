#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """sets up Adam optimization for a keras model"""

    Adam = K.optimizers.Adam(alpha, beta1, beta2)

    network.compile(optimizer=Adam,
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])

    return None
