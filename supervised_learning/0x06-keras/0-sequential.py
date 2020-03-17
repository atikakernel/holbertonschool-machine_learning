#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library """

    model = K.Sequential()

    model.add(K.layers.Dense(layers[0], input_dim=nx,
                             activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lambtha)))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1-keep_prob))
        layer = K.layers.Dense(layers[i],
                               activation=activations[i],
                               kernel_regularizer=K.regularizers.l2(lambtha))
        model.add(layer)

    return model
