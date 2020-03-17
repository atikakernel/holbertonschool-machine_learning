#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build a neuronal network """
    inputs = K.Input(shape=(nx,))
    regularizers = K.regularizers.l2(lambtha)

    model = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=regularizers)(inputs)

    for i in range(1, len(layers)):
        model = K.layers.Dropout(rate=1-keep_prob)(model)
        model = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=k_reg)(model)

    return K.Model(inputs=inputs, outputs=model)
