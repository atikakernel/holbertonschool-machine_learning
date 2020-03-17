#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build a neuronal network """
    inputs = K.Input(shape=(nx,))
    regul = K.regularizers.l2(lambtha)
    
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=regul)(inputs)

    for lyr, act_f in zip(layers[1:], activations[1:]):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(lyr, activation=act_f,
                           kernel_regularizer=regul)(x)
        
    model = K.Model(inputs=inputs, outputs=x)
    return model
