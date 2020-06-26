#!/usr/bin/env python3
"""Bloc"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
         The output of the transition layer and the number of filters within
         the output, respectively.
    """
    init = K.initializers.he_normal()
    batch_normalization_ = K.layers.BatchNormalization(
        axis=-1
    )
    batch_normalization = batch_normalization_(X)
    activation_ = K.layers.Activation(
        activation='relu'
    )
    activation = activation_(batch_normalization)
    conv2d_ = K.layers.Conv2D(
        filters=int(nb_filters * compression),
        kernel_size=[1, 1],
        kernel_initializer=init,
        padding='same',
    )
    conv2d = conv2d_(activation)
    average_pooling2d_ = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )
    average_pooling2d = average_pooling2d_(conv2d)
    nb_filters *= compression
    nb_filters = int(nb_filters)
    return average_pooling2d, nb_filters
