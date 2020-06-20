#!/usr/bin/env python3
"""Bloc"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
         The output of the transition layer and the number of filters within
         the output, respectively.
    """
    filters = int(filters * compression)

    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters=filters,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer='he_normal')(x)

    x = K.layers.AvgPool2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid')(x)

    return x, filters