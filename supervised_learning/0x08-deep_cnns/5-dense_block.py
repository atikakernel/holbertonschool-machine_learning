#!/usr/bin/env python3
"""Dense Block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """
    cat = X
    for _ in range(layers):
        batchn1 = K.layers.BatchNormalization()(cat)
        act_a = K.layers.Activation('relu')(batchn1)
        conv2d_a = K.layers.Conv2D(4 * growth_rate, 1, padding='same',
                                   kernel_initializer='he_normal')(act_a)
        batchn2 = K.layers.BatchNormalization()(conv2d_a)
        act_b = K.layers.Activation('relu')(batchn2)
        conv2d_b = K.layers.Conv2D(growth_rate, 3, padding='same',
                                   kernel_initializer='he_normal')(act_b)
        cat = K.layers.concatenate([cat, conv2d_b])

    return cat, cat.shape[-1]