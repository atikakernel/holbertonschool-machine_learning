#!/usr/bin/env python3
"""inception"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block as described in Going Deeper with
    Convolutions (2014).
    The concatenated output of the inception block
    """
    convly_1= K.layers.Conv2D(filters=filters[0],
                         kernel_size=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(A_prev)

    convly_2P = K.layers.Conv2D(filters=filters[1],
                          kernel_size=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          activation='relu')(A_prev)

    layer_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(1, 1))(A_prev)

    convly_3 = K.layers.Conv2D(filters=filters[2],
                         kernel_size=3,
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(convly_2P)

    convly_3P = K.layers.Conv2D(filters=filters[3],
                          kernel_size=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          activation='relu')(A_prev)

    convly_3s = K.layers.Conv2D(filters=filters[4],
                         kernel_size=5,
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(convly_3P)

    OFPP = K.layers.Conv2D(filters=filters[5],
                           kernel_size=1,
                           padding='same',
                           kernel_initializer='he_normal',
                           activation='relu')(layer_pool)

    mid_layer = K.layers.Concatenate(axis=3)([convly_1, convly_3, convly_3s, OFPP])

    return mid_layer