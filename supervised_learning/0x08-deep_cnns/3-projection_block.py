#!/usr/bin/env python3
"""projection block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """the activated output of the projection block"""
    F11, F3, F12 = filters
    lay_init = K.initializers.he_normal()

    conv_layF11 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                                  padding='same', strides=(s, s),
                                  kernel_initializer=lay_init)
    convF11 = conv_layF11(A_prev)

    norm_lay1 = K.layers.BatchNormalization(axis=3)
    norm1 = norm_lay1(convF11)

    X1 = K.layers.Activation('relu')(norm1)
    conv_layF3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                                 padding='same',
                                 kernel_initializer=lay_init)
    convF3 = conv_layF3(X1)

    norm_lay2 = K.layers.BatchNormalization(axis=3)
    norm2 = norm_lay2(convF3)

    X2 = K.layers.Activation('relu')(norm2)
    conv_layF12 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                  padding='same',
                                  kernel_initializer=lay_init)
    convF12 = conv_layF12(X2)

    norm_lay3 = K.layers.BatchNormalization(axis=3)
    norm3 = norm_lay3(convF12)
    shortcut_lay = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                   padding='same', strides=(s, s),
                                   kernel_initializer=lay_init)
    shortcut = shortcut_lay(A_prev)
    norm_layshort = K.layers.BatchNormalization(axis=3)
    norm_short = norm_layshort(shortcut)
    result = K.layers.Add()([norm3, norm_short])
    X3 = K.layers.Activation('relu')(result)
    return X3
