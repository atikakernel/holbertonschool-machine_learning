#!/usr/bin/env python3
"""projection block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """the activated output of the projection block"""
    init = K.initializers.he_normal()
    activation = 'relu'
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s,
                            padding='same',
                            kernel_initializer=init)(A_prev)

    batchc1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(batchc1)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                            kernel_initializer=init)(relu1)
    batchc2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(batchc2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                            kernel_initializer=init)(relu2)

    conv1_proj = K.layers.Conv2D(filters=F12, kernel_size=1, strides=s,
                                 padding='same',
                                 kernel_initializer=init)(A_prev)

    batch3 = K.layers.BatchNormalization(axis=3)(conv3)

    batch4 = K.layers.BatchNormalization(axis=3)(conv1_proj)
    addit= K.layers.Add()([batch3, batch4])
    finelu = K.layers.Activation('relu')(addit)
    return finlu