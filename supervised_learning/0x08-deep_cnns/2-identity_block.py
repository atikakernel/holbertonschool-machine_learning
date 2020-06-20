#!/usr/bin/env python3
"""Identity Block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """activated output of the identity block"""
    F11, F3, F12 = filters

    conv2d = K.layers.Conv2D(F11, 1, padding='same',
                             kernel_initializer='he_normal')(A_prev)
    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activ = K.layers.Activation('relu')(batch_normalization)

    conv2d_1 = K.layers.Conv2D(F3, 3, padding='same',
                               kernel_initializer='he_normal')(activ)
    batchn1 = K.layers.BatchNormalization()(conv2d_1)
    act_1 = K.layers.Activation('relu')(batchn1)

    conv2d_2 = K.layers.Conv2D(F12, 1, padding='same',
                               kernel_initializer='he_normal')(act_1)
    batchn2 = K.layers.BatchNormalization()(conv2d_2)

    additi = K.layers.Add()([batchn2, A_prev])
    act_2 = K.layers.Activation('relu')(additi)
    return act_2