#!/usr/bin/env python3
"""DenseNet"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ the keras model"""
    X = K.Input(shape=(224, 224, 3))
    batn = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(batn)
    conv2d = K.layers.Conv2D(64, 7, padding='same', strides=2,
                             kernel_initializer='he_normal')(act)
    max_pool = K.layers.MaxPool2D(3, 2, padding='same')(conv2d)
    dbl1, nw_nb = dense_block(max_pool, 64, growth_rate, 6)
    tl1, nw_nb = transition_layer(dbl1, int(nw_nb), compression)
    dbl2, nw_nb = dense_block(tl1, int(nw_nb), growth_rate, 12)
    tl2, nw_nb = transition_layer(dbl2, int(nw_nb), compression)
    dbl3, nw_nb = dense_block(tl2, int(nw_nb), growth_rate, 24)
    tl3, nw_nb = transition_layer(dbl3, int(nw_nb), compression)
    dbl4, nw_nb = dense_block(tl3, int(nw_nb), growth_rate, 16)
    averpool = K.layers.AveragePooling2D(7)(dbl4)
    dense = K.layers.Dense(1000, activation='softmax')(averpool)
    model = K.models.Model(inputs=X, outputs=dense)
    return model
