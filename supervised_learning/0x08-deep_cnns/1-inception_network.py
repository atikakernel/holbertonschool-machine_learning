#!/usr/bin/env python3
"""creating LeNet"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    the keras  model
    """
    X = K.Input(shape=(224, 224, 3))
    lay = K.initializers.he_normal()
    conv_lay1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                                strides=(2, 2), padding='same',
                                activation='relu',
                                kernel_initializer=lay)
    conv1 = conv_lay1(X)
    maxp1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')
    maxpool1 = maxp1(conv1)
    conv_lay2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                                activation='relu',
                                kernel_initializer=lay)
    conv2 = conv_lay2(maxpool1)
    conv_lay3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                                padding='same', activation='relu',
                                kernel_initializer=lay)
    conv3 = conv_lay3(conv2)
    maxp2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')
    maxpool2 = maxp2(conv3)
    incep1 = inception_block(maxpool2, [64, 96, 128, 16, 32, 32])
    incep2 = inception_block(incep1, [128, 128, 192, 32, 96, 64])
    maxp_lay3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool3 = maxp_lay3(incep2)
    incep3 = inception_block(maxpool3, [192, 96, 208, 16, 48, 64])
    incep4 = inception_block(incep3, [160, 112, 224, 24, 64, 64])
    incep5 = inception_block(incep4, [128, 128, 256, 24, 64, 64])
    incep6 = inception_block(incep5, [112, 144, 288, 32, 64, 64])
    incep7 = inception_block(incep6, [256, 160, 320, 32, 128, 128])
    maxp_lay4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool4 = maxp_lay4(incep7)
    incep8 = inception_block(maxpool4, [256, 160, 320, 32, 128, 128])
    incep9 = inception_block(incep8, [384, 192, 384, 48, 128, 128])
    avgp_lay = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))
    avgpool = avgp_lay(incep9)
    drop_lay = K.layers.Dropout(0.4)
    dropout = drop_lay(avgpool)
    FC_lay = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=lay)
    FC = FC_lay(dropout)
    model = K.models.Model(inputs=X, outputs=FC)
    return model
