#!/usr/bin/env python3
"""creating LeNet"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Returns: the keras  model
    """
    X = K.Input(shape=(224, 224, 3))
    lay_init = K.initializers.he_normal()
    conv_lay1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                                strides=(2, 2), padding='same',
                                activation='relu',
                                kernel_initializer=lay_init)
    conv1 = conv_lay1(X)
    max_pooling2d_1  = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool1 =  max_pooling2d_1(conv1)
    conv_lay2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                                activation='relu',
                                kernel_initializer=lay_init)
    conv2 = conv_lay2(maxpool1)
    conv_lay3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                                padding='same', activation='relu',
                                kernel_initializer=lay_init)
    conv3 = conv_lay3(conv2)
    maxpooling2D2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool2 = maxpooling2D2(conv3)
    block1 = inception_block(maxpool2, [64, 96, 128, 16, 32, 32])
    block2 = inception_block(block1, [128, 128, 192, 32, 96, 64])
    maxpooling2D3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool3 = maxpooling2D3(block2)
    block3 = inception_block(maxpool3, [192, 96, 208, 16, 48, 64])
    block4 = inception_block(block3, [160, 112, 224, 24, 64, 64])
    block5 = inception_block(block4, [128, 128, 256, 24, 64, 64])
    block6 = inception_block(block5, [112, 144, 288, 32, 64, 64])
    block7 = inception_block(block6, [256, 160, 320, 32, 128, 128])
    maxpooling2D4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool4 = maxpooling2D4(block7)
    block8 = inception_block(maxpool4, [256, 160, 320, 32, 128, 128])
    block9 = inception_block(block8, [384, 192, 384, 48, 128, 128])
    avergepooling2D= K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))
    avgpool = averagepooling2D(block9)
    droper = K.layers.Dropout(0.4)
    dropout = droper(avgpool)
    denser = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=lay_init)
    dense = denserdropout)
    model = K.models.Model(inputs=X, outputs=dense)
    return model