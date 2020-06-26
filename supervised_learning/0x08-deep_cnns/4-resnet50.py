#!/usr/bin/env python3
"""Creating ResNet"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ the keras model"""
    X = K.Input(shape=(224, 224, 3))
    lay_init = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            padding='same', strides=(2, 2),
                            kernel_initializer=lay_init)(X)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    X1 = K.layers.Activation('relu')(norm1)

    mxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                    padding="same")(X1)

    block0 = projection_block(mxpool1, [64, 64, 256], 1)
    block1 = identity_block(block0, [64, 64, 256])
    block2 = identity_block(block1, [64, 64, 256])
    pro_block1 = projection_block(block2, [128, 128, 512])
    block3 = identity_block(pro_block1, [128, 128, 512])
    block4 = identity_block(block3, [128, 128, 512])
    block5 = identity_block(block4, [128, 128, 512])
    block2 = projection_block(block5, [256, 256, 1024])
    block6 = identity_block(block2, [256, 256, 1024])
    block7 = identity_block(block6, [256, 256, 1024])
    block8 = identity_block(block7, [256, 256, 1024])
    block9 = identity_block(block8, [256, 256, 1024])
    block10 = identity_block(block9, [256, 256, 1024])
    pro_block2 = projection_block(block10, [512, 512, 2048])
    block11 = identity_block(pro_block2, [512, 512, 2048])
    block12 = identity_block(block11, [512, 512, 2048])

    averagepool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                            strides=(1, 1))(block12)

    FC = K.layers.Dense(units=1000, activation='softmax',
                        kernel_initializer=lay_init)(averagepool)

    model = K.models.Model(inputs=X, outputs=FC)
    return model
