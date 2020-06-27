#!/usr/bin/env python3
"""
Transfer Learning.
"""
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """Preprocess data for Resnet50."""
    X = K.applications.resnet50.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    (xt, yt), (x, y) = K.datasets.cifar10.load_data()
    xt, yt = preprocess_data(xt, yt)
    x, y = preprocess_data(x, y)
    model = K.applications.ResNet50(include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3))
    mo1 = K.Sequential()
    mo1.add(K.layers.UpSampling2D((7, 7)))
    mo1.add(model)
    mo1.add(K.layers.AveragePooling2D(pool_size=7))
    mo1.add(K.layers.Flatten())
    mo1.add(K.layers.Dense(10, activation=('softmax')))
    checkpoint = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                             monitor='val_acc',
                                             mode='max',
                                             verbose=1,
                                             save_best_only=True)
    mo1.compile(optimizer=K.optimizers.RMSprop(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['acc'])
    mo1.fit(xt, yt,
            validation_data=(x, y),
            batch_size=32,
            epochs=5,
            verbose=1,
            callbacks=[checkpoint])
    mo1.save('cifar10.h5')
