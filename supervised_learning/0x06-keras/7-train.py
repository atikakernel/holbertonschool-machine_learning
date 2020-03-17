#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    callbacks = []

    def schedule(epochs):
        """ support callback Learning Rate"""
        lr_cb = alpha / (1 + decay_rate * epochs)
        return lr_cb

    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))

    if learning_rate_decay and validation_data:

        callbacks.append(K.callbacks.LearningRateScheduler(schedule, 1))

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=early_stop)
