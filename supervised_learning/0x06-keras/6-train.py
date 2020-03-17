#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    early_stop = None

    if validation_data and early_stopping:
        early_stop = [K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience)]

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=early_stop)
