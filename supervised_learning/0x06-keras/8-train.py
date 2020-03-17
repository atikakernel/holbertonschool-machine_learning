#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    def lrate(epoch):
        """getting the learning rate each epoch"""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if validation_data:
        if early_stopping:
            es = K.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=patience)
            callbacks.append(es)

        if learning_rate_decay:
            lrs = K.callbacks.LearningRateScheduler(lrate, verbose=1)
            callbacks.append(lrs)

        if save_best:
            save = K.callbacks.ModelCheckpoint(filepath)
            callbacks.append(save)

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=early_stop)
