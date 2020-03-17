#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    def schedule(epoch):
        """getting the learning rate each epoch"""
        return alpha * 1.0 / (1.0 + decay_rate * epoch)

    callbacks = []
    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience,
                                                       monitor='val_loss'))

        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(schedule, 1))

        if save_best:
            check = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='val_loss',
                                                verbose=verbose,
                                                save_best_only=save_best,
                                                mode='min')
            callbacks.append(check)

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
