#!/usr/bin/env python3
"""
Object verification model
"""
import tensorflow as tf
import tensorflow.keras as K
from triplet_loss import TripletLoss
import numpy as np


class TrainModel():
    def __init__(self, model_path, alpha):
        """
        Initialize model
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        self.alpha = alpha

        A = K.Input(shape=(96, 96, 3))
        P = K.Input(shape=(96, 96, 3))
        N = K.Input(shape=(96, 96, 3))
        inputs = [A, P, N]

        X_a = self.base_model(A)
        X_p = self.base_model(P)
        X_n = self.base_model(N)
        encoded_input = [X_a, X_p, X_n]

        decoded = TripletLoss(alpha=alpha)(encoded_input)
        decoder = K.models.Model(inputs, decoded)

        self.training_model = decoder
        self.training_model.compile(optimizer='Adam')
        self.training_model.save

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """
        triplets is a list of numpy.ndarrayscontaining
            the inputs to self.training_model
        """
        history = self.training_model.fit(x=triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_split=validation_split,
                                          verbose=verbose)
        return history

    def save(self, save_path):
        """
        saves the base embedding model
        """
        K.models.save_model(self.base_model, save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Returns: The f1 score
        """
        TP = np.count_nonzero(y_pred * y_true)
        TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)
        sensitivity = TP / (TP + FN)
        precision = TP / (TP + FP)

        f1 = (2 * sensitivity * precision) / (sensitivity + precision)
        return f1

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Returns:  the accuracy
        """
        predicted = y_pred
        actual = y_true
        TP = np.count_nonzero(predicted * actual)
        TN = np.count_nonzero((predicted - 1) * (actual - 1))
        FP = np.count_nonzero(predicted * (actual - 1))
        FN = np.count_nonzero((predicted - 1) * actual)

        true_values = TP + TN
        all_values = TP + FP + FN + TN
        accuracy = true_values / all_values
        return accuracy

    @staticmethod
    def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

    def best_tau(self, images, identities, thresholds):
        """
        the accuracy associated with the maximal F1 score
        """

        distancias = []
        identicas = []
        pro_img = self.base_model.predict(images)

        for i in range(len(identities) - 1):
            for j in range(i + 1, len(identities)):
                dist = (np.square(pro_img[i] - pro_img[j]))
                dist = np.sum(dist)
                print(dist, identities[i], identities[j])
                distancias.append(dist)
                if identities[i] == identities[j]:
                    identicas.append(1)
                else:
                    identicas.append(0)

        distancias = np.array(distancias)
        identicas = np.array(identicas)

        f1_list = []
        acc_list = []

        for t in thresholds:
            mask = np.where(distancias <= t, 1, 0)
            f1 = self.f1_score(identicas, mask)
            acc = self.accuracy(identicas, mask)
            f1_list.append(f1)
            acc_list.append(acc)

        f1_max = max(f1_list)
        index = f1_list.index(f1_max)
        acc_max = acc_list[index]
        tau = thresholds[index]

        return(tau, f1_max, acc_max)
