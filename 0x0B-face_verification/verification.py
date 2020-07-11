#!/usr/bi/env python3
"""
Verification face recognition
"""

import tensorflow.keras as K
import tensorflow as tf
import numpy as np


class FaceVerification:
    def __init__(self, model, database, identities):
        with K.utils.CustomObjectScope({'tf': tf}):
            self.model = K.models.load_model(model)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
        a numpy.ndarray of shape (i, e) containing the embeddings
        where e is the dimensionality of the embeddings
        """
        embeddings = np.zeros((images.shape[0], 128))
        for i, m in enumerate(images):
            embeddings[i] = self.model.predict(np.expand_dims(m, axis=0))[0]
        return embeddings

    def verify(self, image, tau=0.5):
        """
        Returns: (identity, distance), or (None, None) on failure
        * identity is a string containing the identity of the verified face
        * distance is the euclidean distance between the verified face
          embedding and the identified database embedding
        """
        n, _, c = image.shape
        image = image.reshape(1, n, n, c)
        process = self.embedding(image)

        distances = []
        for elem in self.database:
            dist = np.square(elem - process)
            dist = np.sum(dist)
            distances.append(dist)

        best_dist = [elem for elem in distances if elem <= tau]
        if len(best_dist) == 0:
            return (None, None)
        min_dist = min(best_dist)
        index = best_dist.index(min_dist)
        identity = self.identities[index]
        return(identity, min_dist)
