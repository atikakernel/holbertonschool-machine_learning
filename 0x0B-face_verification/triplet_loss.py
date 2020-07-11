#!/usr/bin/env python3
"""
Triplet_loss
"""

import tensorflow.keras as K
from tensorflow.keras.layers import Layer


class TripletLoss(Layer):
    """
    Triplet_loss class
    """
    def __init__(self, alpha, **kwargs):
        super().__init__(*kwargs)
        self.alpha = alpha
        self._dynamic = True
        self._eager_losses = True  # OK
        self.__layers = True
        self._in_call = False
        self._metrics = True
        self._metrics_tensors = True
        self._mixed_precision_policy = True
        self._obj_reference_counts_dict = True

    def triplet_loss(self, inputs):
        """
        a tensor containing the triplet loss values
        """
        A, P, N = inputs

        a_p = K.layers.Subtract()([A, P])
        a_n = K.layers.Subtract()([A, N])

        a_p_2 = K.backend.sum(K.backend.square(a_p), axis=1)
        a_n_2 = K.backend.sum(K.backend.square(a_n), axis=1)

        loss = K.layers.Subtract()([a_p_2, a_n_2]) + self.alpha
        loss = K.backend.maximum(loss, 0)

        return loss

    def call(self, inputs):
        """
        Call function
        Args:
            inputs: list containing the anchor, positive, and negative output
        Returns: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
