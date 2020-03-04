#!/usr/bin/env python3
""" updates the learning rate using inverse time decay in numpy"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay in numpy"""
    return (tf.train.inverse_time_decay(alpha, global_step,
                                        decay_step, decay_rate,
                                        staircase=True))
