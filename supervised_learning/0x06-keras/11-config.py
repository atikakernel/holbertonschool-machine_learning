#!/usr/bin/env python3
""" builds a neural network with the Keras library"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves a model’s configuration in JSON format:"""
    json_net = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_net)
    return None


def load_config(filename):
    """ loads a model with a specific configuration:"""
    with open(filename, 'r') as file:
        return K.models.model_from_json(file.read())
