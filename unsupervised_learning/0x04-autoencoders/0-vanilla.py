"""
All layers should use a relu activation except for the last layer in
the decoder, which should use sigmoid
"""
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Autoencoder Vanilla
    """
    nod = len(hidden_layers)

    inimg = K.layers.Input(shape=(input_dims,))

    enco = K.layers.Dense(hidden_layers[0], activation='relu')(inimg)

    for i in range(1, nod):
        enco = K.layers.Dense(hidden_layers[i], activation='relu')(enco)

    lay_latent = K.layers.Dense(latent_dims, activation='relu')(enco)

    der = K.models.Model(inimg, lay_latent)

    input_dec = K.layers.Input(shape=(latent_dims,))

    decoded = K.layers.Dense(hidden_layers[i], activation='relu')(input_dec)

    for i in range(nod-2, 0, -1):
        decoded = K.layers.Dense(hidden_layers[i], activation='relu')(decoded)

    decoded = K.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = K.models.Model(input_dec, decoded)

    enc_out = der(inimg)

    dec_out = decoder(enc_out)

    auto = K.models.Model(inimg, dec_out)

    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return(der, decoder, auto)
