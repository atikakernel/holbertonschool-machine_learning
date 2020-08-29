"""
All layers should use a relu activation except for the last layer in
the decoder, which should use sigmoid
"""
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Autoencoder Sparse
    """

    nod = len(hidden_layers)

    inimg = K.layers.Input(shape=(input_dims,))

    enco = K.layers.Dense(hidden_layers[0], activation='relu')(inimg)

    for i in range(1, nod):
        enco = K.layers.Dense(hidden_layers[i], activation='relu')(enco)

    llat = K.layers.Dense(latent_dims, activation='relu',
                          activity_regularizer=regularizers.
                          l1(lambtha))(encoded)

    encoder = K.models.Model(input_img, llat)

    input_dec = K.layers.Input(shape=(latent_dims,))

    deco = K.layers.Dense(hidden_layers[i], activation='relu',
                          activity_regularizer=regularizers.
                          l1(lambtha))(input_dec)

    for i in range(nod-2, 0, -1):
        deco = K.layers.Dense(hidden_layers[i], activation='relu')(deco)

    deco = K.layers.Dense(input_dims, activation='sigmoid')(deco)

    decoder = K.models.Model(input_dec, deco)

    enc_out = encoder(inimg)

    dec_out = decoder(enc_out)

    auto = K.models.Model(inimg, dec_out)

    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return(encoder, decoder, auto)
