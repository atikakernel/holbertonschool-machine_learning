"""
All layers should use a relu activation except for the last layer in
the decoder, which should use sigmoid
"""
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def autoencoder(input_dims, filters, latent_dims):
    """
    Convolutional Autoencoder
    """

    size = len(filters)

    input_img = K.layers.Input(shape=input_dims)

    x = Conv2D(filters[0], (3, 3), activation='relu',
               padding='same')(input_img)

    x = MaxPooling2D((2, 2), padding='same')(x)

    for i in range(1, size-2):
        x = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(filters[size-1], (3, 3), activation='relu', padding='same')(x)

    lay_latent = MaxPooling2D((2, 2), padding='same')(x)

    enco = K.models.Model(input_img, lay_latent)

    input_dec = K.layers.Input(shape=latent_dims)

    y = Conv2D(8, (3, 3), activation='relu', padding='same')(input_dec)

    y = UpSampling2D((2, 2))(y)

    for i in range(size-2, 0, -1):
        y = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(y)
        y = MaxPooling2D((2, 2), padding='same')(y)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)

    deco = K.models.Model(input_dec, decoded)

    enc_out = enco(input_img)

    dec_out = deco(enc_out)

    auto = K.models.Model(input_img, dec_out)

    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return(enco, deco, auto)
