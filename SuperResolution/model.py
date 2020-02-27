from keras.layers import (
    Conv2D,
    Conv2DTranspose,
)
from keras.models import (
    Model,
    Sequential,
)

import settings

class AutoColorEncoder(object):
    '''
    Convolutional AutoEncoder Model
    '''

    def build_model(self):
        autoencoder = Sequential()
        # Encoder Part
        autoencoder.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(*settings.SIZE, 1)))
        autoencoder.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        # Decoder Part
        autoencoder.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2DTranspose(filters=2, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same'))

        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()
        return autoencoder