from keras.layers import (
    Conv2D,
    Dense,
    Input,
    MaxPool2D,
    UpSampling2D,
    Lambda,
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
        autoencoder.add(Conv2D(32, (3, 3), (1, 1), activation='relu', padding='same', input_shape=(img_size, img_size, 1)))
        autoencoder.add(Conv2D(64, (3, 3), (2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2D(128, (3, 3), (2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2D(256, (3, 3), (2, 2), activation='relu', padding='same'))
        # Decoder Part
        autoencoder.add(Conv2DTranspose(128, (3, 3), (2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2DTranspose(64, (3, 3), (2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2DTranspose(32, (3, 3), (2, 2), activation='relu', padding='same'))
        autoencoder.add(Conv2DTranspose(2, (1, 1), (1, 1), activation='relu', padding='same'))

        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()
        return autoencoder