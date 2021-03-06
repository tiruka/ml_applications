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

class DAE(object):
    '''
    Convolutional AutoEncoder Model
    '''

    def build_model(self, is_mnist=False):
        if is_mnist:
            input_shape = (28, 28, 1)
        else:
            input_shape = (*settings.SIZE, 3)
        autoencoder = Sequential()
        # Encoder Part
        autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
        autoencoder.add(MaxPool2D((2, 2), padding='same'))
        autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        autoencoder.add(MaxPool2D((2, 2), padding='same'))
        # Decoder Part
        autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2, 2)))
        autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2, 2)))
        autoencoder.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same'))
        autoencoder.compile(
            optimizer='adam',
            loss='binary_crossentropy',
        )
        initial_weights = autoencoder.get_weights()
        # autoencoder.summary()
        return autoencoder, initial_weights

if __name__ == "__main__":
    DAE().build_model()