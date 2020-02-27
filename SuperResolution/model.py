import numpy as np
from keras.layers import Conv2D
from keras import backend as K
from keras.models import (
    Model,
    Sequential,
)

import settings

class SuperResolution(object):
    '''
    SuperResolution CNN Model
    '''

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=9, activation='relu', padding='same', input_shape=(None, None, 3)))
        model.add(Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')
        model.add(Conv2D(filters=3, kernel_size=5, activation='relu', padding='same')
        model.compile(optimizer='adam', loss=[self.psnr])
        model.summary()
        return model

    def psnr(self, y_true, y_pred):
        # Estimation Funciton: Peak Signal-to-Noise Ratio
        # The bigger PSNR is, the better.
        # Roughly the range of PSNR is from 20dB to 50dB
        return -10 * K.log(
            K.mean(K.flatten((y_true - y_pred) ** 2))
            ) / np.log(10)