import numpy as np
from keras.layers import Conv2D, Input, Add, Conv2DTranspose
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
        model.add(Conv2D(filters=64, kernel_size=9, padding='same', activation='relu', input_shape=(None, None, 3)))
        model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='relu'))
        model.add(Conv2D(filters=3, kernel_size=5, padding='same'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.psnr])
        # model.summary()
        return model

    def psnr(self, y_true, y_pred):
        # Estimation Funciton: Peak Signal-to-Noise Ratio
        # The bigger PSNR is, the better.
        # Roughly the range of PSNR is from 20dB to 50dB
        return -10 * K.log(
            K.mean(K.flatten((y_true - y_pred)) ** 2)
            ) / np.log(10)


class SuperResolutionWithSkipConnections(SuperResolution):
    '''
    Using Skip Connections for Super Resolution
    '''
    def build_model(self):
        inputs = Input((None, None, 3), dtype='float')
        # Encoder
        conv1 = Conv2D(filters=64, kernel_size=3, padding='same')(inputs)
        conv1 = Conv2D(filters=64, kernel_size=3, padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=3, padding='same')(conv2)
        conv3 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(conv2)
        conv3 = Conv2D(filters=64, kernel_size=3, padding='same')(conv3)
        # Decoder
        deconv3 = Conv2DTranspose(filters=64, kernel_size=3, padding='same')(conv3)
        deconv3 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(deconv3)
        merge_2 = Add()([deconv3, conv2]) # Skip Connections
        deconv2 = Conv2DTranspose(filters=64, kernel_size=3, padding='same')(merge_2)
        deconv2 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(deconv2)
        merge_1 = Add()([deconv2, conv1]) # Skip Connections
        deconv1 = Conv2DTranspose(filters=64, kernel_size=3, padding='same')(merge_1)
        deconv1 = Conv2DTranspose(filters=3, kernel_size=3, padding='same')(deconv1)
        outputs = Add()([deconv1, inputs])
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.psnr])
        # model.summary()
        return model