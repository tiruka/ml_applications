import os

import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    array_to_img,
    ImageDataGenerator
)

import settings

class LoadMNISTData(object):

    def run(self, mode=None, debug=False):
        x_train, x_test = self.load_mnist_data()
        masked_x_train = self.make_masking_noise_data(x_train)
        masked_x_test = self.make_masking_noise_data(x_test)
        gaussian_x_train = self.make_gaussian_noise_data(x_train)
        gaussian_x_test = self.make_gaussian_noise_data(x_test)
        if debug:
            array_to_img(x_train[0]).save(os.path.join(setting.DEBUG_IMG, 'original.png'))
            array_to_img(masked_x_train[0]).save(os.path.join(setting.DEBUG_IMG, 'masked_noise.png'))
            array_to_img(gaussian_x_train[0]).save(os.path.join(setting.DEBUG_IMG, 'gaussian_noise.png'))
        if mode == 'masked':
            return (x_train, x_test), (masked_x_train, masked_x_test)
        elif mode == 'gaussian':
            return (x_train, x_test), (gaussian_x_train, gaussian_x_test)

    def load_mnist_data(self):
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = self._normalize(x_train)
        x_test = self._normalize(x_test)
        return x_train, x_test

    def _normalize(self, data):
        return data / 255

    def make_masking_noise_data(self, data, percentage=0.1):
        '''Masking data by partially changing to 0 randomly'''
        masking = np.random.binomial(n=1, p=percentage, size=data.shape)
        return data * masking

    def make_gaussian_noise_data(self, data, scale=0.8):
        '''Masking data by partially changing to 0 randomly'''
        gaussian_data = np.clip(
            data + np.random.normal(loc=0, scale=scale, size=data.shape),
            0,
            1,
        )
        return gaussian_data

if __name__ == "__main__":
    loader = LoadMNISTData()
    loader.run(debug=True)
