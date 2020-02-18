import glob
import math
import os

import cv2
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    array_to_img,
    ImageDataGenerator
)
from keras.layers import (
    Conv2D,
    Dense,
    Input,
    MaxPool2D,
    UpSampling2D,
    Lambda,
    Conv2DTranspose,
)
from keras.models import (
    Model,
    Sequential,
)


import numpy as np

data_path = ''

data_list = glob.glob(os.path.join(data_path, '*.jpg'))

val_n_sample = math.floor(len(data_list) * 0.1)
test_n_sample = math.floor(len(data_list) * 0.1)
train_n_sample = len(data_list) - val_n_sample - test_n_sample

val_lists = data_list[:val_n_sample]
test_lists = data_list[val_n_sample:val_n_sample + test_n_sample]
train_lists = data_list[val_n_sample + test_n_sample:val_n_sample + test_n_sample + train_n_sample]


img_size = 224

def rgb_to_lab(rgb):
    assert rgb.type == 'uint8'
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def lab_to_rgb(lab):
    assert lab.type == 'uint8'
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


def get_lab_from_data_list(data_list):
    x_lab = []
    for f in data_list:
        rgb = img_to_array(load_img(f, target_size=(img_size, img_size))).astype(np.uint8)
        x_lab.append(rgb_to_lab(rgb))
    return np.stack(x_lab)


def build_model():
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