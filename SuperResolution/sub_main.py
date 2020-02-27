import os
import glob
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.python.keras.layers import Add, Input, Conv2D, Conv2DTranspose, Dense, Input, MaxPooling2D, UpSampling2D, Lambda

import settings

def drop_resolution(x, scale=3.0):
    # resize to small and resize to original in order to drop resolution easily.
    small_size = (int(x.shape[0] / scale), int(x.shape[1] / scale))
    img = array_to_img(x)
    small_img = img.resize(small_size)
    return img_to_array(small_img.rezie(img.size, 3))


def data_generator(data_dir, mode, scale=2.0, target_size=settings.SIZE, batch_size=32, shuffle=True):
    for images in ImageDataGenerator().flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle
    ):
        x = np.array([drop_resolution(img, scale) for img in images])
        yield x / 255., images / 255.


# should change later
N_TRAIN_DATA = 1000
N_TEST_DATA = 100
train_data_generator = data_generator(settings.DATA, 'train', batch_size=settings.BATCH_SIZE)
text_x, test_y = next(data_generator(settings.DATA, 'test', batch_size=N_TEST_DATA))


model.fit_generator(
    train_data_generator,
    validation_data=(text_x, test_y),
    steps_per_epochs=N_TRAIN_DATA // settings.BATCH_SIZE,
    epochs=50,
)

pred = model.predict(x)