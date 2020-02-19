import glob
import math
import os

import cv2
from PIL import Image, ImageOps
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


def generator_with_preprocessing(data_list, batch_size, shuffle=False):
    while True:
        if shuffle:
            np.random.shuffle(data_list)
        for i in range(0, len(data_list, batch_size)):
            batch_list = data_list[i, i + batch_size]
            batch_lab = get_lab_from_data_list(batch_list)
            batch_l = batch_lab[:, :, :, 0:1]
            batch_ab = batch_lab[:, :, :, 1:]
            yield (batch_l, batch_ab)

batch_size = 30
train_gen = generator_with_preprocessing(train_lists, batch_size, shuffle=True)
val_gen = generator_with_preprocessing(val_lists, batch_size)
test_gen = generator_with_preprocessing(test_lists, batch_size)

train_steps = math.ceil(len(train_lists) / batch_size)
val_steps = math.ceil(len(val_lists) / batch_size)
test_steps = math.ceil(len(test_lists) / batch_size)


epochs = 10
autoencoder = build_model()
autoencoder.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=val_steps,
)
autoencoder.save_weights('paht/to/auto_color_model.h5')

preds = autoencoder.predict_generator(test_gen, steps=test_steps, verbose=0)
x_test = []
y_test = []
for i, (l, ab) in enumerate(generator_with_preprocessing(test_lists, batch_size)):
    x_test.append(l)
    y_test.append(ab)
    if i == test_steps - 1:
        break

x_test = np.vstack(x_test)
y_test = np.vstack(y_test) 

test_preds_lab = np.concatenate((x_test, preds), 3).astype(np.int8)

test_preds_rgb = []
for i in range(test_preds_lab.shape[0]):
    preds_rgb = lab_to_rgb(test_preds_lab[i, :, :, :])
    test_preds_rgb.append(preds_rgb)
test_preds_rgb = np.stack(test_preds_rgb)



for i in range(len(test_preds_rgb.shape[0])):
    gray_image = ImageOps.grayscale(array_to_img(test_preds_rgb[i]))