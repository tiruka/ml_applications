import glob
import math
import os

import numpy as np
from sklearn import model_selection
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    array_to_img,
    ImageDataGenerator
)

import settings
from utils import drop_resolution

class DataLoader(object):

    def __init__(self):
        self.batch_size = settings.BATCH_SIZE
        self.img_train_iters = self.img_validation_iters = None

    def load_data(self):
        self.set_iters()
        train_gen = self.data_generator(self.img_train_iters)
        val_gen = self.data_generator(self.img_validation_iters)
        return train_gen, val_gen

    def set_iters(self):
        self.img_train_iters = self._create_img_iters(settings.DATA, 'train')
        self.img_validation_iters = self._create_img_iters(settings.DATA, 'val', shuffle=False)

    def pre_calculation(self):
        self.steps_per_epoch = self.cal_steps_for_epoch(self.img_train_iters)
        self.validation_steps = self.cal_steps_for_epoch(self.img_validation_iters)
        return self.steps_per_epoch, self.validation_steps

    def cal_steps_for_epoch(self, iters):
        return math.ceil(iters.samples / self.batch_size)

    def _create_img_iters(self, data_dir, mode, shuffle=True):
        return ImageDataGenerator().flow_from_directory(
            directory=data_dir,
            classes=[mode],
            class_mode=None,
            color_mode='rgb',
            target_size=settings.SIZE,
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def data_generator(self, iters, scale=2.0, shuffle=True):
        for images in iters:
            x = np.array([drop_resolution(img, scale) for img in images])
            yield x / 255., images / 255.