import glob
import math
import os

from PIL import Image
import numpy as np
from sklearn import model_selection
from keras.datasets import mnist
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    array_to_img,
    ImageDataGenerator
)

import settings
from utils import rgb_to_lab, lab_to_rgb

class DataLoader(object):

    def __init__(self):
        self.data_list = glob.glob(os.path.join(settings.DATA, '*.jpg'))
        self.val_n_sample = math.floor(len(self.data_list) * 0.1)
        self.test_n_sample = math.floor(len(self.data_list) * 0.1)
        self.train_n_sample = len(self.data_list) - self.val_n_sample - self.test_n_sample
        self.batch_size = settings.BATCH_SIZE
        self.train_list, self.val_list, self.test_list = self._generate_cross_validation_data(self.data_list)

    def load_data(self):
        train_gen = self.generator_with_preprocessing(self.train_list, self.batch_size, shuffle=True)
        val_gen = self.generator_with_preprocessing(self.val_list, self.batch_size)
        test_gen = self.generator_with_preprocessing(self.test_list, self.batch_size)
        return train_gen, val_gen, test_gen

    def cal_steps(self):
        train_steps = self._cal_steps(self.train_list)
        val_steps = self._cal_steps(self.val_list)
        test_steps = self._cal_steps(self.test_list)
        return train_steps, val_steps, test_steps

    def _generate_cross_validation_data(self, data_list):
        val_list = data_list[:self.val_n_sample]
        test_list = data_list[self.val_n_sample:self.val_n_sample + self.test_n_sample]
        train_list = data_list[self.val_n_sample + self.test_n_sample:self.val_n_sample + self.test_n_sample + self.train_n_sample]
        return val_list, test_list, train_list

    def _cal_steps(self, data_list):
        return math.ceil(len(data_list) / self.batch_size)

    def generator_with_preprocessing(self, data_list, batch_size, shuffle=False):
        while True:
            if shuffle:
                np.random.shuffle(data_list)
            for i in range(0, len(data_list), batch_size):
                batch_list = data_list[i:i + batch_size]
                batch_lab = self.get_lab_from_data_list(batch_list)
                batch_l = batch_lab[:, :, :, 0:1]
                batch_ab = batch_lab[:, :, :, 1:]
                yield (batch_l, batch_ab)

    def get_lab_from_data_list(self, data_list):
        x_lab = []
        for f in data_list:
            rgb = img_to_array(load_img(f, target_size=settings.SIZE)).astype(np.uint8)
            x_lab.append(rgb_to_lab(rgb))
        return np.stack(x_lab)

    def _change_png(self, image):
        img = Image.open(image)
        img_resize = img.resize(settings.SIZE)
        title, ext = os.path.splitext(image)
        img_resize.save(title + 'resized' +'.png')
        os.remove(image)

    def _change_extensions(self):
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        for ext in extensions:
            other_images = glob.glob(os.path.join(settings.DATA, f'*.{ext}'))
            for fp in other_images:
                self._change_png(fp)