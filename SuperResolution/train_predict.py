import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from load_data import DataLoader
from model import SuperResolution, SuperResolutionWithSkipConnections

import settings
from utils import drop_resolution

class ImageStore:

    def save(self, num, np_img_list):
        comparable_img = self._get_concat_horizontal_multi_resize(np_img_list)
        comparable_img.save(os.path.join(settings.COMPARABLE_IMG, f'{num}_comparable.jpg'))

    def _get_concat_horizontal_multi_resize(self, np_img_list, resample=Image.BICUBIC):
        im_list = [array_to_img(x) for x in np_img_list]
        # for i, im in enumerate(im_list):
        #     print(i, im.format, im.size, im.mode, im.getextrema())
        min_height = min(im.height for im in im_list)
        im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height), resample=resample)
                        for im in im_list]
        total_width = sum(im.width for im in im_list_resize)
        dst = Image.new('RGB', (total_width, min_height))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst

class TrainSuperResolution(SuperResolution):

    def __init__(self, epochs=10):
        self.model = self.build_model()
        self.loader = DataLoader()
        self.epochs = epochs
        self.model_name = 'super_resolution_model'

    def train(self):
        train_gen, val_gen = self.loader.load_data()
        train_steps, val_steps = self.loader.pre_calculation()
        self._train(train_gen, train_steps, val_gen, val_steps)

    def _train(self, train_gen, train_steps, val_gen, val_steps):
        self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
        )
        self.model.save_weights(os.path.join(settings.MODEL, f'{self.model_name}.h5'))


class TrainHyperResolution(TrainSuperResolution, SuperResolutionWithSkipConnections):
    def __init__(self, epochs=10):
        super().__init__(epochs)
        self.model_name = 'hyper_resolution_model'


class PredictSuperResolution(SuperResolution, ImageStore):

    def __init__(self):
        self.model = self.build_model()
        self.model_name = 'super_resolution_model'
        self._load_model()

    def predict(self, path):
        original = self._load_img(path)
        X = drop_resolution(original[0])
        X = np.expand_dims(X, axis=0)
        preds = self._predict(X)
        for i in range(1):
            np_img_list = [original[i], X[i], preds[i]]
            self.save(i, np_img_list)

    def val_predict(self):
        loader = DataLoader()
        _, val_gen = loader.load_data()
        for i, (val_x, val_y) in enumerate(val_gen):
            preds = self._predict(val_x)
            np_img_list = [val_y[0], val_x[0], preds[0]]
            self.save(i, np_img_list)
            if i >= 3:
                break

    def _predict(self, X):
        return self.model.predict(X)

    def _load_model(self):
        self.model.load_weights(os.path.join(settings.MODEL, f'{self.model_name}.h5'))

    def _load_img(self, path):
        img_np = img_to_array(load_img(path))
        return np.expand_dims(img_np, axis=0) / 255.


class PredictHyperResolution(PredictSuperResolution, SuperResolutionWithSkipConnections):

        def __init__(self):
            super().__init__()
            self.model_name = 'hyper_resolution_model'