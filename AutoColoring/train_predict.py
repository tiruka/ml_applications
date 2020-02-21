import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from load_data import DataLoader
from model import AutoColorEncoder

import settings
from utils import lab_to_rgb, rgb_to_lab


class ImageStore:

    def save(self, num, np_img_list):
        comparable_img = self._get_concat_horizontal_multi_resize(np_img_list)
        comparable_img.save(os.path.join(settings.COMPARABLE_IMG, f'{num}_comparable.png'))

    def _get_concat_horizontal_multi_resize(self, np_img_list, resample=Image.BICUBIC):
        im_list = [array_to_img(x) for x in np_img_list]
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

class TrainAutoColor(AutoColorEncoder, ImageStore):

    def __init__(self, epochs=100):
        self.autoencoder = self.build_model()
        self.loader = DataLoader()
        self.epochs = epochs

    def train(self):
        train_gen, val_gen, test_gen = self.loader.load_data()
        train_steps, val_steps, test_steps = self.loader.cal_steps()
        self._train(train_gen, train_steps, val_gen, val_steps)
        self._predict(test_gen, test_steps, self.loader.test_list, self.loader.batch_size)

    def _train(self, train_gen, train_steps, val_gen, val_steps):
        self.autoencoder.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
        )
        self.autoencoder.save_weights(os.path.join(settings.MODEL, 'auto_color_model.h5'))

    def _predict(self, test_gen, test_steps, test_lists, batch_size):
        preds = self.autoencoder.predict_generator(test_gen, steps=test_steps, verbose=0)
        x_test = []
        y_test = []
        for i, (l, ab) in enumerate(self.loader.generator_with_preprocessing(test_lists, batch_size)):
            x_test.append(l)
            y_test.append(ab)
            if i == test_steps - 1:
                break
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test) 

        test_preds_lab = np.concatenate((x_test, preds), 3).astype(np.uint8)
        test_preds_rgb = []
        for i in range(test_preds_lab.shape[0]):
            preds_rgb = lab_to_rgb(test_preds_lab[i, :, :, :])
            test_preds_rgb.append(preds_rgb)
        test_preds_rgb = np.stack(test_preds_rgb)

        original_lab = np.concatenate((x_test, y_test), 3).astype(np.uint8)
        original_rgb = []
        for i in range(original_lab.shape[0]):
            original_rgb.append(lab_to_rgb(original_lab[i, :, :, :]))
        original_rgb = np.stack(original_rgb)

        for i in range(test_preds_rgb.shape[0]):
            gray_image = img_to_array(ImageOps.grayscale(array_to_img(test_preds_rgb[i])))
            auto_colored_image = test_preds_rgb[i]
            original_image = original_rgb[i]
            np_img_list = [gray_image, auto_colored_image, original_image]
            self.save(i, np_img_list)


class PredictAutoColor(AutoColorEncoder, ImageStore):
    
    def __init__(self):
        self.autoencoder = self.build_model()

    def predict(self, path):
        rgb = img_to_array(load_img(path, target_size=settings.SIZE)).astype(np.uint8) # shape (224, 224, 3)
        x_lab = [rgb_to_lab(rgb)] 
        lab = np.stack(x_lab) # shape (1, 224, 224, 3)
        l = lab[:, :, :, 0:1] # shape (1, 224, 224, 1)
        preds = self.autoencoder.predict(l, verbose=0)
        preds_lab = np.concatenate((l, preds), 3).astype(np.uint8)
        preds_rgp = lab_to_rgb(preds_lab[0, :, :, :])

        gray_image = img_to_array(ImageOps.grayscale(array_to_img(rgb)))
        auto_colored_image = preds_rgp
        np_img_list = [gray_image, auto_colored_image, rgb]
        self.save(datetime.now().strftime('%Y%m%d%H%M%S'), np_img_list)
