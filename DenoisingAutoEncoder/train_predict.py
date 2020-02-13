import os

from PIL import Image
from keras.preprocessing.image import array_to_img, load_img
from load_data import DataLoader
from model import DAE

import settings


class ImageStore:

    def save(self, num, np_img_list):
        comparable_img = self._get_concat_horizontal_multi_resize(np_img_list)
        comparable_img.save(os.path.join(settings.COMPARABLE_IMG, f'{num}_original.png'))

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

class TrainDAE(DAE, ImageStore):

    def __init__(self, mode, is_mnist=False):
        self.mode = mode
        self.autoencoder, _ = self.build_model()
        self.is_mnist = is_mnist

    def train(self, debug=False):
        (x_train, x_test), (noised_x_train, noised_x_test) = DataLoader().run(mode=self.mode, is_mnist=self.is_mnist)
        self._train(noised_x_train, x_train)
        noised_preds = self._predict(noised_x_test)
        for i in range(10):
            np_img_list = [x_test[i], noised_x_test[i], noised_preds[i]]
            self.save(i, np_img_list)

    def _train(self, X, Y):
        self.autoencoder.fit(
            X,
            Y,
            epochs=1,
            batch_size=20,
            shuffle=True,
        )
        self.autoencoder.save_weights(os.path.join(settings.MODEL, f'{self.mode}_gae_model.h5'))

    def _predict(self, noised_x_test):
        return self.autoencoder.predict(noised_x_test)


class PredictDAE(DAE, ImageStore):
    
    def __init__(self):
        self.autoencoder, _ = self.build_model()

    def predict(self, path):
        X = self._load_img(path)
        preds = self._predict(X)
        for i in range(10):
            np_img_list = [X[i], preds[i]]
            self.save(i, np_img_list)
    
    def _predict(self, X):
        return self.autoencoder.predict(X)

    def _load_model(self):
        self.autoencoder.load_weights(os.path.join(settings.MODEL, f'{self.mode}_gae_model.h5'))

    def _load_img(self, path):
        img_np = img_to_array(load_img(path, target_size=(settings.SIZE)))
        return np.expand_dims(img_np, axis=0) / 255
