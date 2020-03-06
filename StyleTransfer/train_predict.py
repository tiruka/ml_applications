import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from load_data import DataLoader
from model import StyleTransfer

import settings


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

class TrainStyleTransfer(StyleTransfer, ImageStore):

    def __init__(self, epochs=10):
        self.model = self.build_model()
        self.loader = DataLoader()
        self.epochs = epochs

    def train(self):
        gen = self.loader.load_data()
        self._train(gen)

    def _train(self, gen):
        img_test = load_img(path, target_size=input_shape[:2])
        img_arr_test = np.expand_dims(img_to_array(img_test), axis=0)
        steps_per_epochs = math.ceil(len(image_path_list) // settings.BATCH_SIZE)
        iters_vobose = 1000
        iters_save_img = 1000
        iters_save_model = 1000

        cur_epoch = 0
        losses = []
        path_tmp = 'epoch_{}_iters_{}_loss_{:.2f}_{}'
        for i, (x_train, y_train) in enumerate(gen):
            if i % steps_per_epochs == 0:
                cur_epoch += 1
            loss = model.train_on_batch(x_train, y_train)
            losses.append(loss)
            if i % iters_vobose == 0:
                print('epoch:{}\titers:{}\tloss:{:.2f}'.format(cur_epoch, i, loss[0]))
            if i % iters_save_img == 0:
                pred = model_gen.predict(img_arr_test)
                img_pred = array_to_img(pred.squeeze())
                path_trs_img = path_tmp.format(cur_epoch, i, loss[0], '.jpg')
                img_pred.save(os.path.join(setting.DEBUG_IMG, path_trs_img))
                print('saved {}'.format(path_trs_img))
            if i % iters_save_model == 0:
                model.save(os.path.join(settings.MODEL, path_tmp.format(cur_epoch, i, loss[0], '.h5')))
                path_loss = os.path.join(settings.LOG, 'loss.pkl')
                with open(path_loss, 'wb') as f:
                    pickle.dump(losses, f)


class PredictStyleTransfer(StyleTransfer, ImageStore):

    def __init__(self):
        self.model = self.build_model()

    def predict(self, path):
        X = self._load_img(path)
        preds = self._predict(X)
        for i in range(1):
            np_img_list = [X[i], preds[i]]
            self.save(i, np_img_list)
    
    def _predict(self, X):
        return self.model.predict(X)

    def _load_model(self):
        self.model.load_weights(os.path.join(settings.MODEL, 'super_resolution_model.h5'))

    def _load_img(self, path):
        img_np = img_to_array(load_img(path, target_size=(settings.SIZE)))
        return np.expand_dims(img_np, axis=0) / 255
