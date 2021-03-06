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

from model import Contents, Style
import settings

class DataLoader(object):

    def __init__(self):
        self.batch_size = settings.BATCH_SIZE
        self.model_contents = Contents().build_model()
        self.model_style = Style()

    def load_data(self):
        y_true_style = self.model_style.predict()
        image_path_list = glob.glob(os.path.join(settings.DATA, '*.jpg'))
        gen = self.train_generator(
            image_path_list,
            y_true_style,
            epochs=10,
        )
        return gen, image_path_list


    def train_generator(self, image_path_list, y_true_style, shuffle=True, epochs=10):
        '''
        Generate train data
        '''
        n_samples = len(image_path_list)
        indices = [i for i in range(n_samples)]
        steps_per_epochs = math.ceil(n_samples / settings.BATCH_SIZE)
        image_path_ndarray = np.array(image_path_list)
        count_epochs = 0
        while True:
            count_epochs += 1
            if shuffle:
                np.random.shuffle(indices)
            for i in range(steps_per_epochs):
                start = settings.BATCH_SIZE * i
                end = settings.BATCH_SIZE * (i + 1)
                X = self.load_images(image_path_ndarray[indices[start:end]])
                batch_size_act = X.shape[0]
                y_true_style_t = [np.repeat(feat, batch_size_act, axis=0) for feat in y_true_style]
                y_true_contents = self.model_contents.predict(X)
                yield X, y_true_style_t + [y_true_contents]
            if epochs is not None:
                if count_epochs >= epochs:
                    raise StopIteration

    def load_images(self, image_path_list):
        '''
        Return batch of array from image_path_list
        '''
        _load_img = lambda x: img_to_array(load_img(x, target_size=settings.INPUT_SHAPE[:2]))
        image_list = [np.expand_dims(_load_img(path), axis=0) for path in image_path_list]
        return np.concatenate(image_list, axis=0)