from PIL import Image
from keras.preprocessing.image import array_to_img
from load_data import LoadMNISTData
from model import DAE

class TrainDAE:

    def __init__(self):
        self.autoencoder, init_weights = DAE().build_model()

    def run(self, kind, debug=False):
        (x_train, x_test), (noised_x_train, noised_x_test) = LoadMNISTData().run(kind)
        self._train(noised_x_train, x_train)
        noised_preds = self._predict(noised_x_test)
        for i in range(10):
            self.save(i, x_test[i], noised_x_test[i], noised_preds[i])

    def _train(self, X, Y):
        self.autoencoder.fit(
            X,
            Y,
            epochs=1,
            batch_size=20,
            shuffle=True,
        )

    def _predict(self, noised_x_test):
        return self.autoencoder.predict(noised_x_test)

    def save(self, num, x_test, noised_x_test, noised_preds):
        comparable_img = self._get_concat_horizontal_multi_resize([x_test, noised_x_test, noised_preds])
        comparable_img.save(f'./comparable_img/{num}_original.png')

    def _get_concat_horizontal_multi_resize(self, im_list, resample=Image.BICUBIC):
        im_list = [array_to_img(x) for x in im_list]
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
