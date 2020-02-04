from load_data import LoadMNISTData
from model import DAE

class TrainDAE:

    def __init__(self):
        self.autoencoder, init_weights = DAE().build_model()

    def run(self, kind, debug=False):
        (x_train, x_test), (noised_x_train, noised_x_test) = LoadMNISTData().run(kind)
        self._train(noised_x_train, x_train)
        noised_preds = self._predict(noised_x_test)
        self.save(x_test, noised_x_test, noised_preds)

    def _train(self, X, Y):
        self.autoencoder.fit(
            X,
            Y,
            epochs=10,
            batch_size=20,
            shuffle=True,
        )

    def _predict(self, noised_x_test):
        return self.autoencoder.predict(noised_x_test)

    def save(self, x_test, noised_x_test, noised_preds):
        pass

if __name__ == "__main__":
    TrainDAE().run('masked')