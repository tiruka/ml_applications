import glob
import os
from sys import argv


from train_predict import TrainDAE, PredictDAE
import settings

class Initializer:

    def cleanup(self):
        self._clean_files('png')
        self._clean_files('h5')

    def _clean_files(self, mode='png'):
        targets = glob.glob(os.path.join(settings.VAR_DIR, '**', f'*.{mode}'), recursive=True)
        for t in targets:
            if os.path.isfile(t):
                os.remove(t)

is_mnist = False
if __name__ == "__main__":
    if len(argv) < 1:
        raise Exception('Please add args')
    args = frozenset(argv)
    if 'mnist' in args:
        is_mnist = True
    if 'train' in args:
        Initializer().cleanup()
        if 'masked' in args:
            TrainDAE('masked', is_mnist=is_mnist).train()
        elif 'gaussian' in args:
            TrainDAE('gaussian', is_mnist=is_mnist).train()
        elif 'both' in args:
            TrainDAE('both', is_mnist=is_mnist).train()
    elif 'predict' in args:
        PredictDAE().predict(argv[2])