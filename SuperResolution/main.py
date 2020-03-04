import glob
import os
from sys import argv

from train_predict import (
    TrainSuperResolution,
    PredictSuperResolution,
    TrainHyperResolution,
    PredictHyperResolution,
)
import settings


class Initializer:
    
    def cleanup(self):
        self._clean_files('png')
        self._clean_files('jpg')
        # self._clean_files('h5')

    def _clean_files(self, mode='png'):
        targets = glob.glob(os.path.join(settings.VAR_DIR, '**', f'*.{mode}'), recursive=True)
        for t in targets:
            if os.path.isfile(t):
                os.remove(t)

if __name__ == "__main__":
    if len(argv) < 2:
        raise Exception('Please add args')
    Initializer().cleanup()
    args = frozenset(argv)
    if 'train_super' in args:
        TrainSuperResolution(epochs=50).train()
    elif 'train_hyper' in args:
        TrainHyperResolution(epochs=50).train()
    elif 'predict_super' in args:
        PredictSuperResolution().predict(argv[2])
    elif 'predict_hyper' in args:
        PredictHyperResolution().predict(argv[2])
    elif 'test' in args:
        PredictSuperResolution().val_predict()
    elif 'enhance_super' in args:
        PredictSuperResolution().enhance(argv[2])
    elif 'enhance_hyper' in args:
        PredictHyperResolution().enhance(argv[2])