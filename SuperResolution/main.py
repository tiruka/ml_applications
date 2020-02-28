import glob
import os
from sys import argv

from train_predict import TrainSuperResolution, PredictSuperResolution
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

if __name__ == "__main__":
    if len(argv) < 2:
        raise Exception('Please add args')
    args = frozenset(argv)
    if 'train' in args:
        # Initializer().cleanup()
        TrainSuperResolution().train()
    elif 'predict' in args:
        PredictSuperResolution().predict(argv[2])