import os
import glob

from train import TrainDAE
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
    Initializer().cleanup()
    TrainDAE('masked').train()