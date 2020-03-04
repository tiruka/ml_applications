import os

BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__name__)))
VAR_DIR = os.path.join(BASE_DIR, 'var')
DEBUG_IMG = os.path.join(VAR_DIR, 'debug_img')
COMPARABLE_IMG = os.path.join(VAR_DIR, 'comparable_img')
MODEL = os.path.join(VAR_DIR, 'model')
DATA = os.path.join(BASE_DIR, 'data')

SIZE = (200, 200)
BATCH_SIZE = 32