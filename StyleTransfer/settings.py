import os

BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__name__)))
VAR_DIR = os.path.join(BASE_DIR, 'var')
DEBUG_IMG = os.path.join(VAR_DIR, 'debug_img')
COMPARABLE_IMG = os.path.join(VAR_DIR, 'comparable_img')
MODEL = os.path.join(VAR_DIR, 'model')
LOG = os.path.join(VAR_DIR, 'log')
DATA = os.path.join(BASE_DIR, 'data', 'train')
STYLE_IMAGE = os.path.join(BASE_DIR, 'data', 'style', 'style.jpg')
TEST_IMAGE = os.path.join(BASE_DIR, 'data', 'test', 'test.jpg')

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 2