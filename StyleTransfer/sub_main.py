import os
import glob
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, BatchNormalization, Add, Activation

import settings



def residual_block(input_tensor):
    # Creating Residual Block
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([x, input_tensor])