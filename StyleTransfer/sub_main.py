import os
import glob
import math
import pickle
import random

import numpy as np
from keras import backend as K
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Lambda, BatchNormalization, Conv2D, BatchNormalization, Add, Activation, Input, Conv2DTranspose
from keras.applications import VGG16
from keras.optimizers import Adadelta

import settings

input_shape = (224, 224, 3)


def residual_block(input_tensor):
    # Creating Residual Block
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([x, input_tensor])


def build_encoder_decoder():
    # Encoder
    input_tensor = Input(shape=input_shape, name='input')
    x = Lambda(lambda x: x / 255.)(input_tensor) # normalization [0, 1]
    x = Conv2D(filters=32, kernel_size=(9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same')(x)
    for _ in range(5): # Add five residual blocks
        x = residual_block(x)
    # Decoder
    x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tahn')(x)
    outputs = Lambda(lambda x: (x + 1) * 127.5) # transform outputs to [0, 255]
    model_gen = Model(inputs=input_tensor, outputs=outputs)
    return model_gen

model_gen = build_encoder_decoder()

vgg16 = VGG16()
for layer in vgg16.layers:
    layyer.trainable = False

def norm_vgg16(x):
    '''
    RGB -> BGR transformation and approximate centerlization
    '''
    return (x[:, :, :, ::-1] - 120) / 255.

style_layer_names = frozenset(['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3'])
contents_layer_names = frozenset(['block3_conv3'])
style_outputs_gen = []
contents_outputs_gen = []

input_gen = model_gen.output # inputs will come from stlyle-transform model
z = Lambda(norm_vgg16)(input_gen)
for layer in vgg16.layers: # Recontructing network by piling up 
    z = layer(z)
    if layer.name in style_layer_names:
        style_outputs_gen.append(z)
    if layer.name in contents_layer_names:
        contents_outputs_gen.append(z)

model = Model(inputs=model_gen.input, outputs=style_outputs_gen + contents_outputs_gen)


img_arr_style = np.expand_dims(array_to_img(load_img(settings.Style, target_size=input_shape[:2])), axis=0)
style_outputs = []
input_style = Input(shape=input_shape, name='input_style')
x = Lambda(norm_vgg16)(input_style)
for layer in vgg16.layers:
    x = layer(x)
    if layer.name in style_layer_names:
        style_outputs.append(x)

model_style = Model(inputs=input_style, outputs=style_outputs)
y_true_style = model_style.predict(img_arr_style)


input_contents = Input(shape=input_shape, name='input_contents')
contents_outputs = []
y = Lambda(norm_vgg16)(input_contents)
for layer in vgg16.layers:
    y = layer(y)
    if layer.name in contents_layer_names:
        contents_outputs.append(y)
model_contents = Model(inputs=input_contents, outputs=contents_outputs)



def load_images(image_path_list, target_size):
    '''
    Return batch of array from image_path_list
    '''
    _load_img = lambda x: img_to_array(load_img(x, target_size=target_size))
    image_list = [np.expand_dims(_load_img(path), axis=0) for path in image_path_list]
    return np.concatenate(image_list, axis=0)

def train_generator(image_path_list, model, y_true_style, shuffle=True, epochs=10):
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
            X = load_images(image_path_ndarray[indices[start:end]])
            batch_size_act = X.shape[0]
            y_true_style_t = [np.repeat(feat, batch_size_act, axis=0) for feat in y_true_style]
            y_true_contents = model.predict(X)
            yield X, y_true_style_t + [y_true_contents]


image_path_list = ''
gen = train_generator(
    settings.DATA,
    settings.BATCH_SIZE,
    model_contents,
    y_true_style,
    epochs=10,
)



def feature_loss(y_true, y_pred):
    '''
    Loss function of contents features
    '''
    norm = K.prod(K.cast(K.shape(y_true)[1:], 'float32'))
    return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3)) / norm


def gram_matrix(x):
    '''
    Measure approximation of contents by gram matrix which calculates dot of feature maps
    '''
    x_sw = K.permute_dimensions(x, (0, 3, 2, 1)) # axis change
    s = K.shape(x_sw)
    new_shape = (s[0], s[1], s[2] * s[3])
    x_rs = K.reshape(x_sw, new_shape)
    x_rs_t = K.permute_dimensions(x_rs, (0, 2, 1)) # transpose
    dot = K.batch_dot(x_rs, x_rs_t) # calculates dot
    norm = K.prod(K.cast(s[1:], 'float32'))
    return dot / norm


def style_loss(y_true, y_pred):
    '''
    Loss function of style features
    '''
    return K.sum(K.square(gram_matrix(y_pred) - gram_matrix(y_true)), axis=(1, 2))


model.compile(optimizer=Adadelta(),
              loss=[style_loss, style_loss, style_loss, style_loss, feature_loss],
              loss_weights=[1.0, 1.0, 1.0, 1.0, 3.0])


img_test = load_img(path, target_size=input_shape[:2])
img_arr_test = np.expand_dims(img_to_array(img_test), axis=0)


steps_per_epochs = math.ceil(len(image_path_list) // settings.BATCH_SIZE)
iters_vobose = 1000
iters_save_img = 1000
iters_save_model = 1000

cur_epoch = 0
losses = []
path_tmp = 'epoch_{}_iters_{}_loss_{:.2f}_{}'
for i, (x_train, y_train) in enumerate(gen):
    if i % steps_per_epochs == 0:
        cur_epoch += 1
    loss = model.train_on_batch(x_train, y_train)
    losses.append(loss)
    if i % iters_vobose == 0:
        print('epoch:{}\titers:{}\tloss:{:.2f}'.format(cur_epoch, i, loss[0]))
    if i % iters_save_img == 0:
        pred = model_gen.predict(img_arr_test)
        img_pred = array_to_img(pred.squeeze())
        path_trs_img = path_tmp.format(cur_epoch, i, loss[0], '.jpg')
        img_pred.save(os.path.join(setting.DEBUG_IMG, path_trs_img))
        print('saved {}'.format(path_trs_img))
    if i % iters_save_model == 0:
        model.save(os.path.join(settings.MODEL, path_tmp.format(cur_epoch, i, loss[0], '.h5')))
        path_loss = os.path.join(settings.LOG, 'loss.pkl')
        with open(path_loss, 'wb') as f:
            pickle.dump(losses, f)