import math
import random

import numpy as np
from keras import backend as K
from keras.models import Model, Sequential
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers import Lambda, BatchNormalization, Conv2D, BatchNormalization, Add, Activation, Input, Conv2DTranspose
from keras.applications import VGG16
from keras.optimizers import Adadelta

import settings


class CommonModel:
    
    style_layer_names = frozenset(['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3'])
    contents_layer_names = frozenset(['block3_conv3'])

    def __init__(self):
        self.vgg16 = self._initialize_vgg16()

    def _initialize_vgg16(self):
        vgg16 = VGG16()
        for layer in vgg16.layers:
            layer.trainable = False
        return vgg16

    def norm_vgg16(self, x):
        '''
        RGB -> BGR transformation and approximate centerlization
        '''
        return (x[:, :, :, ::-1] - 120) / 255.


class StyleTransfer(CommonModel):
    '''
    StyleTransfer CNN Model
    '''
    def residual_block(self, input_tensor):
        # Creating Residual Block
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation()(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        return Add()([x, input_tensor])

    def build_encoder_decoder(self):
        # Encoder
        input_tensor = Input(shape=INPUT_SHAPE, name='input')
        x = Lambda(lambda x: x / 255.)(input_tensor) # normalization [0, 1]
        x = Conv2D(filters=32, kernel_size=(9, 9), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same')(x)
        for _ in range(5): # Add five residual blocks
            x = self.residual_block(x)
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


    def feature_loss(self, y_true, y_pred):
        '''
        Loss function of contents features
        '''
        norm = K.prod(K.cast(K.shape(y_true)[1:], 'float32'))
        return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3)) / norm

    def gram_matrix(self, x):
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

    def style_loss(self, y_true, y_pred):
        '''
        Loss function of style features
        '''
        return K.sum(K.square(self.gram_matrix(y_pred) - self.gram_matrix(y_true)), axis=(1, 2))

    def create_inputs_and_outputs(self):
        model_gen = self.build_encoder_decoder()
        style_outputs_gen = []
        contents_outputs_gen = []
        input_gen = model_gen.output # inputs will come from stlyle-transform model
        z = Lambda(norm_vgg16)(input_gen)
        for layer in vgg16.layers: # Recontructing network by piling up 
            z = layer(z)
            if layer.name in cls.style_layer_names:
                style_outputs_gen.append(z)
            if layer.name in cls.contents_layer_names:
                contents_outputs_gen.append(z)
        inputs = model_gen.input
        outputs = style_outputs_gen + contents_outputs_gen
        return inputs, outputs

    def build_model(self):
        inputs, outputs = self.create_inputs_and_outputs()
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adadelta(),
            loss=[self.style_loss,
                  self.style_loss,
                  self.style_loss,
                  self.style_loss,
                  self.feature_loss],
            loss_weights=[1.0, 1.0, 1.0, 1.0, 3.0])
        return model


class Contents(CommonModel):

    def build_model(self):
        input_contents = Input(shape=INPUT_SHAPE, name='input_contents')
        contents_outputs = []
        y = Lambda(self.norm_vgg16)(input_contents)
        for layer in self.vgg16.layers:
            y = layer(y)
            if layer.name in cls.contents_layer_names:
                contents_outputs.append(y)
        model_contents = Model(inputs=input_contents, outputs=contents_outputs)
        return model_contents


class Style(CommonModel):

    def __init__(self):
        super().__init__()
        self.model = self.build_model()

    def build_model(self):
        img_arr_style = np.expand_dims(array_to_img(load_img(settings.STYLE_IMAGE, target_size=INPUT_SHAPE[:2])), axis=0)
        style_outputs = []
        input_style = Input(shape=INPUT_SHAPE, name='input_style')
        x = Lambda(self.modelnorm_vgg16)(input_style)
        for layer in self.vgg16.layers:
            x = layer(x)
            if layer.name in cls.style_layer_names:
                style_outputs.append(x)
        model_style = Model(inputs=input_style, outputs=style_outputs)
        return model_style

    def predict(self):
        y_true_style = self.model.predict(img_arr_style)
        return y_true_style