

'''
함수들 모아놓은 파일
'''
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50
)

from recognition.layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists
)


def l2_norm(x, axis=1):
    '''
    L2 정규화
    '''
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm


def get_embeddings(recognition_model, img):
    '''
    이미지 리사이즈, 임베딩(L2 정규화를 통한 임베딩)
    '''
    img = cv2.resize(img, (224, 224))
    img = img / 255.
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    embeds = l2_norm(recognition_model(img))
    return embeds

def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

  
    return Model(inputs, embds, name=name)

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer

def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        else:
            raise TypeError('backbone_type error!')
    return backbone