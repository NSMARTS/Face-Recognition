import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, DepthwiseConv2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
from tqdm import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Mixed precision policy 설정
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def arcface_loss(y_true, y_pred, scale=64.0, margin=0.5):
    cos_m = tf.math.cos(margin)
    sin_m = tf.math.sin(margin)
    th = tf.math.cos(tf.constant(3.141592653589793) - margin)
    mm = tf.math.sin(tf.constant(3.141592653589793) - margin) * margin

    y_true = tf.cast(y_true, tf.float32)
    cosine = tf.clip_by_value(y_pred, -1.0 + 1e-6, 1.0 - 1e-6)
    sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
    phi = cosine * cos_m - sine * sin_m

    phi = tf.where(cosine > th, phi, cosine - mm)
    output = (y_true * phi) + ((1.0 - y_true) * cosine)
    output *= scale
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, output, from_logits=True)


# def create_arcface_model(input_shape, num_classes):
#     inputs = Input(shape=input_shape)
#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = BatchNormalization()(x)
#     x = PReLU(shared_axes=[1, 2])(x)

#     x = DepthwiseConv2D((3, 3), padding='same')(x)
#     x = BatchNormalization()(x)
#     x = PReLU(shared_axes=[1, 2])(x)
    
#     x = Flatten()(x)
#     x = Dense(512)(x)
#     x = BatchNormalization()(x)

#     outputs = Dense(num_classes)(x)
#     model = Model(inputs, outputs)
#     return model

def create_arcface_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)  # 필터 수 줄임
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)  # 필터 수 줄임
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(num_classes, dtype='float32')(x)  # 출력은 float32로 설정
    model = tf.keras.models.Model(inputs, outputs)
    return model


# def create_arcface_model(input_shape, num_classes):
#     inputs = tf.keras.layers.Input(shape=input_shape)
#     x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

#     x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(512)(x)
#     x = tf.keras.layers.BatchNormalization()(x)

#     outputs = tf.keras.layers.Dense(num_classes, dtype='float32')(x)  # 출력은 float32로 설정
#     model = tf.keras.models.Model(inputs, outputs)
#     return model

def parse_tfrecord_fn(example, input_shape):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, input_shape)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(example['label'], tf.int64)
    return image, label

def load_dataset(tfrecord_path, batch_size, input_shape, is_training=True):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parse_fn = lambda example: parse_tfrecord_fn(example, input_shape)
    
    dataset = raw_dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(10000).repeat()
        dataset = dataset.map(lambda image, label: (tf.image.random_flip_left_right(image), label),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def count_samples(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    return sum(1 for _ in raw_dataset)

input_shape = (112, 112, 3)
num_classes = 394  # 클래스 수

batch_size = 8
epochs = 50
learning_rate = 0.001

tfrecord_path = 'kface.tfrecord'
val_tfrecord_path = 'kface_validation.tfrecord'

# 샘플 수 계산
num_train_samples = count_samples(tfrecord_path)
num_val_samples = count_samples(val_tfrecord_path)

# 스텝 수 계산
steps_per_epoch = num_train_samples // batch_size
validation_steps = num_val_samples // batch_size


# TFRecord 데이터 로드
train_dataset = load_dataset(tfrecord_path, batch_size, input_shape, is_training=True)
val_dataset = load_dataset(val_tfrecord_path, batch_size, input_shape, is_training=False)

model = create_arcface_model(input_shape, num_classes)
opt = tf.keras.optimizers.Adam(learning_rate)
opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')  # loss_scale 설정

model.compile(optimizer=opt,
              loss=arcface_loss,
              metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
#               loss=arcface_loss,
#               metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_filepath = './checkpoints/best_checkpoint'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset,
                    callbacks=[checkpoint_callback],
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps)


# 가장 성능이 좋은 체크포인트 불러오기
model.load_weights(checkpoint_filepath)

# 평가 데이터 로드 및 평가 (예시 경로)
# test_tfrecord_path = 'kface.tfrecord'
test_dataset = load_dataset(val_tfrecord_path, batch_size, input_shape, is_training=False)

# 모델 평가
loss, accuracy = model.evaluate(test_dataset)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")