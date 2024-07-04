import numpy as np
import cv2
import os
from tqdm import tqdm
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def load_data(data_dir, input_shape):
#     subfolders = sorted(
#         [file.path for file in os.scandir(data_dir) if file.is_dir()])
#     print(subfolders)
#     images = []
#     labels = []
#     for idx, folder in tqdm(enumerate(subfolders)):
#         if os.path.isdir(folder):
#             for img_name in os.listdir(folder):
#                 img_path = os.path.join(folder, img_name)
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     img = cv2.resize(img, input_shape[:2])
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     images.append(img)
#                     labels.append(idx)
#     return np.array(images), np.array(labels)


def image_to_tfrecord(data_dir, output_path,val_output_path, input_shape):
    writer = tf.io.TFRecordWriter(output_path)
    writer_val = tf.io.TFRecordWriter(val_output_path)
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    for idx, folder in tqdm(enumerate(subfolders), total=len(subfolders)):
        person_id = int(os.path.basename(folder))
        # print('person_id : ', person_id)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, input_shape[:2])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_raw = img.tostring()

                    if person_id > 393:
                        example = tf.train.Example(features=tf.train.Features(feature={
                        'image': _bytes_feature(img_raw),
                        'label': _int64_feature(person_id)
                        }))
                        writer_val.write(example.SerializeToString())
                    else:
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image': _bytes_feature(img_raw),
                            'label': _int64_feature(person_id)
                        }))
                        writer.write(example.SerializeToString())
                else:
                    print(f"Failed to load image: {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    writer.close()
    writer_val.close()

input_shape = (112, 112, 3)
data_dir = '../kface/'  # 데이터셋 경로
output_path = 'kface.tfrecord'
val_output_path = 'kface_validation.tfrecord'

# 데이터 TFRecord로 변환
image_to_tfrecord(data_dir, output_path, val_output_path, input_shape)

