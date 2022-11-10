# from . import utils_test
import utils_test
import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)

# os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')


## 모델이 경로에 있는지 확인
files = os.listdir()
print('files : ', files)
if 'siamese_nn.h5' not in files:
    print("Error: trained Neural Network not found!")
    print("Please check siamese_nn.py path")
    sys.exit() 


## 테스트 셋을 만들 얼굴데이터가 있는 경로 설정
faces_dir = './kface/'

# 데이터 가져오기 및 데이터 라벨링
X_test, Y_test = utils_test.get_data_test_set(faces_dir)

# 가져온 데이터를 긍정페어, 부정페어로 만들기
num_classes = len(np.unique(Y_test))
test_pairs, test_labels = utils_test.create_pairs(X_test, Y_test, num_classes=num_classes)

# 페어 확인 
utils_test.visualize(test_pairs[:-1], test_labels[:-1], to_show=12, num_col=4, main_title='Test pairs sample')

## 모델 가져오기
from keras.models import load_model
model = load_model('./siamese_nn_90.h5', custom_objects={'contrastive_loss': utils_test.loss(margin=1), 'euclidean_distance2': utils_test.euclidean_distance})

if model:
    print('Model Load Success')
else:
    print('Model not load')



# 모델 정확도 계산 및 출력
results = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
print()
print("시험 데이터의 손실율, 시험 데이터의 정답률:", results)

# '''
# 정확도 산출 수식
# accuracy(정확도) = (TP + TN) / (TP + FP + TN + FN)
# TP : 동일 인을페어를 동일인물이라고 판별한 수
# TN : 다른 인물페어를 다른 인물이라고 판별한 수
# FP : 다른 인물페어를 동일인물이라고 판별한 수
# FN : 동일 인물페어를 다른 인물이라고 판별한 수
# '''


# 모델 예측 시각화
predictions = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
utils_test.visualize(test_pairs, test_labels, to_show=10, predictions=predictions, test=True, main_title = "predictions")