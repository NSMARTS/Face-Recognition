import fileinput
import sys
import face_detection
import base64
import cv2
from io import BytesIO
from PIL import Image
import io
import numpy as np

from rembg import remove    # 얼굴 배경 제거를 위한 모듈

parentInput = ''

# 부모가 보낸것을 받음
# base64 머시기
for line in fileinput.input():
    parentInput += line

# base64 -> numpy로 변환
# cv2를 통해서 np.array 형식으로 반환하는 것이고,
# 반환하기전 image 상태는 말그대로 img 형식의 데이터이다.
imgdata = base64.b64decode(parentInput)
dataBytesIO = io.BytesIO(imgdata)
image = Image.open(dataBytesIO)

# 배경제거 ( 얼굴을 탐지하고 배경을 제거하는거 보다는 그냥 배경을 제거하는게 더 낫다는 생각 )
source = remove(image)

# 이미지를 grayscale, np array 형태로 변환
source = cv2.cvtColor(np.array(source), cv2.COLOR_BGR2GRAY)

# 이미지에서 얼굴찾기
frame, add_img, face_coords = face_detection.detect_faces(source, True)

# # 이미지 리사이즈(학습된 모델에 따라 변경 (106 x 106 사이즈로 학습됨))
face_box = cv2.resize(add_img, (156, 156))
cv2.imwrite('detect_Face.png', face_box)    # 이미지 저장( 안해도 되긴함, 어떻게 나오는지 확인용 )

# face_box 2차원배열로 된 image임 즉, 1024*1024인 경우는 행이 1024개 열이 1024개인 2차원배열이다
rawBytes = BytesIO()

# 2차원 배열의 type은 uint8이여야 인코딩 가능
img_buffer = Image.fromarray(face_box.astype('uint8'))
img_buffer.save(rawBytes, 'PNG')

rawBytes.seek(0)
# base64_img = base64.b64encode(rawBytes.read())
base64_img = base64.b64encode(rawBytes.getvalue())

base64_img = "data:image/jpeg;base64," + (base64_img.decode('utf-8'))

# img_base64 = bytes("data:image/jpeg;base64,", encoding='utf-8') + base64_img
# base64_img
# print(base64_img)

sys.stdout.write(base64_img)
sys.stdout.flush()
