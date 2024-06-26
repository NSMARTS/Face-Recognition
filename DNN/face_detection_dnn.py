# import argparse
#  https://github.com/dldudcks1779/Face-Detection-dnn_caffemodel/blob/master/face_detection_image.py

import imutils
import numpy as np
import cv2
from imutils.video import VideoStream
import time

print('face detection DNN opencv')

# 모델의 레이어 구성 및 속성 정의
prototxt = './faceDetector_dnn/deploy.prototxt.txt'

# 얼굴 인식을 위해 ResNet 기본 네트워크를 사용하는 SSD(single shot detector)
weights = './faceDetector_dnn/res10_300x300_ssd_iter_140000.caffemodel'

# 네트워크를 메모리에 로드
net = cv2.dnn.readNet(prototxt, weights)

# 이미지 읽기
image = cv2.imread('./sample_human_img/sample_human2.jpg')

# resize
image = imutils.resize(image, width=500)

# 이미지 크기
(h,w) = image.shape[:2]

# print(h, w)

blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))

net.setInput(blob) # setInput() : blob 이미지를 네트워크의 입력으로 설정
detections = net.forward() # forward() : 네트워크 실행(얼굴 인식)

# 인식할 최소 확률
minimum_confidence = 0.8

# 얼굴 번호
number = 0

# 얼굴 인식을 위한 반복
for i in range(0, detections.shape[2]):
    # 얼굴 인식 확률 추출
    confidence = detections[0, 0, i, 2]
    
    # 얼굴 인식 확률이 최소 확률보다 큰 경우
    if confidence > minimum_confidence:
        # bounding box 위치 계산
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # bounding box 가 전체 좌표 내에 있는지 확인
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))        
        
        # cv2.putText(image, "Face[{}]".format(number + 1), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # 얼굴 번호 출력
        # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2) # bounding box 출력

        detection_face = image[startY:endY, startX:endX]

        number = number + 1 # 얼굴 번호 증가

        # 이미지 저장
        # cv2.imwrite(f'./sample_human_img/sample{i}.jpg', detection_face) # 파일로 저장, 포맷은 확장자에 따름

# 이미지 show
# cv2.imshow("Face Detection", image)

def main():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        frame = vs.read()
        if frame is None:
            print('NNNNNNNNNNNNNNNNONE')
            break
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # 'q' 키를 누르면 루프 종료
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()
cv2.waitKey(0)
