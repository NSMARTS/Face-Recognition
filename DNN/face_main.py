# import argparse
#  https://github.com/dldudcks1779/Face-Detection-dnn_caffemodel/blob/master/face_detection_image.py

import imutils
import numpy as np
import cv2
from imutils.video import VideoStream
import time
import utils 
import tensorflow as tf
import os

print('face detection DNN opencv')

        # 이미지 저장
        # cv2.imwrite(f'./sample_human_img/sample{i}.jpg', detection_face) # 파일로 저장, 포맷은 확장자에 따름

# 이미지 show
# cv2.imshow("Face Detection", image)

def get_all_checkpoints(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.index')]


def main(arcface):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    recog_thresh = 1.36


    save_face = cv2.imread('./data/kface/0/0_0_24.jpg')
    # ./data/kface/0/0_0_24.jpg
    # ./sample_human_img/baek.jpg

    while True:
        frame = vs.read()
        if frame is None:
            print('frame is None')
            break
        
        # print(frame.shape)
        # 프레임 크기 조정
        frame = imutils.resize(frame, width=400)

        # 이미지에서 블롭 생성
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # 블롭을 네트워크의 입력으로 설정
        net.setInput(blob)
        
        # 객체 검출 수행
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            # 얼굴 인식 확률 추출
            
            confidence = detections[0, 0, i, 2]
            
            # 얼굴 인식 확률이 최소 확률보다 큰 경우
            if confidence > minimum_confidence:
                # bounding box 위치 계산
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                
                detection_face = frame[startY:endY, startX:endX]
                detection_face = cv2.resize(detection_face, (224, 224))
                cv2.imwrite(f'./sample_human_img/baek.jpg', detection_face) # 파일로 저장, 포맷은 확장자에 따름

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2) # bounding box 출력
                
                # 신뢰도 표시
                text = f"{confidence * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                id_embeds = utils.get_embeddings(arcface, detection_face)
                cam_embeds = utils.get_embeddings(arcface, save_face)
                embeds_distance = np.subtract(id_embeds, cam_embeds)
                embeds_distance = np.sum(np.square(embeds_distance), axis=1)
                print(f"[INFO] Embeddings Distance: {embeds_distance[0]:.4f} | Threshold: {recog_thresh}")
                

                if embeds_distance < recog_thresh:
                    cv2.putText(frame, "[INFO] (1: Pass)", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "[INFO] (0: Fail)", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                
                
                break
        cv2.imshow("Frame", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    # 모델의 레이어 구성 및 속성 정의
    prototxt = './faceDetector_dnn/deploy.prototxt.txt'

    # 얼굴 인식을 위해 ResNet 기본 네트워크를 사용하는 SSD(single shot detector)
    weights = './faceDetector_dnn/res10_300x300_ssd_iter_140000.caffemodel'

    # 네트워크를 메모리에 로드
    net = cv2.dnn.readNet(prototxt, weights)

    # 인식할 최소 확률
    minimum_confidence = 0.8

    # ArcFace model load


    # 디렉토리 또는 파일이 존재하는지 확인
    path = './checkpoints'
    if os.path.exists(path):
        print(f"@@@@@@@@@@ The path '{path}' exists.")
    else:
        print(f"The path '{path}' does not exist.")

    ckpt_dir = './checkpoints/arc_res50_kface_finetune-lr0.001-bs8-trainable2/'

    arcface = utils.ArcFaceModel(size=224, backbone_type='ResNet50', training=False)
    # ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

    all_checkpoints = get_all_checkpoints(ckpt_dir)
    third_checkpoint = all_checkpoints[2].replace('.index', '')

    # if ckpt_path is None:
    #     print(f"@@@@@@@@@@@@@@  No checkpoint found in directory: {ckpt_dir}")
    # else:
    #     arcface.load_weights(third_checkpoint)
    #     print(f"@@@@@@@@@@@@@@@@@  Checkpoint loaded from: {ckpt_path}")

    
    arcface.load_weights(third_checkpoint)



    main(arcface)
cv2.waitKey(0)
