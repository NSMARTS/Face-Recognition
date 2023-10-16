# Face-Recognition

이 프로젝트는 고급 얼굴 인식 및 탐지 기능을 제공하는 AI 기반 시스템입니다. 다양한 알고리즘과 기술을 활용하여 얼굴 인식의 정확성을 높이고 사용자 경험을 향상시킵니다.

model dir path change

## 주요 기능
- 얼굴 인식: 사용자가 업로드한 사진을 기반으로 얼굴을 인식합니다.
- 얼굴 탐지: 실시간으로 얼굴을 탐지하고 분석합니다.
- 데이터 처리: 대규모 데이터셋을 효율적으로 처리하기 위한 분산 데이터 관리 시스템을 포함합니다.


- data.h5 
    - AI 모델링을 할때 데이터를 분산으로 처리해서 스펙이 좀 부족해도 커버할 수 있는 방법 [blog to link](https://nuxlear.tistory.com/4)
    - [ When performing AI modeling, data is processed in a distributed manner to cover cases where the specifications may be insufficient. [blog to link](https://nuxlear.tistory.com/4) ]

## Files

- face_recognition_system.py 
    - 얼굴 인식 테스트 코드 ( 사진하나 올려서 비교 )
    - [ Test code for facial recognition (upload a photo for comparison). ]

- siamese_nn2_original.py 
    - 이걸 수정 중에 있다. -> 어느정도 수정 완료 브렌치 병합
    - [ Currently being modified. -> Modifications are mostly complete, ready for branch merge. ]

- face_detection.py 
    - 얼굴 탐지할 때 사용
    - [ Used for face detection. ]

## TEST
- utils_test 
    - 데이터 페어 등 모델에 필요한 대부분의 함수가 여기 있음 -> 수정 후 브렌치 병합
    - [ Contains most of the functions needed for the model, such as data pairs. After modification, ready for branch merge. ]


## Directory
- face_recognition_system.py: 얼굴 인식 테스트 코드입니다. 사진을 업로드하여 얼굴을 비교합니다.
- siamese_nn2_original.py: 현재 수정 중인 파일입니다. 브랜치 병합을 위해 수정이 거의 완료되었습니다.
- face_detection.py: 얼굴 탐지를 위해 사용되는 파일입니다.
- utils_test: 모델에 필요한 대부분의 함수가 포함된 테스트 유틸리티입니다. 수정 후 브랜치 병합이 준비됩니다.

## Install
이 섹션에서는 프로젝트를 설치하고 실행하기 위한 단계별 지침을 제공합니다.

```
git clone [프로젝트 URL]
cd [프로젝트 디렉토리]
pip install -r requirements.txt
```

## Start
```
python face_recognition_system.py
```

