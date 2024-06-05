import numpy as np
import zipfile
import os
import shutil
import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob

 
def unzip():
    # Middle_Resolution 폴더에 있는 확장자가 zip인 경로 모두 가져오기
    zip_names = glob("Middle_Resolution/*.zip")

    # 가져온거에서 파일명만 추출하고 class_names에 저장
    class_names = []
    for z in zip_names:
        class_names.append(os.path.basename(z).split('.')[0])

    # Aihub 한국인 안면데이터 - 데이터 설명 참고
    lux = ["L1", "L3"]  # 밝기
    emotion = ["E01", "E02", "E03"] # 감정표현
    angle = ["C6", "C7", "C8", "C9"]    # 보이는 각도
    img_names = []
    txt_names = []

    for l in lux:
        for e in emotion:
            for c in angle:
                img_names.append(l + '/' + e + '/' + c + '.jpg')
                txt_names.append(l + '/' + e + '/' + c + '.txt')

    for z, c in tqdm(zip(zip_names, class_names)):
        if not os.path.exists("MR/" + c):
            os.makedirs("MR/" + c)

        for i, t in zip(img_names, txt_names):
            zipfile.ZipFile(z).extract("S001/" + i)
            zipfile.ZipFile(z).extract("S001/" + t)
        shutil.move("S001", "MR/" + c)

    # crop
    for j, c in enumerate(class_names):

        imgs = glob("MR/" + c + "/*/*/*/*.jpg")
        txts = glob("MR/" + c + "/*/*/*/*.txt")

        for i, (img, txt) in enumerate(zip(imgs, txts)):
            name = str(i)
            with open(txt, 'r') as f:
                bbox = f.read().split('\n')[7].split()
                bbox = list(map(int, bbox))
                (x, y, w, h) = bbox

                img = cv2.imread(img)
                img = img[y: y + h, x: x + w]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if j >= 390:
                    base_val = "kface_val/" + str(j - 390)
                    if not os.path.exists(base_val):
                        os.makedirs(base_val)
                    Image.fromarray(img).save(os.path.join(base_val, str(j-390) + '_' + name) + '.jpg')
                else:
                    base = "kface/" + str(j)

                    if not os.path.exists(base):
                        os.makedirs(base)
                    Image.fromarray(img).save(os.path.join(base, str(j) + '_' + name) + '.jpg')


def main():
    unzip()

if __name__ == "__main__":
    main()