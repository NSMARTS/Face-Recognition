import numpy as np
import os
import shutil
from tqdm import tqdm
import os
import random



def img_load(dir_path):
    subfolders = sorted(
            [file.path for file in os.scandir(dir_path) if file.is_dir()])

    # all_img 에 우선 다 넣음
    all_img = []
    for idx, folder in tqdm(enumerate(subfolders)):

            # all_img 에 넣으면서 한 반절은 같은 쌍으로 진행??
            for file in sorted(os.listdir(folder)):
                    all_img.append(folder+"/"+file)
    return all_img

def random_pair(arr):
    # 짝을 지어 튜플로 만들기
    paired_list = []

    # 같은 얼굴 라벨 먼저 하고
    for i in range(0, len(arr), 2):
        print(i)
        pair = (arr[i], arr[i + 1])

        pair1 = os.path.normpath(arr[i]).split(os.sep)
        pair2 = os.path.normpath(arr[i+1]).split(os.sep)

        label = 0 if pair1[-2] == pair2[-2] else 1
        paired_list.append((arr[i], arr[i + 1], label))

        if i == 13000:
             arr = arr[13000:]
             break
            
    # 섞은 다음
    random.shuffle(arr)
    
    # 배열 길이가 홀수일 경우 마지막 요소 제거
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    
    # 랜덤으로 섞은거 다시 라벨링 -> 이렇게 하는 이유가 다 랜덤 돌리면 같은 라벨링이 너무 안나옴..
    for i in range(0, len(arr), 2):
        pair = (arr[i], arr[i + 1])

        pair1 = os.path.normpath(arr[i]).split(os.sep)
        pair2 = os.path.normpath(arr[i+1]).split(os.sep)

        label = 0 if pair1[-2] == pair2[-2] else 1
        paired_list.append((arr[i], arr[i + 1], label))

        # if i == 10:
        #      break

    return paired_list
       
if __name__ == '__main__':
    dir_path = '../kface'
    img = img_load(dir_path)
    print(len(img))

    pair = random_pair(img)
    print('전체 페어 길이 : ', len(pair))
    filtered_data = [tup for tup in pair if tup[2] == 0]
    print('페어에서 라벨이 0 인거 : ',len(filtered_data))
    filtered_data = [tup for tup in pair if tup[2] == 1]
    print('페어에서 라벨이 1 인거 : ',len(filtered_data))
