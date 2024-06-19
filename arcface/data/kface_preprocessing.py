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

def make_pair_numpy():
    ref_imgs = []
    query_imgs = []
    is_same = []

    with open("references.csv", 'w', encoding='utf-8') as f:
        f.write("references, queries, is_same\n")
        for _ in range(400):
            classes = os.listdir("kface_val")

            ref_number = np.random.randint(24)
            ref_class = np.random.choice(classes)
            ref_name = os.path.join("kface_val", ref_class, f"{ref_class}_{str(ref_number)}.jpg")
            f.write(f"{os.path.basename(ref_name)}, ")
            ref_imgs.append(np.array(Image.open(ref_name).resize((112, 112)), dtype=np.uint8))

            is_pair = np.random.randint(2)
            query_number = np.random.randint(24)
            if is_pair:
                is_same.append(1)
                query_name = os.path.join("kface_val", ref_class, f"{ref_class}_{str(query_number)}.jpg")
            else:
                is_same.append(0)
                classes.remove(ref_class)
                query_class = np.random.choice(classes)
                query_name = os.path.join("kface_val", query_class, f"{query_class}_{str(query_number)}.jpg")

            f.write(f"{os.path.basename(query_name)}, {is_pair}\n")
            query_imgs.append(np.array(Image.open(query_name).resize((112, 112)), dtype=np.uint8))

    ref_imgs = np.array(ref_imgs, dtype=np.uint8)
    query_imgs = np.array(query_imgs, dtype=np.uint8)
    is_same = np.array(is_same, dtype=np.uint8)

    if not os.path.exists('kface_val_npy'):
        os.makedirs('kface_val_npy')

    np.save('kface_val_npy/references.npy', ref_imgs)
    np.save('kface_val_npy/queries.npy', query_imgs)
    np.save('kface_val_npy/is_same.npy', is_same)


# def make_pair_numpy_benchmark():
#     ref_imgs = []
#     query_imgs = []
#     is_same = []
#     ref_img_paths = glob("ids/*.jpg")
#     query_img_paths = glob("faces/*.jpg")

#     with open("references_benchmark.csv", 'w', encoding='utf-8') as f:
#         f.write("references, queries, is_same\n")
#         for i, ref_name in enumerate(ref_img_paths):
#             print(f"Iter {i+1:2d}: {os.path.basename(ref_name)}")
#             # random_ref = np.random.randint(40)
#             # random_query = np.random.randint(40)
#             # ref_name = ref_img_paths[random_ref]
#             # query_name = query_img_paths[random_query]
#             for query_name in tqdm(query_img_paths):
#                 f.write(f"{os.path.basename(ref_name)}, {os.path.basename(query_name)}, ")
#                 if os.path.basename(ref_name) == os.path.basename(query_name):
#                     is_pair = 1
#                 else:
#                     is_pair = 0
#                 f.write(f"{is_pair}\n")

#                 ref_img = cv2.imread(ref_name)
#                 ref_img = crop_face_from_id(ref_img, weight_path="../weights")
#                 ref_img = cv2.resize(ref_img, (112, 112))
#                 query_img = cv2.imread(query_name)
#                 query_img = crop_face_from_id(query_img, weight_path="../weights")
#                 query_img = cv2.resize(query_img, (112, 112))

#                 ref_imgs.append(ref_img)
#                 query_imgs.append(query_img)
#                 is_same.append(is_pair)

#         ref_imgs = np.array(ref_imgs, dtype=np.uint8)
#         query_imgs = np.array(query_imgs, dtype=np.uint8)
#         is_same = np.array(is_same, dtype=np.uint8)

#         if not os.path.exists('benchmark_npy'):
#             os.makedirs('benchmark_npy')

#         np.save('benchmark_npy/references.npy', ref_imgs)
#         np.save('benchmark_npy/queries.npy', query_imgs)
#         np.save('benchmark_npy/is_same.npy', is_same)

def main():
    # unzip()

    # shutil.rmtree("MR")
    # shutil.rmtree("Middle_Resolution")

    make_pair_numpy()
    # make_pair_numpy_benchmark()

if __name__ == "__main__":
    main()