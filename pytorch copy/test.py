import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model_define import ArcFaceResNet50
import numpy as np
from tqdm import tqdm
import os
import random
import csv
import matplotlib.pyplot as plt
import math

# 모델 로드 (분류기 부분 제외)
class FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # 'backbone'이라는 이름의 속성이 있다고 가정
        self.features = original_model.backbone
    def forward(self, x):
        return self.features(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ArcFaceResNet50(num_classes=399)
# model.load_state_dict(torch.load('./checkpoint/best_model_checkpoint.pth.tar')['model_state_dict'])
model.load_state_dict(torch.load('./first_checkpoint/best_model_checkpoint.pth.tar')['model_state_dict'])
feature_extractor = FeatureExtractor(model)
# print(model)

feature_extractor.eval()
feature_extractor.to(device)
# print(feature_extractor)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((124, 124)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_feature(img_path, model, transform):
    # print('img_path : ',img_path)
    
    img = Image.open(img_path).convert('RGB')

    # print('convert img : ',img)
    img = transform(img).unsqueeze(0).to(device)
    # print('img shape : ',img.shape)

    with torch.no_grad():
        feature = model(img)
    
    # shape 확인 및 조정
    feature = feature.squeeze()  # 불필요한 차원 제거
    # print('feature.shape squeeze',feature.shape)

    if feature.dim() == 1:
        feature = feature.unsqueeze(0)  # (2048,) -> (1, 2048)
    elif feature.dim() > 2:
        feature = feature.view(1, -1)  # 2차원 이상인 경우 (1, 2048)로 변경
    
    # print(f"Adjusted feature shape: {feature.shape}")
    
    # normalize 함수에 dim 매개변수 추가
    normalized_feature = F.normalize(feature, dim=1)
    # print('normalized_feature : ', normalized_feature.shape)
    return normalized_feature.cpu().numpy()

def compute_similarity(img1_path, img2_path, model, transform):
    feature1 = extract_feature(img1_path, model, transform)
    feature2 = extract_feature(img2_path, model, transform)
    
    # 벡터의 shape를 명시적으로 (1, -1)로 변경
    feature1 = feature1.reshape(1, -1)
    # feature2 = feature2.reshape(1, -1)
    # print('feature1 reshape : ', feature1)
    # print('feature2 reshape : ', feature2)
    similarity = np.dot(feature1, feature2.T)
    # similarity = np.dot(feature1, feature2)

    # print('similarity : ', similarity)

    return similarity[0][0]  # 스칼라 값으로 변환

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

def random_pair(arr, seedArr):

    # 짝을 지어 튜플로 만들기
    all_list = []
    paired_list = []

    for s in seedArr:
        cp_arr = arr.copy()
        random.seed(s)

        paired_list = []
        # 같은 얼굴 라벨 먼저 하고
        for i in tqdm(range(0, len(cp_arr), 2)):
            # print(i)
            # pair = (arr[i], arr[i + 1])

            pair1 = os.path.normpath(cp_arr[i]).split(os.sep)
            pair2 = os.path.normpath(cp_arr[i+1]).split(os.sep)
            # print('앞 반복문',pair1[-2], pair2[-2])
            label = 0 if pair1[-2] == pair2[-2] else 1
            similarity = compute_similarity(cp_arr[i], cp_arr[i + 1], feature_extractor, transform)
            paired_list.append((cp_arr[i], cp_arr[i + 1], label, similarity))

        random.shuffle(cp_arr)
        
        # 배열 길이가 홀수일 경우 마지막 요소 제거
        if len(cp_arr) % 2 != 0:
            cp_arr = cp_arr[:-1]
        
        # 랜덤으로 섞은거 다시 라벨링
        for i in tqdm(range(0, len(cp_arr), 2)):

            pair1 = os.path.normpath(cp_arr[i]).split(os.sep)
            pair2 = os.path.normpath(cp_arr[i+1]).split(os.sep)
            # print('뒤 반복문',pair1[-2], pair2[-2])
            label = 0 if pair1[-2] == pair2[-2] else 1
            similarity = compute_similarity(cp_arr[i], cp_arr[i + 1], feature_extractor, transform)
            paired_list.append((cp_arr[i], cp_arr[i + 1], label, similarity))

        all_list.append(paired_list)

    return all_list

def random_pair_one(arr,seed):
    # 짝을 지어 튜플로 만들기
    paired_list = []

    # 같은 얼굴 라벨 먼저 하고
    for i in tqdm(range(0, len(arr), 2)):
        # print(i)
        # pair = (arr[i], arr[i + 1])

        pair1 = os.path.normpath(arr[i]).split(os.sep)
        pair2 = os.path.normpath(arr[i+1]).split(os.sep)
        # print('앞 반복문',pair1[-2], pair2[-2])
        label = 0 if pair1[-2] == pair2[-2] else 1
        similarity = compute_similarity(arr[i], arr[i + 1], feature_extractor, transform)
        paired_list.append((arr[i], arr[i + 1], label, similarity))

    random.seed(seed)
    random.shuffle(arr)
    
    # 배열 길이가 홀수일 경우 마지막 요소 제거
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    
    # 랜덤으로 섞은거 다시 라벨링
    for i in tqdm(range(0, len(arr), 2)):

        pair1 = os.path.normpath(arr[i]).split(os.sep)
        pair2 = os.path.normpath(arr[i+1]).split(os.sep)
        # print('뒤 반복문',pair1[-2], pair2[-2])
        label = 0 if pair1[-2] == pair2[-2] else 1
        similarity = compute_similarity(arr[i], arr[i + 1], feature_extractor, transform)
        paired_list.append((arr[i], arr[i + 1], label, similarity))


    return paired_list

# CSV 파일로 저장하는 함수
def save_to_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# imshow로 보이게
def show_images(pair, cols=4, batches=6):

    random.shuffle(pair)

    batch_size = math.ceil(len(pair) / batches)  # 각 배치당 표시할 페어 수
    for batch_idx in range(batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pair))
        batch = pair[start_idx:end_idx]
        
        # 각 배치에 대해 이미지 표시
        rows = (len(batch) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols * 2, figsize=(15, rows * 5))
        axes = axes.flatten()
        
        for idx, (img1_path, img2_path, label, score) in enumerate(batch):
            # 첫 번째 이미지
            img1 = Image.open(img1_path)
            axes[idx * 2].imshow(img1)
            axes[idx * 2].axis('off')
            axes[idx * 2].set_title(f"Img1\nLabel: {label}, Score: {score:.2f}")
            
            # 두 번째 이미지
            img2 = Image.open(img2_path)
            axes[idx * 2 + 1].imshow(img2)
            axes[idx * 2 + 1].axis('off')
            axes[idx * 2 + 1].set_title("Img2")
        
        # 남은 빈 공간 숨김 처리
        for ax in axes[len(batch) * 2:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def score_cal(pair):
    all_yes = []
    all_no =[]

    for i, p in enumerate(pair):
        yes = []
        no = []

        for idx,(img1, img2, label, score) in enumerate(p):

            if (score > threshold and label == 0) or (score < threshold and label == 1):
                yes.append(1)
                all_yes.append(1)
            elif (score < threshold and label == 0) or (score > threshold and label == 1):
                no.append(1)
                all_no.append(1)

        print(f'{i+1}번째 시드 임계값 맞은 수 : ', len(yes))
        print(f'{i+1}번째 시드 임계값 틀린 수 : ', len(no))
        print(f'{i+1}번째 시드 임계값 맞은 수 / 전체 데이터 = {len(yes)/(len(yes)+len(no)) *100} %')
        print('--------------------------------------------------------------------------')


    print('전체 임계값 맞은 수 : ', len(all_yes))
    print('전체 임계값 틀린 수 : ', len(all_no))
    print(f'전체 임계값 맞은 수 / 전체 데이터 = {len(all_yes)/(len(all_yes)+len(all_no))*100} %')
    print('--------------------------------------------------------------------------')
    return

def score_cal_one(pair):
    yes = []
    no = []
    for idx,(img1, img2, label, score) in enumerate(pair):
            
            if (score > threshold and label == 0) or (score < threshold and label == 1):
                yes.append(1)
            elif (score < threshold and label == 0) or (score > threshold and label == 1):
                no.append(1)

    print(f'시드 임계값 맞은 수 : ', len(yes))
    print(f'시드 임계값 틀린 수 : ', len(no))
    print(f'시드 임계값 맞은 수 / 전체 데이터 = {len(yes)/(len(yes)+len(no)) *100} %')
    print('--------------------------------------------------------------------------')
    return


# 유사도 계산 예시 #############################################
# img1_path = '../kface_all/0/0_20_20.jpg'
# img2_path = '../kface_all/9/9_0_0.jpg'

# result = compute_similarity(img1_path, img2_path, feature_extractor, transform)
# print('result : ', result)
###############################################################

threshold = 0.9
dir_path = './kface_testSet'
seedArr = [34,2,6,4,5]
seed = 4

print('모델 및 이미지 로드')
img = img_load(dir_path)

print('이미지 페어 생성, 얼굴 판독')

# seedArr 사용
pair = random_pair(img, seedArr)
score_cal(pair)

# seed 하나만 사용
# pair = random_pair_one(img, seed)
# score_cal_one(pair)


# save_to_csv('./test/all_img.csv',pair)

# # label_0 = [tup for tup in pair if tup[2] == 0]
# # label_0_similarity = [tup[3] for tup in label_0]
# # average = sum(label_0_similarity) / len(label_0_similarity)
# # print('라벨이 0 인 페어 수 : ',len(label_0))
# # print('라벨이 0 인 페어 유사도 평균 : ',average)


# # label_1 = [tup for tup in pair if tup[2] == 1]
# # label_1_similarity = [tup[3] for tup in label_1]
# # average = sum(label_1_similarity) / len(label_1_similarity)
# # print('라벨이 1 인 페어 수 : ',len(label_1))
# # print('라벨이 1 인거 페어 유사도 평균 : ',average)

# # # show_images(pair)

# # save_to_csv('./test/label_0.csv',label_0)
# # save_to_csv('./test/label_1.csv',label_1)
