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
model.load_state_dict(torch.load('./checkpoint/best_model_checkpoint.pth.tar')['model_state_dict'])
feature_extractor = FeatureExtractor(model)
# print(model)

feature_extractor.eval()
feature_extractor.to(device)
# print(feature_extractor)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_feature(img_path, model, transform):
    print('img_path : ',img_path)
    img = Image.open(img_path).convert('RGB')
    print('convert img : ',img)
    img = transform(img).unsqueeze(0).to(device)
    print('img shape : ',img.shape)

    with torch.no_grad():
        feature = model(img)
    
    # shape 확인 및 조정
    # print(f"Raw feature shape: {feature.shape}")
    print('feature.shape : ',feature.shape)
    feature = feature.squeeze()  # 불필요한 차원 제거
    print('feature.shape squeeze',feature.shape)

    if feature.dim() == 1:
        feature = feature.unsqueeze(0)  # (2048,) -> (1, 2048)
    elif feature.dim() > 2:
        feature = feature.view(1, -1)  # 2차원 이상인 경우 (1, 2048)로 변경
    
    # print(f"Adjusted feature shape: {feature.shape}")
    
    # normalize 함수에 dim 매개변수 추가
    normalized_feature = F.normalize(feature, dim=1)
    print('normalized_feature : ', normalized_feature.shape)
    return normalized_feature.cpu().numpy()

def compute_similarity(img1_path, img2_path, model, transform):
    feature1 = extract_feature(img1_path, model, transform)
    feature2 = extract_feature(img2_path, model, transform)
    
    # 벡터의 shape를 명시적으로 (1, -1)로 변경
    feature1 = feature1.reshape(1, -1)
    feature2 = feature2.reshape(1, -1)
    print('feature1 reshape : ', feature1)
    print('feature2 reshape : ', feature2)
    # similarity = np.dot(feature1, feature2.T)
    similarity = np.dot(feature1, feature2)

    print('similarity : ', similarity)

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

def random_pair(arr):
    # 짝을 지어 튜플로 만들기
    paired_list = []

    # # 같은 얼굴 라벨 먼저 하고
    # for i in tqdm(range(0, len(arr), 2)):
    #     # print(i)
    #     pair = (arr[i], arr[i + 1])

    #     pair1 = os.path.normpath(arr[i]).split(os.sep)
    #     pair2 = os.path.normpath(arr[i+1]).split(os.sep)
    #     print('앞 반복문',pair1[-2], pair2[-2])
    #     label = 0 if pair1[-2] == pair2[-2] else 1
    #     similarity = compute_similarity(arr[i], arr[i + 1], feature_extractor, transform)
    #     paired_list.append((arr[i], arr[i + 1], label, similarity))

        
    # 섞은 다음
    random.shuffle(arr)
    
    # 배열 길이가 홀수일 경우 마지막 요소 제거
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    
    # 랜덤으로 섞은거 다시 라벨링 -> 이렇게 하는 이유가 다 랜덤 돌리면 같은 라벨링이 너무 안나옴..
    for i in tqdm(range(0, len(arr), 2)):

        pair1 = os.path.normpath(arr[i]).split(os.sep)
        pair2 = os.path.normpath(arr[i+1]).split(os.sep)
        print('뒤 반복문',pair1[-2], pair2[-2])
        label = 0 if pair1[-2] == pair2[-2] else 1
        similarity = compute_similarity(arr[i], arr[i + 1], feature_extractor, transform)
        paired_list.append((arr[i], arr[i + 1], label, similarity))


    return paired_list

# CSV 파일로 저장하는 함수
def save_to_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# 유사도 계산 예시
img1_path = '../kface_all/0/0_20_20.jpg'
img2_path = '../kface_all/9/9_0_0.jpg'

result = compute_similarity(img1_path, img2_path, feature_extractor, transform)
print('result : ', result)

# dir_path = '../kface'
# img = img_load(dir_path)
# pair = random_pair(img)

# print('전체 페어 길이 : ', len(pair))
# save_to_csv('./test/all_img.csv',pair)

# threshold = 0.89

# yes = []
# no = []
# for idx,(img1, img2, label, score) in enumerate(pair):
#     if (score > threshold and label == 0) or (score < threshold and label == 1):
#         yes.append(1)
#     elif (score < threshold and label == 0) or (score > threshold and label == 1):
#         no.append(1)
# print('임계값 맞은 수 : ', len(yes))
# print('임계값 틀린 수 : ', len(no))


# label_0 = [tup for tup in pair if tup[2] == 0]
# save_to_csv('./test/label_0.csv',label_0)
# print('페어에서 라벨이 0 인거 : ',len(label_0))
# label_0_similarity = [tup[3] for tup in label_0]
# average = sum(label_0_similarity) / len(label_0_similarity)
# print('페어에서 라벨이 0 인거 유사도 평균 : ',average)


# label_1 = [tup for tup in pair if tup[2] == 1]
# save_to_csv('./test/label_1.csv',label_1)
# print('페어에서 라벨이 1 인거 : ',len(label_1))
# label_1_similarity = [tup[3] for tup in label_1]
# average = sum(label_1_similarity) / len(label_1_similarity)
# print('페어에서 라벨이 1 인거 유사도 평균 : ',average)
