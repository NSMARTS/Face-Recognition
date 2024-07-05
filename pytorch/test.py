import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model_define import ArcFaceResNet50
import numpy as np

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
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(img)
    
    # shape 확인 및 조정
    print(f"Raw feature shape: {feature.shape}")
    feature = feature.squeeze()  # 불필요한 차원 제거
    
    if feature.dim() == 1:
        feature = feature.unsqueeze(0)  # (2048,) -> (1, 2048)
    elif feature.dim() > 2:
        feature = feature.view(1, -1)  # 2차원 이상인 경우 (1, 2048)로 변경
    
    print(f"Adjusted feature shape: {feature.shape}")
    
    # normalize 함수에 dim 매개변수 추가
    normalized_feature = F.normalize(feature, dim=1)
    
    return normalized_feature.cpu().numpy()

def compute_similarity(img1_path, img2_path, model, transform):
    feature1 = extract_feature(img1_path, model, transform)
    feature2 = extract_feature(img2_path, model, transform)
    
    # 벡터의 shape를 명시적으로 (1, -1)로 변경
    feature1 = feature1.reshape(1, -1)
    feature2 = feature2.reshape(1, -1)

    similarity = np.dot(feature1, feature2.T)
    return similarity[0][0]  # 스칼라 값으로 변환

# 유사도 계산 예시
img1_path = '../kface/0/0_20_20.jpg'
img2_path = '../kface/9/9_0_0.jpg'
similarity = compute_similarity(img1_path, img2_path, feature_extractor, transform)

print(f"두 이미지의 유사도: {similarity:.4f}")