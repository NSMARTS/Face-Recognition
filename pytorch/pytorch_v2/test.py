# evaluate.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
# from .model2 import ResNetBackbone, ArcFaceLoss
# from .dataload import get_data_loaders
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import torch.nn.functional as F
from PIL import Image

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 512)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x)
    

def get_data_loaders(test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # train_dataset = ImageFolder(root=train_dir, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    print(test_dataset)
    for i, (data) in enumerate(test_dataset):
        print(data)
        print()
        if i == 2:
            break
        
    print('=========================')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader

def evaluate(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(dataloader):
            print(imgs)
            print(imgs.shape)
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings = model(imgs)
            # print(embeddings)
            print(embeddings.shape)
            predicted = torch.argmax(embeddings, dim=1)
            print('predicted : ',predicted)
            print('labels : ', labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if idx == 1:
                break

    print(f'Accuracy: {correct / total * 100:.2f}%')

# def modelTest(img1, img2, dataloader):

#     for idx, (imgs, labels) in enumerate(dataloader)

#     cosine_sim = F.cosine_similarity(img1, img2, dim=1)

#     print(img1, img2)
#     print(cosine_sim)


if __name__ == '__main__':
    # freeze_support()
    transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader = get_data_loaders('../../kface', batch_size=32)

    img1_path = '../../kface_all/0/0_20_20.jpg'
    img2_path = '../../kface_all/9/9_0_0.jpg'  

    img = Image.open(img1_path).convert('RGB')
    
    img = transform(img)
    print('convert img : ',img)
    print(img.shape)

    model = ResNetBackbone().to(device)
    model.load_state_dict(torch.load('best_arcface_model.pth'))
    model.eval()

    # modelTest()
    evaluate(model, val_loader, device)
