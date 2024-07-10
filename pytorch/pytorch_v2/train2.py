# train.py
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

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, margin=0.5, scale=30):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, 512))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        cos_theta = F.linear(embeddings, F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        theta = torch.acos(cos_theta)
        target_logits = torch.cos(theta + self.margin)
        logits = cos_theta * (1 - labels) + target_logits * labels
        logits *= self.scale
        return logits

def get_data_loaders(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def train_one_epoch(epoch, model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(imgs)
        labels_one_hot = F.one_hot(labels, num_classes=criterion.num_classes).float()
        logits = criterion(embeddings, labels_one_hot)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings = model(imgs)
            labels_one_hot = F.one_hot(labels, num_classes=criterion.num_classes).float()
            logits = criterion(embeddings, labels_one_hot)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)




if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_data_loaders('../../kface_all', '../../kface', batch_size=32)

    model = ResNetBackbone().to(device)
    criterion = ArcFaceLoss(num_classes=len(train_loader.dataset.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_loss = 9999
    for epoch in tqdm(range(num_epochs)):
        train_one_epoch(epoch, model, criterion, optimizer, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_arcface_model.pth')
            print(f'Best model saved with validation loss: {best_loss}')

    torch.save(model.state_dict(), 'final_arcface_model.pth')