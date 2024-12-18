import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataLoader import FaceDataset
from model_define import ArcFaceResNet50
from checkpoint import save_checkpoint, load_checkpoint
import torch.nn as nn
import multiprocessing
from tqdm import tqdm
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[ transform ] initialize')
    transform = transforms.Compose([
        transforms.Resize((124, 124)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터셋 및 데이터 로더 생성
    print('[ dataset, dataloader ] generation')
    dataset = FaceDataset('../kface', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    print(len(dataset.classes))

    # 모델, 손실 함수, 옵티마이저 초기화
    print('[ model, optimizer ] initialize')
    model = ArcFaceResNet50(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    num_epochs = 100

    best_accuracy = 0 
    print('[ train start ]')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
        

        # 매 에폭마다 체크포인트 저장
        save_checkpoint(model, optimizer, epoch+1, running_loss/len(dataloader), accuracy, 
                        filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
        
        # 최고 정확도 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(model, optimizer, epoch+1, running_loss/len(dataloader), accuracy, 
                            filename="best_model_checkpoint.pth.tar")

        scheduler.step()

        # 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'arcface_model_epoch_{epoch+1}.pth')

    print('Finished Training')


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows에서 필요
    main()