import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
