import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import math

class ArcFaceResNet50(nn.Module):
    def __init__(self, num_classes, emb_size=512, s=30.0, m=0.50):
        super(ArcFaceResNet50, self).__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.s = s
        self.m = m
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.embedding = nn.Linear(2048, self.emb_size)
        self.arc_margin_product = ArcMarginProduct(self.emb_size, self.num_classes, s=self.s, m=self.m)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        
        if self.training:
            output = self.arc_margin_product(embedding, labels)
            return output
        else:
            return F.normalize(embedding)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(math.pi - m))
        self.mm = torch.sin(torch.tensor(math.pi - m)) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
    

class ArcFaceResNet50_ver2(nn.Module):
    def __init__(self, num_classes, emb_size=512, s=30.0, m=0.50):
        super(ArcFaceResNet50, self).__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.s = s
        self.m = m
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.embedding = nn.Linear(2048, self.emb_size)
        self.arc_margin_product = ArcMarginProduct(self.emb_size, self.num_classes, s=self.s, m=self.m)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        
        if self.training:
            output = self.arc_margin_product(embedding, labels)
            return output
        else:
            return F.normalize(embedding)