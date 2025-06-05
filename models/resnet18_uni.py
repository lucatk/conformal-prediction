import torch
from torch import nn
from torchvision.models import resnet18

from models.unimodal_head import UnimodalHead


class UnimodalResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super(UnimodalResNet18, self).__init__()
        base_model = resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # exclude FC layer
        self.unimodal_head = UnimodalHead(num_classes, input_dim=512) # 512 is the output dimension of ResNet18's feature extractor

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.unimodal_head(x)
