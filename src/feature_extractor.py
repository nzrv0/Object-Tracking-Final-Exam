from helpers import get_device

import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


device = get_device()


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        for param in vgg_backbone.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(vgg_backbone.features.children())[:-1])

    def forward(self, image):
        self.backbone.eval().to(device)
        return self.backbone(image)
