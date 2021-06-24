import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNetBackbone(nn.Module):
  def __init__(self):
      super().__init__()
      self.resnet = nn.Sequential(*(list(resnet50(pretrained=True).children())[:-1]))
      for param in self.resnet.parameters():
        param.requires_grad = False

  def forward(self, src, images):
    with torch.no_grad():
      images_0 = images[:, 0]
      images_1 = images[:, 1]
      
      images_0 = self.resnet(images_0).flatten(1)
      images_1 = self.resnet(images_1).flatten(1)

      images = torch.cat((images_1.repeat(20, 1, 1), images_0.repeat(20, 1, 1)))
      return torch.cat((src, images), 2)