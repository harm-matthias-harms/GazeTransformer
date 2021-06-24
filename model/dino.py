import torch
import torch.nn as nn

class DinoBackbone(nn.Module):
  def __init__(self):
      super().__init__()
      self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
      for param in self.dino.parameters():
        param.requires_grad = False

  def forward(self, src, images):
    with torch.no_grad():
      images_0 = images[:, 0]
      images_1 = images[:, 1]
      
      images_0 = self.dino(images_0)
      images_1 = self.dino(images_1).flatten(1)

      images = torch.cat((images_1.repeat(20, 1, 1), images_0.repeat(20, 1, 1)))
      return torch.cat((src, images), 2)
