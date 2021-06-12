import torch
import torch.nn as nn
from torchvision.transforms import Resize, Grayscale

class FlattenImages(nn.Module):
  # maybe try black and white image, otherwise too much data and no benefit
  def __init__(self):
      super().__init__()
      self.to_grayscale = Grayscale()
      self.resize = Resize((32, 32))
      # 24 * 24 * 3 = 1728

  def forward(self, x, images):
    images_0 = images[:, 0]
    images_1 = images[:, 1]
    images_0 = self.resize(self.to_grayscale(images_0)).flatten(1)
    images_1 = self.resize(self.to_grayscale(images_1)).flatten(1)
    # time descending, last image -> first image
    images = torch.cat((images_1.repeat(20, 1, 1), images_0.repeat(20, 1, 1)))
    return torch.cat((x, images), 2)