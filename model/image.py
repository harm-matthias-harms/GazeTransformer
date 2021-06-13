import torch
import torch.nn as nn
from torchvision.transforms import Resize, Grayscale

class FlattenImages(nn.Module):
  # black and white images used, otherwise too much data and no benefit
  def __init__(self):
      super().__init__()
      self.to_grayscale = Grayscale()
      self.resize = Resize((32, 32))
      # 32 * 32= 1024 => 1040

  def forward(self, x, images):
    images_0 = images[:, 0]
    images_1 = images[:, 1]
    images_0 = self.resize(self.to_grayscale(images_0)).flatten(1)
    images_1 = self.resize(self.to_grayscale(images_1)).flatten(1)
    # time descending, last image -> first image
    images = torch.cat((images_1.repeat(20, 1, 1), images_0.repeat(20, 1, 1)))
    return torch.cat((x, images), 2)

class ImagePatches(nn.Module):
  def __init__(self):
      super().__init__()
      self.pool = nn.FractionalMaxPool2d(3, output_size=(16, 16))
      # 16 * 16 * 3 = 768 => 784

  def forward(self, x, images):
    images_0 = images[:, 0]
    images_1 = images[:, 1]
    images_0 = self.pool(images_0).flatten(1)
    images_1 = self.pool(images_1).flatten(1)
    # time descending, last image -> first image
    images = torch.cat((images_1.repeat(20, 1, 1), images_0.repeat(20, 1, 1)))
    return torch.cat((x, images), 2)