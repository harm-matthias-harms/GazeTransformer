import torch
from model.image import FlattenImages

m = FlattenImages()

x = torch.rand(40, 256, 16)
images = torch.rand(256, 2, 3, 256, 256)
out = m(x, images)
print(out.shape)