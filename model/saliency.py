import os
import torch
import torch.nn as nn
from torchvision import transforms

from eml_net import resnet, decoder


class SaliencyMap(nn.Module):
    def __init__(self, size):
        super(SaliencyMap, self).__init__()
        self.img_model = resnet.resnet50(os.path.join(
            os.path.dirname(__file__), 'eml_net/checkpoints/res_imagenet.pth'))
        self.pla_model = resnet.resnet50(os.path.join(
            os.path.dirname(__file__), 'eml_net/checkpoints/res_places.pth'))
        self.decoder_model = decoder.build_decoder(os.path.join(
            os.path.dirname(__file__), 'eml_net/checkpoints/res_decoder.pth'), size, 5, 5)

        self.img_model.eval()
        self.pla_model.eval()
        self.decoder_model.eval()

        self.resize = transforms.Resize(size)
        self.resize_map = transforms.Resize((24, 24))

    def forward(self, x):
        x = self.resize(x)
        with torch.no_grad():
            img_feat = self.img_model(x, decode=True)
            pla_feat = self.pla_model(x, decode=True)
            pred = self.decoder_model([img_feat, pla_feat])
        pred = self.resize_map(pred)
        self.normalize(pred)
        return pred

    def normalize(self, x):
        x -= x.min()
        x /= x.max()
