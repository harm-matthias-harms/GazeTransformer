import torch.nn as nn

from .head import Head

# works for small sequences
# head brings no improvement
class Transformer1(nn.Module):
    def __init__(self, feature_number):
        super().__init__()
        self.transformer = nn.Transformer(feature_number)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return output.transpose(-2, -3)[:, :, :2]

# no considerable result
class Transformer2(nn.Module):
    def __init__(self, feature_number):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            feature_number, nhead=8, dim_feedforward=feature_number)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.decoder = Head(feature_number)

    def forward(self, src, tgt):
        output = self.encoder(src)
        output = output.transpose(-2, -3)
        return self.decoder(output)

# learns to slow for a single sample
# but needs only gaze as future data
# learns even to slow for a big dataset
# bigger nhead not possible...
# seems to not learn with images
class Transformer3(nn.Module):
    def __init__(self, feature_number):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            feature_number, nhead=8, dim_feedforward=feature_number)
        decoder_layers = nn.TransformerDecoderLayer(2, nhead=1)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.encoder_head = Head(feature_number)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=6)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        memory = self.encoder_head(memory)
        output = self.decoder(tgt[:, :, :2], memory)
        output = output.transpose(-2, -3)
        return output[:, :, :2]
