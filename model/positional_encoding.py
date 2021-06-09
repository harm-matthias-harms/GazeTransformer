import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, feature_number):
        super(LearnedPositionalEncoding, self).__init__()
        self.seq_length = 40
        self.pe = nn.Embedding(self.seq_length, feature_number)

        self.register_buffer(
            "position_ids",
            torch.arange(50).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        position_embeddings = position_embeddings.transpose(-3, -2)
        output = x + position_embeddings
        return output

class Time2VecPositionalEncoding(nn.Module):
    def __init__(self, feature_number, kernel_size=8):
        super(Time2VecPositionalEncoding, self).__init__()
        self.k = kernel_size

        self.wb = nn.parameter.Parameter(torch.Tensor(feature_number,))
        self.bb = nn.parameter.Parameter(torch.Tensor(feature_number,))

        self.wa = nn.parameter.Parameter(torch.Tensor(1, feature_number, self.k))
        self.ba = nn.parameter.Parameter(torch.Tensor(1, 40, self.k))

        self.wb.data.uniform_(-1, 1)
        self.bb.data.uniform_(-1, 1)
        self.wa.data.uniform_(-1, 1)
        self.ba.data.uniform_(-1, 1)

    def forward(self, x):
        bias = self.wb * x + self.bb
        dp = torch.tensordot(x, self.wa) + self.ba
        wgts = torch.sin(dp).transpose(-3, -2)
        wgts = wgts.repeat(1, x.size(1), 1)
        return torch.cat((bias, wgts[:x.size(0)]), -1)
