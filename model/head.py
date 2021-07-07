import torch.nn as nn


class Head(nn.Module):
    def __init__(self, feature_number, inner_head_features = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_number, inner_head_features),
            nn.ReLU(),
            nn.Linear(inner_head_features, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(80, 2)
        )

    def forward(self, x):
        return self.layers(x).unsqueeze(1)
