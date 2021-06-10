import torch.nn as nn


class Head(nn.Module):
    def __init__(self, feature_number):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_number, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(80, 2)
        )

    def forward(self, x):
        return self.layers(x).unsqueeze(1)
