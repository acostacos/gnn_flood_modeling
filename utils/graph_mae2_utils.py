import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, *args):
        logits = self.linear(x)
        return logits
