import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        hidden_features = in_features * 2
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, *args):
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x
