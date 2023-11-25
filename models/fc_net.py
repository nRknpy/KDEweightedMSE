import torch
import torch.nn as nn
import torch.nn.functional as F


class FcNet(nn.Module):
    def __init__(self, x_dim):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(x_dim, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
