import torch
import torch.nn as nn
import torch.nn.functional as F


class FcNet(nn.Module):
    def __init__(self, x_dim):
        self.fc1 = nn.Linear(x_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.25)
        self.do3 = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        x = self.do2(x)
        x = F.relu(self.fc3(x))
        x = self.do3(x)
        return self.fc4(x)
