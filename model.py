import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)   # 32x32 → 30x30
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 30x30 → 28x28

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)                 # 28x28 → 14x14

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
