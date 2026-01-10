import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1. Convolutional Layer
        # Input channels=3 (RGB), Output channels=16, Kernel=3x3, Stride=1, Padding=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # 2. Activation Function
        self.relu = nn.ReLU()
        
        # 3. Max Pooling Layer
        # Takes a 2x2 window and reduces the size by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Forward pass: Conv -> ReLU -> Pool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Create the model
model = SimpleCNN()
print(model)