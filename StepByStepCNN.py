import torch
import torch.nn as nn

class DebugCNN(nn.Module):
    def __init__(self):
        super(DebugCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        print(f"Input Shape: {x.shape}") # [Batch, Color, H, W]
        
        x = self.conv1(x)
        print(f"After Conv1 (Extracting 16 features): {x.shape}")
        
        x = torch.relu(x)
        x = self.pool(x)
        print(f"After Pool1 (Size cut in half): {x.shape}")
        
        x = self.conv2(x)
        print(f"After Conv2 (Deepening to 32 features): {x.shape}")
        
        x = torch.relu(x)
        x = self.pool(x)
        print(f"After Pool2 (Size cut in half again): {x.shape}")
        
        x = self.flatten(x)
        print(f"After Flatten (Turning into a single line): {x.shape}")
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        print(f"Final Output (10 Class probabilities): {x.shape}")
        return x

# Create a fake 'image' to test the flow
test_image = torch.randn(1, 3, 32, 32) # 1 image, 3 colors, 32x32 pixels
model = DebugCNN()
model(test_image)