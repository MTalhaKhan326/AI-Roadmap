import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. THE DATA (Loading the MNIST Images)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True) # Batch size 1 for testing 1 image

# 2. THE CNN STRUCTURE (The Visual Brain)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: Conv + ReLU + Pooling (Detects edges)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # Shrinks image from 28x28 to 14x14
        
        # Layer 2: Conv + ReLU + Pooling (Detects shapes)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Final Layer: Flatten and Decide (Fully Connected)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 outputs for digits 0-9

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # Block 1
        x = self.pool(self.relu(self.conv2(x))) # Block 2
        x = x.view(-1, 64 * 7 * 7)             # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss() # Standard for Classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. TRAINING (Quickly for 1 epoch)
print("Training the CNN...")
for images, labels in trainloader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 4. THE "TELL ME WHAT THIS IS" TEST
model.eval()
with torch.no_grad():
    # Grab one random image from the test set
    image, label = next(iter(testloader))
    
    # Let the system "look" at it
    output = model(image)
    _, predicted = torch.max(output, 1) # Get the highest probability
    
    # Show the result
    print(f"\nAI Guess: {predicted.item()}")
    print(f"Actual Label: {label.item()}")
    
    # Visualize it
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"AI says this is a: {predicted.item()}")
    plt.show()