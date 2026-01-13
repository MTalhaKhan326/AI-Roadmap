import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 1. SETUP TRANSFORMATIONS
# Training Transform: Uses Augmentation (Random Flips/Rotation) to make the model "tougher"
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip the image
    transforms.RandomRotation(10),           # Rotate by 10 degrees
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Testing Transform: No Augmentation, just standard prep
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. DEFINE THE CNN ARCHITECTURE
class AnimalCNN(nn.Module):
    def __init__(self):
        super(AnimalCNN, self).__init__()
        # Increased filters from 16/32 to 32/64 to help distinguish Cat vs Dog
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 
        x = self.flatten(x)                     
        x = self.relu(self.fc1(x))              
        x = self.fc2(x)                         
        return x

model = AnimalCNN()
criterion = nn.CrossEntropyLoss()
# Set Learning Rate to a balanced 0.0005
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 3. TRAINING FUNCTION
def train_model(epochs=15, save_path='animal_model_v3.pth'):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    print(f"Starting training for {epochs} epochs with Data Augmentation...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as '{save_path}'")

# 4. PREDICTION FUNCTION
def predict_local_image(image_path, model):
    if not os.path.exists(image_path):
        print(f"File {image_path} not found.")
        return

    img_raw = Image.open(image_path).convert('RGB')
    img_tensor = test_transform(img_raw).unsqueeze(0) 

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted = torch.max(outputs, 1)
        
    class_idx = predicted.item()
    plt.imshow(img_raw)
    plt.title(f"Pred: {classes[class_idx]} ({probabilities[class_idx].item()*100:.2f}%)")
    plt.axis('off')
    plt.show()

# --- EXECUTION ---
MODEL_NAME = 'animal_model_v3.pth' # Changed name to force a fresh start

if os.path.exists(MODEL_NAME):
    model.load_state_dict(torch.load(MODEL_NAME))
    print(f"Loaded {MODEL_NAME} weights!")
else:
    train_model(epochs=15, save_path=MODEL_NAME)

# Test on a dog image from the test set
testset_raw = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
for img, label in testset_raw:
    if classes[label] == 'dog':
        img.save("test_dog.jpg")
        break

predict_local_image("test_dog.jpg", model)