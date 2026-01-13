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
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. DEFINE THE CNN ARCHITECTURE
class AnimalCNN(nn.Module):
    def __init__(self):
        super(AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 
        x = self.flatten(x)                     
        x = self.relu(self.fc1(x))              
        x = self.fc2(x)                         
        return x

# --- ADDED THIS SIMPLE FUNCTION TO STOP THE ERROR ---
def train_model(epochs):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")
    torch.save(model.state_dict(), 'animal_model.pth')

# Initialize Model
model = AnimalCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Use the Optimizer in your Training/Prediction logic
if os.path.exists('animal_model.pth'):
    model.load_state_dict(torch.load('animal_model.pth'))
    print("Loaded weights!")
else:
    # This now works because we defined the function above
    train_model(epochs=30) 

# 3. SINGLE IMAGE PREDICTION FUNCTION
def predict_local_image(image_path, model):
    try:
        img_raw = Image.open(image_path).convert('RGB')
        img_tensor = transform(img_raw).unsqueeze(0) 

        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            probability = probabilities[class_idx].item() * 100

        plt.imshow(img_raw)
        plt.title(f"Prediction: {classes[class_idx]} ({probability:.2f}%)")
        plt.axis('off')
        plt.show()

        print(f"\nResults for: {image_path}")
        print("-" * 30)
        for i in range(len(classes)):
            print(f"{classes[i]:<10}: {probabilities[i].item()*100:>6.2f}%")
            
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")

# --- RUN PREDICTION ---
predict_local_image('horse.jpg', model)