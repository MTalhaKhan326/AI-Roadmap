import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. THE DATA
X_raw = torch.tensor([[20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75], [77], [79], [81], [84], [87], [90]], dtype=torch.float32)
y_raw = torch.tensor([[95], [92], [88], [85], [80], [78], [57], [65], [60], [55], [50], [45], [56], [52], [63], [54], [47], [39]], dtype=torch.float32)
# 2. THE SPLIT (80% Train, 20% Test)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_raw.numpy(), y_raw.numpy(), test_size=0.2, random_state=42)

# Convert back to Tensors one last time (Clean and safe)
X_train = torch.tensor(X_train_np)
y_train = torch.tensor(y_train_np)
X_test = torch.tensor(X_test_np)
y_test = torch.tensor(y_test_np)

# 3. THE MODEL STRUCTURE
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(1, 100)
        self.layer2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

model = MLPModel()
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. THE VERBOSE TRAINING LOOP
print("--- Starting Training (Verbose Mode) ---")
epochs = 20000
for epoch in range(epochs):
    # Forward Pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward Pass (The Update)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # --- THIS PART PRINTS TO YOUR TERMINAL ---
    # We print every 500 epochs so we don't slow down the computer too much
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Training Loss: {loss.item():.4f}")

print("\nTraining Complete! Now testing on unseen data...")

# 5. TESTING & VISUALIZATION (Test Data Only)
model.eval() 
with torch.no_grad():
    y_pred_test = model(X_test)
    X_range = torch.linspace(X_raw.min(), X_raw.max(), 100).view(-1, 1)
    y_line = model(X_range)

plt.figure(figsize=(12, 8))
plt.plot(X_range.numpy(), y_line.numpy(), color='purple', linewidth=2, label='Model Prediction Curve')
plt.scatter(X_test.numpy(), y_test.numpy(), color='red', s=120, edgecolor='black', zorder=5, label='Actual Test Data (Unseen)')

# Error labels for the test points
for i in range(len(X_test)):
    curr_x = X_test[i].item()
    curr_y_actual = y_test[i].item()
    curr_y_pred = y_pred_test[i].item()
    plt.vlines(x=curr_x, ymin=curr_y_actual, ymax=curr_y_pred, color='gray', linestyle='--', alpha=0.7)
    plt.text(curr_x + 1, curr_y_actual, 
             f"Act: {curr_y_actual}\nPred: {curr_y_pred:.1f}\nErr: {abs(curr_y_actual - curr_y_pred):.1f}", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Age')
plt.ylabel('Health Score')
plt.title('Final Exam: Model Performance on Unseen Test Data Only')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()