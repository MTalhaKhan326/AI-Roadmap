import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. DATA & FULL NORMALIZATION
X_raw = np.array([[20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75], [77], [79], [81], [84], [87], [90]])
y_raw = np.array([[95], [92], [88], [85], [80], [78], [59], [65], [60], [57], [50], [45], [56], [52], [63], [54], [47], [39]])

# Scale both X and y to 0-1 range for stable gradients
X_min, X_max = X_raw.min(), X_raw.max()
y_min, y_max = y_raw.min(), y_raw.max()

X_norm = (X_raw - X_min) / (X_max - X_min)
y_norm = (y_raw - y_min) / (y_max - y_min)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.3, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 2. MODEL STRUCTURE
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = MLPModel()
criterion = nn.MSELoss() # Optimizer uses MSE for smooth gradients
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. TRAINING WITH EARLY STOPPING & MAE LOGGING
epochs = 20000
patience = 100  
best_loss = float('inf')
counter = 0

print("--- Starting Training (Monitoring MAE) ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)
    
    loss_mse = criterion(preds, y_train) 
    loss_mse.backward()
    optimizer.step()
    
    # Calculate MAE for terminal display (converted to real health points)
    with torch.no_grad():
        mae_train = torch.mean(torch.abs(preds - y_train)) * (y_max - y_min)
    
    if loss_mse.item() < best_loss:
        best_loss = loss_mse.item()
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print(f"Early Stopping triggered at Epoch {epoch+1}!")
        break

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}] | Training MAE: {mae_train.item():.4f}")

# 4. FINAL EVALUATION & DE-NORMALIZATION
model.eval()
with torch.no_grad():
    y_pred_norm = model(X_test)
    
    # Scale back to original Health Score units
    y_test_orig = y_test.numpy() * (y_max - y_min) + y_min
    y_pred_orig = y_pred_norm.numpy() * (y_max - y_min) + y_min
    
    r2 = r2_score(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)

# 5. PROFESSIONAL VISUALIZATION
X_range_norm = torch.linspace(0, 1, 100).view(-1, 1)
with torch.no_grad():
    y_line_norm = model(X_range_norm)

# Scale back X and Y for the plot
X_plot = X_range_norm.numpy() * (X_max - X_min) + X_min
y_plot = y_line_norm.numpy() * (y_max - y_min) + y_min
X_test_orig = X_test.numpy() * (X_max - X_min) + X_min

plt.figure(figsize=(12, 8))
plt.plot(X_plot, y_plot, color='purple', linewidth=3, label='Prediction Curve')
plt.scatter(X_test_orig, y_test_orig, color='red', s=130, edgecolor='black', zorder=5, label='Actual Test Data')

# Add the Scorecard Box
stats_text = f"Model Scorecard:\n----------------\nR² Score: {r2:.4f}\nMAE: {mae:.2f} points"
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add error labels to each point
for i in range(len(X_test)):
    curr_x, curr_y_act, curr_y_pred = X_test_orig[i].item(), y_test_orig[i].item(), y_pred_orig[i].item()
    plt.vlines(x=curr_x, ymin=curr_y_act, ymax=curr_y_pred, color='gray', linestyle='--', alpha=0.6)
    plt.text(curr_x + 1, curr_y_act, f"Err: {abs(curr_y_act - curr_y_pred):.1f}", fontsize=9)

plt.xlabel('Age')
plt.ylabel('Health Score')
plt.title('Final MLP Performance: Full Normalization & Professional Metrics')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)
plt.show()