import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# 1. THE DATA
X = np.array([[20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75]])
y = np.array([95, 92, 88, 85, 80, 78, 57, 65, 60, 55, 50, 45])

# 2. THE SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CREATE THE MLP WITH VERBOSE OUTPUT
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50), 
    activation='relu',            
    solver='adam',                
    max_iter=500,                 
    learning_rate_init=0.01,
    verbose=True,  # <--- THIS LINE prints the Loss for every Epoch in your terminal
    random_state=42
)

# 4. TRAIN (You will see the Loss dropping in the terminal now)
print("Starting Training...\n")
mlp.fit(X_train, y_train)

# 5. PREDICT
y_pred_test = mlp.predict(X_test)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = mlp.predict(X_range)

# 6. VISUALIZATION WITH LABELS
plt.figure(figsize=(12, 7))
plt.plot(X_range, y_line, color='purple', linewidth=2, label='MLP Prediction Curve')
plt.scatter(X_test, y_test, color='red', s=100, edgecolor='black', zorder=5, label='Actual Test Data')

# DRAW ERROR LINES AND ADD NUMERIC LABELS
for i in range(len(X_test)):
    curr_x = X_test[i].item()
    curr_y_actual = y_test[i].item()
    curr_y_pred = y_pred_test[i].item()
    
    # Line between dot and curve
    plt.vlines(x=curr_x, ymin=curr_y_actual, ymax=curr_y_pred, 
               color='gray', linestyle='--', alpha=0.6)
    
    # Text labels on the graph
    plt.text(curr_x + 1, curr_y_actual, 
             f"Act: {curr_y_actual}\nPred: {curr_y_pred:.1f}\nErr: {abs(curr_y_actual - curr_y_pred):.1f}", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('Age')
plt.ylabel('Health Score')
plt.title('MLP Training Progress & Final Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()