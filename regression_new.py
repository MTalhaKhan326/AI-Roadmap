import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. CREATE DATA (X = Age, y = Health Score)
X = np.array([[20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75]])
y = np.array([95, 92, 88, 85, 80, 78, 57, 65, 60, 55, 50, 45])

# 2. THE TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN THE MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 4. PREDICT
y_pred_test = model.predict(X_test)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_range)

# 5. VISUALIZATION
plt.figure(figsize=(12, 7))

# Plot the Prediction Line (The AI's Logic)
plt.plot(X_range, y_line, color='blue', linewidth=2, label='AI Trend Line')

# Plot ONLY Testing Data (The Truth)
plt.scatter(X_test, y_test, color='red', s=120, edgecolor='black', zorder=5, label='Actual Test Data')

# 6. ADD ERROR LINES AND TEXT LABELS
for i in range(len(X_test)):
    # Unwrapping the data from numpy arrays to avoid errors
    current_age = X_test[i].item()
    actual_health = y_test[i].item()
    pred_health = y_pred_test[i].item()
    
    # Draw dashed error line (Residual)
    plt.vlines(x=current_age, ymin=actual_health, ymax=pred_health, 
               color='gray', linestyle='--', alpha=0.6)
    
    # Calculate difference
    error = actual_health - pred_health
    
    # Label each point on the graph
    plt.text(current_age + 1, actual_health, 
             f"Age: {current_age}\nActual: {actual_health}\nPred: {pred_health:.1f}\nErr: {error:.1f}", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver'))

plt.xlabel('Age (X-axis)')
plt.ylabel('Health Score (Y-axis)')
plt.title('Predicting Health Score from Age (Testing Analysis)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.4)
plt.show()