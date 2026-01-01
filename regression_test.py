import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. CREATE DATA (Roadmap Page 1: NumPy)
# X = Age, y = Health Score
X = np.array([[20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75]])
y = np.array([95, 92, 88, 85, 80, 78, 57, 65, 60, 55, 50, 45])

# 2. THE TRAIN/TEST SPLIT (The Pro Way to Verify Truth)
# We save 20% of the data for testing (the AI won't see these during training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN THE MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 4. PREDICT
y_pred_test = model.predict(X_test)
y_line = model.predict(X) # For drawing the regression line

# 5. EVALUATE (Roadmap Page 2: Evaluation)
print(f"Model Accuracy (R2 Score): {r2_score(y_test, y_pred_test):.4f}")
print(f"Average Error (MSE): {mean_squared_error(y_test, y_pred_test):.2f}")

# 6. VISUALIZATION
plt.figure(figsize=(10, 6))

# Plot Training Data (Blue)
plt.scatter(X_train, y_train, color='blue', label='Training Data (AI Saw This)')

# Plot Testing Data (Red) - These are the "Truth" points we used to check the AI
plt.scatter(X_test, y_test, color='red', s=100, edgecolor='black', label='Testing Data (The Truth Check)')

# Plot the Regression Line (The AI's Pattern)
plt.plot(X, y_line, color='green', linewidth=2, label='AI Prediction Line')

plt.xlabel('Age')
plt.ylabel('Health Score')
plt.title('Health vs Age: Training vs Testing Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()