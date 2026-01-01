import numpy as np

# 1. Input Data (e.g., Features of a house: Square feet, Age)
# We have 1 house with 2 features.
X = np.array([0.5, 0.8]) 

# 2. Weights & Bias (The AI's 'current knowledge')
# In a real model, these start as random numbers.
W = np.array([0.4, 0.7])  # Importance of each feature
b = 0.1                   # The bias (starting offset)

# 3. The Math (The Forward Pass)
# Z = (Input * Weight) + Bias
z = np.dot(X, W) + b

# 4. Activation Function (The Decision)
# We use Sigmoid to squish the answer between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

prediction = sigmoid(z)

print(f"Input: {X}")
print(f"Weighted Sum (z): {z}")
print(f"Final Prediction (Probability): {prediction:.4f}")