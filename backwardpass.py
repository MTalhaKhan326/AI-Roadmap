import numpy as np
import time

# 1. Setup: Goal is 20, Input is 10
x = 10.0          # Input
target = 20.0     # The "Truth" (Goal)
weight = 0.5      # Starting with a random "bad" weight
learning_rate = 0.001

print(f"Starting Training... Target: {target}\n")

# 2. The Training Loop (Forward + Backward)
for epoch in range(50):
    # --- FORWARD PASS ---
    prediction = x * weight
    
    # Calculate Loss (Mean Squared Error)
    loss = (prediction - target) ** 2
    
    print(f"Epoch {epoch+1}:")
    print(f"   Prediction: {prediction:.2f}")
    print(f"   Current Loss: {loss:.2f}")

    # --- BACKWARD PASS ---
    # 1. Find the "Error" (how far off are we?)
    error = prediction - target
    
    # 2. Calculate the Gradient (The "Blame" for the weight)
    # Derivative of loss with respect to weight: 2 * error * x
    gradient = 2 * error * x
    
    # 3. Update the Weight (The "Fix")
    weight = weight - (learning_rate * gradient)
    
    print(f"   *Weight updated to: {weight:.2f}*\n")
    time.sleep(1) # Slows it down so you can read it

print("Training Complete! The AI has learned.")