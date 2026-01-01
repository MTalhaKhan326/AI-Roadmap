# In Regression, the AI creates a line to follow the data. 
# In Classification, the AI creates a Decision Boundary to separate the data.
# Regression: "How much?" (e.g., What is the health score?)
# Classification: "Which one?" (e.g., Is this person "Healthy" or "Unhealthy"?)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. DATA (X = Age, y = 1 for Fit, 0 for Unfit)
X = np.array([[20], [25], [30], [35], [40], [41], [43], [45], [50], [55], [60], [70]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]) # Younger are Fit (1), Older are Unfit (0)

# 2. SPLIT & TRAIN (Roadmap Page 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 3. PREDICT
# Let's check a 42-year-old
prediction = clf.predict([[42]])
status = "Fit" if prediction[0] == 1 else "Unfit"
print(f"Prediction for age 42: {status}")

# 4. VISUALIZE THE BOUNDARY
plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k', label='Data Points')
# Generate a smooth curve for the probability
X_range = np.linspace(15, 75, 100).reshape(-1, 1)
y_prob = clf.predict_proba(X_range)[:, 1]

plt.plot(X_range, y_prob, color='green', label='Probability Curve')
plt.axhline(0.5, color='black', linestyle='--', label='Decision Boundary')
plt.xlabel('Age')
plt.ylabel('Probability of being Fit')
plt.legend()
plt.show()