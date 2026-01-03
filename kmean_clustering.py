import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. THE DATA
X = np.array([
    [20, 95], [22, 90], [25, 92], [30, 85], [35, 80], 
    [40, 60], [45, 55], [50, 50], [55, 45], [60, 40],
    [65, 30], [70, 25], [75, 20]
])

# 2. TRAIN THE MODEL
K = 3
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(X)

# 3. CREATE THE BOUNDARY MESH (The Background Territories)
h = 0.5 
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict for every "tiny dot" on the background
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4. VISUALIZATION
plt.figure(figsize=(12, 8))

# Draw the colored "Territories"
plt.contourf(xx, yy, Z, cmap='Pastel1', alpha=0.6)

# Plot actual data points
labels = kmeans.labels_
plt.scatter(X[:, 0], X[:, 1], c=labels, s=100, cmap='viridis', edgecolor='k', label='Data Points', zorder=5)

# Plot Centroids (The "Kings")
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=400, marker='X', label='Centroids', zorder=6)

# --- THE ADDITION: Dotted Lines + Distance Numbers ---
for i in range(len(X)):
    centroid = centroids[labels[i]]
    
    # Draw the dotted line
    plt.plot([X[i, 0], centroid[0]], [X[i, 1], centroid[1]], 
             color='black', linestyle=':', alpha=0.4, zorder=2)
    
    # Calculate Euclidean Distance
    dist = np.sqrt(np.sum((X[i] - centroid)**2))
    
    # Add the Distance Number as a label
    plt.text(X[i, 0] + 0.8, X[i, 1] + 1, f'{dist:.1f}', 
             fontsize=9, fontweight='bold', color='darkred',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

plt.title(f'K-Means: Territories & Precise Distances (K={K})')
plt.xlabel('Age')
plt.ylabel('Health Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()