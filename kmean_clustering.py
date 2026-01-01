import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. CREATE DATA (Age and Health Score)
# We don't provide a 'y' (labels) because this is Unsupervised Learning
X = np.array([
    [20, 95], [22, 90], [25, 92], [30, 85], [35, 80], 
    [40, 60], [45, 55], [50, 50], [55, 45], [60, 40],
    [65, 30], [70, 25], [75, 20]
])

# 2. DEFINE THE NUMBER OF CLUSTERS (Change K here!)
K = 3

# 3. TRAIN THE K-MEANS MODEL
# The AI will start with random centroids and move them until they hit the 'mean'
# We can also run this code with out random state => Use random_state to keep your centroids from jumping around every time you refresh your screen.
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(X)

# 4. GET RESULTS
labels = kmeans.labels_  # Which cluster each point selected
centroids = kmeans.cluster_centers_  # The final "perfect middle" of each group

# 5. VISUALIZE THE "ART"
plt.figure(figsize=(10, 6))

# Plot the data points, colored by their selected cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, s=100, cmap='viridis', label='Data Points')

# Plot the Centroids (The "Kings" of each cluster)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X', label='Centroids')

plt.title(f'K-Means Clustering with K={K}')
plt.xlabel('Age')
plt.ylabel('Health Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()