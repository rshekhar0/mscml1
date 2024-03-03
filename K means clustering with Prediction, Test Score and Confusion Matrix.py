#pip install matplotlib scikit-learn pandas numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(iris.target, columns=['Targets'])

model = KMeans(n_clusters=3, n_init=10)
model.fit(X)

plt.figure(figsize=(14, 7))

# Real clusters
colormap = np.array(['red', 'cyan', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title("Real clusters")

# K Means Cluster
plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title("K Means Cluster")

plt.show()

#or

# Predict clusters for the data points
predicted_labels = model.predict(X)

# Compute silhouette score
silhouette_score = sm.silhouette_score(X, model.labels_)

# Compute inertia
inertia = model.inertia_

# Plot the clusters with predictions
plt.figure(figsize=(14, 7))

# Real clusters
plt.subplot(1, 3, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title("Real clusters")

# K Means Cluster
plt.subplot(1, 3, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title("K Means Cluster")

# Predicted clusters
plt.subplot(1, 3, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[predicted_labels], s=40)
plt.title("Predicted Clusters")

plt.show()

# Print test score (silhouette score)
print("Silhouette Score:", silhouette_score)

# Print inertia
print("Inertia:", inertia)
