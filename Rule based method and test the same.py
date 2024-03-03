#pip install matplotlib pandas numpy scipy scikit-learn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# Load the data
customer_data = pd.read_csv("Practical/Practical 7/P7B/diabetes.csv")

# Display the shape and the first few rows of the data
print("Shape of the data:", customer_data.shape)
print("First few rows of the data:")
print(customer_data.head())

# Select relevant columns for clustering
data = customer_data.iloc[:, 3:5].values

# Plot dendrogram
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward', metric='euclidean'))

# Perform hierarchical clustering
cluster = AgglomerativeClustering(n_clusters=5, linkage='ward', distance_threshold=None, compute_full_tree=True, affinity='euclidean')
cluster_labels = cluster.fit_predict(data)

# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='rainbow')
plt.title("Clusters of Customers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Rule-based method
def rule_based_clustering(data):
    cluster_labels = []
    for point in data:
        if point[0] < 100 and point[1] < 50:
            cluster_labels.append(0)  # Cluster 0
        elif point[0] >= 100 and point[1] < 50:
            cluster_labels.append(1)  # Cluster 1
        elif point[0] < 100 and point[1] >= 50:
            cluster_labels.append(2)  # Cluster 2
        else:
            cluster_labels.append(3)  # Cluster 3
    return np.array(cluster_labels)

# Apply rule-based clustering
rule_based_labels = rule_based_clustering(data)

# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c=rule_based_labels, cmap='rainbow')
plt.title("Rule-based Clusters of Customers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Summary Report
summary_report = pd.DataFrame({
    'Method': ['Hierarchical Clustering', 'Rule-based Clustering'],
    'Number of Clusters': [5, 4],  # Number of clusters used in each method
    'Silhouette Score': [None, None],  # Silhouette score can be calculated if needed
})

print("Summary Report:")
print(summary_report)
