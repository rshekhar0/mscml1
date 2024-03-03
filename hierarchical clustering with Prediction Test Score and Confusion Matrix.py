# Install scikit-learn, which includes KMeans, LogisticRegression, train_test_split, accuracy_score, and confusion_matrix pip install scikit-learn

# Install matplotlib for plotting pip install matplotlib

# Install pandas for data manipulation pip install pandas

# Install numpy for numerical computations pip install numpy

# Install seaborn for advanced visualization pip install seaborn

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(iris.target, columns=['Targets'])

# Apply KMeans clustering
kmeans_model = KMeans(n_clusters=3, n_init=10)
kmeans_model.fit(X)
X['KMeans_Clusters'] = kmeans_model.labels_

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier with increased max_iter
classifier = LogisticRegression(max_iter=1000)  # Increase max_iter value
classifier.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Hierarchical clustering doesn't have a direct predict method, so we use the KMeans clusters for prediction.
# The actual hierarchical clustering steps are not performed for prediction in this example.

# Display the scatter plot with KMeans clusters
plt.figure(figsize=(14, 7))

# Real clusters
colormap = np.array(['red', 'cyan', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(X_test['Petal_Length'], X_test['Petal_Width'], c=colormap[y_test['Targets']], s=40)
plt.title("Real clusters")

# K Means Cluster
plt.subplot(1, 2, 2)
plt.scatter(X_test['Petal_Length'], X_test['Petal_Width'], c=colormap[X_test['KMeans_Clusters']], s=40)
plt.title("K Means Cluster (Used for Prediction)")

plt.show()

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Perform hierarchical clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=3)
cluster_labels = hierarchical_clustering.fit_predict(X)

# Split the clustered data and target labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cluster_labels.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Train a classification model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot hierarchical clustering
plt.figure(figsize=(10, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Create a DataFrame for the summary report
summary_report = pd.DataFrame({
    'Metric': ['Accuracy'],
    'Value': [accuracy]
})

# Convert confusion matrix to DataFrame and add to summary report
conf_matrix_df = pd.DataFrame(conf_matrix, columns=iris.target_names, index=iris.target_names)
conf_matrix_df.index.name = 'Actual'
conf_matrix_df.columns.name = 'Predicted'
summary_report = pd.concat([summary_report, conf_matrix_df.unstack().reset_index(name='Value')])

# Display the summary report
print("Summary Report:")
print(summary_report)
