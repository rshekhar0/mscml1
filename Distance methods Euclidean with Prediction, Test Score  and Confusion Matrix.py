#pip install scikit-learn matplotlib numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN Classifier with Euclidean distance
knn_classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

# Example of classifying a new sample
new_sample = [[5.0, 3.5, 1.5, 0.2]]  # Example data for a new sample
predicted_class = knn_classifier.predict(new_sample)
print(f"Predicted Class for New Sample: {predicted_class}")

#or
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN Classifier with Euclidean distance
knn_classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Train the model
knn_classifier.fit(X_train, y_train)

# Define the number of nearest neighbors
n_neighbors = 3

# Define the colormap for the plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Define the feature indices to plot (choose any two features)
feature1_index = 0
feature2_index = 1

# Extract the feature data for the plot
X_plot = X[:, [feature1_index, feature2_index]]
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

# Generate a meshgrid to create the decision boundary plot
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Use all four features to create the meshgrid for prediction
X_meshgrid = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())]

# Predict the class for each meshgrid point
Z = knn_classifier.predict(X_meshgrid)

# Plot the decision boundaries
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"k-Nearest Neighbors (k={n_neighbors}) Decision Boundaries")
plt.xlabel(iris.feature_names[feature1_index])
plt.ylabel(iris.feature_names[feature2_index])
plt.show()

# Evaluate the model
accuracy = knn_classifier.score(X_test, y_test)
y_pred = knn_classifier.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print summary report
print("Summary Report")
print("--------------")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)
print("\nConfusion Matrix:")
print(conf_matrix)
