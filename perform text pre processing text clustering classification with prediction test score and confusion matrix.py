# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Sample text data and corresponding labels
text_data = [
    "this is a sample text",
    "sample text for classification",
    "text clustering is interesting",
    "classification and clustering are different tasks",
    "sample text for clustering analysis",
    "this is another text document"
]
labels = [0, 1, 1, 0, 1, 0]  # Example labels (binary classification)

# Text preprocessing and vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(text_data)

# Text clustering (using KMeans)
kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Text classification with prediction
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate classification performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Text Pre-processing
print("\nText Pre-processing:")
print("Number of Documents:", len(text_data))
print("Vocabulary Size:", len(vectorizer.vocabulary_))
print("Features Extracted:", X.shape[1])

# Text Clustering
print("\nText Clustering:")
print("Cluster Centers:", kmeans.cluster_centers_)
print("Cluster Assignments:", clusters)

# Classification with Prediction
print("\nClassification with Prediction:")
print("Predicted Labels:", y_pred)

# Test Score
print("\nTest Score:")
print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Summary Report
print("\nSummary Report:")
print("---------------")
print("Text Pre-processing:")
print("Number of Documents:", len(text_data))
print("Vocabulary Size:", len(vectorizer.vocabulary_))
print("Features Extracted:", X.shape[1])
print("\nText Clustering:")
print("Cluster Centers:", kmeans.cluster_centers_)
print("Cluster Assignments:", clusters)
print("\nClassification with Prediction:")
print("Predicted Labels:", y_pred)
print("\nTest Score:")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
