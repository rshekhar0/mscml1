#pip install scikit-learn seaborn matplotlib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.3)

# Initialize the KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=13)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Print classification report
print("Accuracy is: ")
print(classification_report(y_test, y_pred))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Summary Report
summary = {
    "Dataset": "Iris",
    "Number of samples": len(iris_data),
    "Number of features": len(iris.feature_names),
    "Number of classes": len(iris.target_names),
    "Split ratio (train:test)": "70:30",
    "k-Nearest Neighbors Algorithm": "k=13",
    "Accuracy": classifier.score(X_test, y_test)
}

# Display the summary report in table format
print("\nSummary Report")
print("--------------")
for key, value in summary.items():
    print(f"{key}: {value}")
