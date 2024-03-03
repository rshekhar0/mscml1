from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_iris 
# Load the Iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
# Initialize the Decision Tree Classifier 
dt_classifier = DecisionTreeClassifier() 
# Train the model 
dt_classifier.fit(X_train, y_train) 
# Make predictions 
y_pred = dt_classifier.predict(X_test) 
# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy}") 
# Example of classifying a new sample 
new_sample = [[5.0, 3.5, 1.5, 0.2]]  # Example data for a new sample 
predicted_class = dt_classifier.predict(new_sample) 
print(f"Predicted Class for New Sample: {predicted_class}") 

#or

#pip install scikit-learn matplotlib
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier with ID3 algorithm
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.show()


# Summary Report
summary = {
    "Dataset": "Iris",
    "Number of samples": len(X),
    "Number of features": len(iris.feature_names),
    "Number of classes": len(iris.target_names),
    "Split ratio (train:test)": "80:20",
    "Random state": 42,
    "Decision Tree Algorithm": "ID3 (Entropy)",
    "Accuracy": dt_classifier.score(X_test, y_test)
}

# Display the summary report in table format
print("Summary Report")
print("--------------")
for key, value in summary.items():
    print(f"{key}: {value}")