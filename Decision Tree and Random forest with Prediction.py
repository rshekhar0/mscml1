from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

# Loading the data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75  # Fix the bug in the threshold value

# Creating dataframes with test rows and train rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# Show the number of observations for the test and train dataframes
print('No. of observations in the train data: ', len(train))
print('No. of observations in the test data: ', len(test))

features = df.columns[:4]

# Converting each species into digits
y = pd.factorize(train['species'])[0]

# Creating a random forest classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Training the classifier
clf.fit(train[features], y)

# Making predictions
predictions = clf.predict(test[features])

# Mapping names for the plants for each predicted plant
preds = iris.target_names[clf.predict(test[features])]

# Displaying actual vs. predicted species
print("Actual vs. Predicted Species:")
print(pd.DataFrame({'Actual Species': test['species'], 'Predicted Species': preds}).head())

# Creating confusion matrix
conf_matrix = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
print("\nConfusion Matrix:")
print(conf_matrix)

# Displaying additional metrics
print("\nClassification Report:")
print(classification_report(test['species'], preds))

# Calculating and displaying accuracy score
accuracy = accuracy_score(test['species'], preds)
print('\nThe accuracy score is:', accuracy)

# or
#Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
print("Decision Tree Classifier:")
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predictions
dt_predictions = dt_classifier.predict(X_test)

# Test Score
dt_test_score = accuracy_score(y_test, dt_predictions)
print("Test Score:", dt_test_score)

# Confusion Matrix
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
print("Confusion Matrix:\n", dt_conf_matrix)

# Random Forest Classifier
print("\nRandom Forest Classifier:")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
rf_predictions = rf_classifier.predict(X_test)

# Test Score
rf_test_score = accuracy_score(y_test, rf_predictions)
print("Test Score:", rf_test_score)

# Confusion Matrix
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
print("Confusion Matrix:\n", rf_conf_matrix)
