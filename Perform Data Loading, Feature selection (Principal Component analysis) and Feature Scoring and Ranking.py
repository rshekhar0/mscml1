# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression  # Change to mutual_info_regression for continuous target variable

# Step 1: Data Loading
# Load your dataset (replace 'diabetes.csv' with your actual file path or URL)
data = pd.read_csv("C:/Users/rshek/OneDrive/OneDrive - Tang.Xing/Desktop/Machine Learning/Practical/Practical 2/P2A/diabetes.csv")

# Display the first few rows of the dataset to inspect the data
print("Step 1: Data Loading")
print(data.head())

# Step 2: Feature Selection using Principal Component Analysis (PCA)
# Assuming 'X' contains your features and 'y' contains your target variable
X = data.drop('BMI', axis=1)  # Adjust 'target_variable' with the actual name
y = data['BMI']

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Specify the number of components you want to keep
n_components = 5  # Adjust as needed

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Display the explained variance ratio to understand how much variance is retained
print("\nStep 2: Feature Selection using PCA")
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Step 3: Feature Scoring and Ranking
# Compute mutual information scores for each feature
mi_scores = mutual_info_regression(X, y)  # Change to mutual_info_regression for continuous target variable

# Create a DataFrame to display feature scores
feature_scores = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
feature_scores = feature_scores.sort_values(by='MI Score', ascending=False)

# Display the feature scores
print("\nStep 3: Feature Scoring and Ranking")
print(feature_scores)
