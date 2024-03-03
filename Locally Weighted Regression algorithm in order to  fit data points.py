import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def locally_weighted_regression(x, y, query_point, tau):
    # Add a bias term to x
    X = np.vstack((np.ones_like(x), x)).T
    
    # Calculate weights based on the distance between query_point and each data point
    weights = np.exp(-(x - query_point)**2 / (2 * tau**2))
    
    # Calculate the weighted least squares solution
    W = np.diag(weights)
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    
    # Predict the value at the query point
    query_point_vector = np.array([1, query_point])
    prediction = query_point_vector @ theta
    
    return prediction

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 100)
y_true = np.sin(x)
y_noisy = y_true + np.random.normal(0, 0.1, size=len(x))

# Set the query point and bandwidth parameter (tau)
query_point = np.pi / 2
tau = 0.1

# Perform locally weighted regression
prediction = locally_weighted_regression(x, y_noisy, query_point, tau)

# Create summary report table
summary_data = {
    'Query Point': [query_point],
    'Prediction': [prediction],
    'Bandwidth (tau)': [tau]
}
summary_df = pd.DataFrame(summary_data)

# Display the summary report table
print("Summary Report:")
print(summary_df)

# Plot the results
plt.scatter(x, y_noisy, label='Noisy Data')
plt.plot(x, y_true, label='True Function', linestyle='--', color='black')
plt.scatter(query_point, prediction, color='red', marker='x', label='Prediction at Query Point')

plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
