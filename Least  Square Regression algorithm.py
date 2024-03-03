import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from CSV
data = pd.read_csv("Practical/Practical 4/P4A/diabetes.csv")
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Plot the original data
plt.scatter(X, Y)
plt.title('Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)
num = 0
den = 0

for i in range(len(X)):
    num += (X[i] - X_mean) * (Y[i] - Y_mean)
    den += (X[i] - X_mean) ** 2

m = num / den
c = Y_mean - m * X_mean

print(f"Slope (m): {m}, Intercept (c): {c}")

# Making predictions
Y_pred = m * X + c

# Plotting the original and predicted data
plt.scatter(X, Y, label='Actual Data')
plt.scatter(X, Y_pred, color='red', label='Predicted Data')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
