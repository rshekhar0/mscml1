import pandas as pd
import numpy as np

# Read the data from the CSV file
data = pd.read_csv("C:/Users/rshek/OneDrive/OneDrive - Tang.Xing/Desktop/Machine Learning/Practical/Practical 1/P1B/weather.csv")
print(data, "\n")

# Extracting attributes
attributes = np.array(data)[:, :-1]
print("The attributes are: ", attributes, "\n")

# Segregating the target with positive and negative examples
target = np.array(data)[:, -1]
print("The target is: ", target, "\n")

# Training function to implement the find-s algorithm
def train(c, t):
    specific_hypothesis = None  # Initialize specific_hypothesis to None

    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break

    if specific_hypothesis is not None:
        for i, val in enumerate(c):
            if t[i] == "Yes":
                for x in range(len(specific_hypothesis)):
                    if val[x] != specific_hypothesis[x]:
                        specific_hypothesis[x] = '?'
                    else:
                        pass
    return specific_hypothesis

# Obtaining the final hypothesis
print("The final hypothesis is:", train(attributes, target))
