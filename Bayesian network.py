import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Load the dataset
data = pd.read_csv("Practical/Practical 8/P8A/ds4.csv")
heart_disease = pd.DataFrame(data)
print(heart_disease)

# Define the Bayesian Network model
model = BayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease'),
    ('diet', 'cholestrol')
])

# Fit the model to the data using Maximum Likelihood Estimation
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Perform variable elimination for inference
HeartDisease_infer = VariableElimination(model)

# Print instructions for entering data
print('For Age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4')
print('For Gender enter Male:0, Female:1')
print('For Family History enter Yes:1, No:0')
print('For Diet enter High:0, Medium:1')
print('For LifeStyle enter Athlete:0, Active:1, Moderate:2, Sedentary:3')
print('For Cholesterol enter High:0, BorderLine:1, Normal:2')

# Prompt user for input data
age = int(input('Enter Age: '))
gender = int(input('Enter Gender: '))
family = int(input('Enter Family History: '))
diet = int(input('Enter Diet: '))
lifestyle = int(input('Enter Lifestyle: '))
cholestrol = int(input('Enter Cholesterol: '))

# Perform inference and print the result
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age': age,
    'Gender': gender,
    'Family': family,
    'diet': diet,
    'Lifestyle': lifestyle,
    'cholestrol': cholestrol
})

print(q)
