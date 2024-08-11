import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset stored as a CSV file
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Handle missing values
# Random Forest gives better predictions when the people with missing BMI are not included
# Remove rows with missing BMI values
df = df.dropna(subset=['bmi'])

# Data cleaning and preprocessing
df = df[df['gender'] != 'Other']
df = df[df['smoking_status'] != 'Unknown']
df = df.drop(columns=['id'])

# Convert categorical data to numeric values
gender_mapping = {'Male': 0, 'Female': 1}
df['gender'] = df['gender'].map(gender_mapping)

married_mapping = {'No': 0, 'Yes': 1}
df['ever_married'] = df['ever_married'].map(married_mapping)

residence_mapping = {'Rural': 0, 'Urban': 1}
df['Residence_type'] = df['Residence_type'].map(residence_mapping)

work_type_mapping = {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4}
df['work_type'] = df['work_type'].map(work_type_mapping)

smoking_status_mapping = {'formerly smoked': 1, 'never smoked': 0, 'smokes': 1}
df['smoking_status'] = df['smoking_status'].map(smoking_status_mapping)


#Randomforest
X = df.drop(columns=['stroke'])
y = df['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE and RandomUnderSampler
smote = SMOTE(random_state=42)
rus = RandomUnderSampler(random_state=42)

# Apply SMOTE to the training data
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Apply RandomUnderSampler to the resampled data
X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the resampled data
rf_model.fit(X_train_res, y_train_res)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))