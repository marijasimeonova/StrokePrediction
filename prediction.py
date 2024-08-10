import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset stored as a CSV file
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Display the first few rows to understand the structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing BMI values with the median BMI
#df['bmi'] = df['bmi'].fillna(df['bmi'].median())


# Random Forest gives better predictions when the people with missing BMI are not included
# Remove rows with missing BMI values
df = df.dropna(subset=['bmi'])

# # Remove rows with missing BMI values
# bmi_notna = df['bmi'].dropna()

# # Convert to a list or array for calculation
# bmi_values = bmi_notna.values

# # Sort the values
# sorted_bmi = np.sort(bmi_values)

# # Calculate the median
# n = len(sorted_bmi)
# if n % 2 == 0:
#     median = (sorted_bmi[n // 2 - 1] + sorted_bmi[n // 2]) / 2
# else:
#     median = sorted_bmi[n // 2]

# print(f'The calculated median BMI is: {median}')

# # Fill missing BMI values with the calculated median
# df['bmi'] = df['bmi'].replace(np.nan, median)

print(df.head())

# Verify that there are no more missing values in the BMI column
print(df.isnull().sum())

# Display value counts for categorical columns
print(df['gender'].value_counts())
print(df['smoking_status'].value_counts())
print(df['work_type'].value_counts())
print(df['Residence_type'].value_counts())
print(df['ever_married'].value_counts())
print(df['hypertension'].value_counts())
print(df['heart_disease'].value_counts())
print(df['stroke'].value_counts())

# Remove rows with "Other" in the 'gender' column
df = df[df['gender'] != 'Other']

# Remove rows with "Unknown" in the 'smoking_status' column
df = df[df['smoking_status'] != 'Unknown']

# Display basic statistics for numerical columns after data cleaning
print(df.describe())

# Display value counts for categorical columns after data cleaning
print(df['gender'].value_counts())
print(df['smoking_status'].value_counts())
print(df['work_type'].value_counts())
print(df['Residence_type'].value_counts())
print(df['ever_married'].value_counts())
print(df['hypertension'].value_counts())
print(df['heart_disease'].value_counts())
print(df['stroke'].value_counts())

# Drop 'id' column as it is a unique identifier and not useful for analysis
df = df.drop(columns=['id'])

# Convert 'gender' to numeric values
gender_mapping = {'Male': 0, 'Female': 1}
df['gender'] = df['gender'].map(gender_mapping)

# Convert 'ever_married' to numeric values
married_mapping = {'No': 0, 'Yes': 1}
df['ever_married'] = df['ever_married'].map(married_mapping)

# Convert 'Residence_type' to numeric values
residence_mapping = {'Rural': 0, 'Urban': 1}
df['Residence_type'] = df['Residence_type'].map(residence_mapping)

# Convert 'work_type' to numeric values
work_type_mapping = {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4}
df['work_type'] = df['work_type'].map(work_type_mapping)

# Convert 'smoking_status' to numeric values
smoking_status_mapping = {'formerly smoked': 1, 'never smoked': 0, 'smokes': 1}
df['smoking_status'] = df['smoking_status'].map(smoking_status_mapping)

# Now the dataset is fully numeric
print(df.head())

# Check if the data is now all numeric
print(df.info())

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