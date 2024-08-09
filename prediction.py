import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset stored as a CSV file
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Display the first few rows to understand the structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing BMI values with the median BMI
#df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Remove rows with missing BMI values
bmi_notna = df['bmi'].dropna()

# Convert to a list or array for calculation
bmi_values = bmi_notna.values

# Sort the values
sorted_bmi = np.sort(bmi_values)

# Calculate the median
n = len(sorted_bmi)
if n % 2 == 0:
    median = (sorted_bmi[n // 2 - 1] + sorted_bmi[n // 2]) / 2
else:
    median = sorted_bmi[n // 2]

print(f'The calculated median BMI is: {median}')

# Fill missing BMI values with the calculated median
df['bmi'] = df['bmi'].replace(np.nan, median)
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