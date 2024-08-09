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
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Verify that there are no more missing values in the BMI column
print(df.isnull().sum())