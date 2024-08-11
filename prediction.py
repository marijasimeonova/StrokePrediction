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

# Display the first few rows to understand the structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing values
# Random Forest gives better predictions when the people with missing BMI are not included
# Remove rows with missing BMI values
df = df.dropna(subset=['bmi'])

# Calculate the median for missing values  => maybe delete later
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


# Data cleaning and preprocessing
# Remove rows with "Other" in the 'gender' column
df = df[df['gender'] != 'Other']
# Remove rows with "Unknown" in the 'smoking_status' column
df = df[df['smoking_status'] != 'Unknown']
# Drop 'id' column as it is a unique identifier and not useful for analysis
df = df.drop(columns=['id'])

# Maybe delete later or comment the checkups
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
# X = df.drop(columns=['stroke'])
# y = df['stroke']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize SMOTE and RandomUnderSampler
# smote = SMOTE(random_state=42)
# rus = RandomUnderSampler(random_state=42)

# # Apply SMOTE to the training data
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Apply RandomUnderSampler to the resampled data
# X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)

# # Initialize Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)

# # Train the model on the resampled data
# rf_model.fit(X_train_res, y_train_res)

# # Make predictions
# y_pred = rf_model.predict(X_test)

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))


# Model training and evaluation
X = df.drop(columns=['stroke'])
y = df['stroke']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
feature_names = X.columns

# Create DataFrame for feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display all feature importances
print(feature_importance_df)

# List of categorical and continuous features
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
continuous_features = ['age', 'avg_glucose_level', 'bmi']

# Loop through each feature and perform the analysis
for feature in df.columns[:-1]:  # Exclude the target column 'stroke'
    print(f"\nFeature: {feature}")
    
    # Feature Importance
    feature_importance = feature_importance_df[feature_importance_df['Feature'] == feature]
    print(feature_importance)
    
    # Correlation for continuous features
    if feature in continuous_features:
        correlation = df[feature].corr(df['stroke'])
        print(f'Correlation between {feature} and stroke: {correlation}')
    
    # Chi-Square Test for categorical features
    if feature in categorical_features:
        contingency_table = pd.crosstab(df[feature], df['stroke'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f'Chi-Square Statistic for {feature}: {chi2}')
        print(f'P-value for {feature}: {p}')


# Visualizations
# Set plot style
sns.set(style="whitegrid")

# Distribution of Avg Glucose Level by Stroke
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='avg_glucose_level', hue='stroke', kde=True, element="step", palette="Set2")
plt.title('Distribution of Avg Glucose Level by Stroke')
plt.xlabel('Avg Glucose Level')
plt.ylabel('Density')
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Distribution of Age by Stroke
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='stroke', kde=True, element="step", palette="Set2")
plt.title('Distribution of Age by Stroke')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Distribution of BMI by Stroke
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='bmi', hue='stroke', kde=True, element="step", palette="Set2")
plt.title('Distribution of BMI by Stroke')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Boxplot for Avg Glucose Level by Stroke
plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='avg_glucose_level', data=df, palette="Set2")
plt.title('Boxplot of Avg Glucose Level by Stroke')
plt.xlabel('Stroke')
plt.ylabel('Avg Glucose Level')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Boxplot for Age by Stroke
plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='age', data=df, palette="Set2")
plt.title('Boxplot of Age by Stroke')
plt.xlabel('Stroke')
plt.ylabel('Age')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Boxplot for BMI by Stroke
plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='bmi', data=df, palette="Set2")
plt.title('Boxplot of BMI by Stroke')
plt.xlabel('Stroke')
plt.ylabel('BMI')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Scatterplot of Age vs. Avg Glucose Level colored by Stroke
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='avg_glucose_level', hue='stroke', data=df, palette="Set2")
plt.title('Age vs. Avg Glucose Level by Stroke')
plt.xlabel('Age')
plt.ylabel('Avg Glucose Level')
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Scatterplot of BMI vs. Avg Glucose Level colored by Stroke
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='avg_glucose_level', hue='stroke', data=df, palette="Set2")
plt.title('BMI vs. Avg Glucose Level by Stroke')
plt.xlabel('BMI')
plt.ylabel('Avg Glucose Level')
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Countplot for Hypertension by Stroke
plt.figure(figsize=(10, 6))
sns.countplot(x='hypertension', hue='stroke', data=df, palette="Set2")
plt.title('Stroke Distribution by Hypertension')
plt.xlabel('Hypertension')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Countplot for Heart Disease by Stroke
plt.figure(figsize=(10, 6))
sns.countplot(x='heart_disease', hue='stroke', data=df, palette="Set2")
plt.title('Stroke Distribution by Heart Disease')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Countplot for Ever Married by Stroke
plt.figure(figsize=(10, 6))
sns.countplot(x='ever_married', hue='stroke', data=df, palette="Set2")
plt.title('Stroke Distribution by Marital Status')
plt.xlabel('Ever Married')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Countplot for Work Type by Stroke
plt.figure(figsize=(10, 6))
sns.countplot(x='work_type', hue='stroke', data=df, palette="Set2")
plt.title('Stroke Distribution by Work Type')
plt.xlabel('Work Type')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4], ['Children', 'Govt Job', 'Never Worked', 'Private', 'Self-Employed'])
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Countplot for Residence Type by Stroke
plt.figure(figsize=(10, 6))
sns.countplot(x='Residence_type', hue='stroke', data=df, palette="Set2")
plt.title('Stroke Distribution by Residence Type')
plt.xlabel('Residence Type')
plt.ylabel('Count')
plt.xticks([0, 1], ['Rural', 'Urban'])
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()

# Countplot for Smoking Status by Stroke
plt.figure(figsize=(10, 6))
sns.countplot(x='smoking_status', hue='stroke', data=df, palette="Set2")
plt.title('Stroke Distribution by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Smoker', 'Smoker'])
plt.legend(title='Stroke', labels=['No', 'Yes'])
plt.show()