import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset stored as a CSV file
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Display the first few rows to understand the structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

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

y = df.stroke
X = df.drop('stroke', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = .80)

# Calculate the null accuracy
null_accuracy = y_test.value_counts().max() / len(y_test)
print(f"Null Accuracy: {null_accuracy:.4f}")

# For testing withouth SMOTE change X_train_smote, y_train_smote = X_train, y_train
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(solver='liblinear', class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'), 
    "Gradient Boosting": GradientBoostingClassifier(), 
    "Support Vector Machine": SVC(probability=True, class_weight='balanced'),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Evaluate classifiers
results = []
confusion_matrices = {}

# For testing withouth SMOTE change X_train_smote, y_train_smote = X_train, y_train
for name, clf in classifiers.items():
    clf.fit(X_train_smote, y_train_smote)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) 

    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Classifier": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    })

    # Store confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

# Create a DataFrame to display the results
results_df = pd.DataFrame(results)
print(results_df)

# Plot confusion matrices
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], square=True, linewidths=.5)
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')
    
    # Add text annotations for clarity
    for j in range(2):
        for k in range(2):
            plt.text(j-0.3, k+0.1, str(cm[k, j]), color="black", fontsize=12)

# Remove unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 8))
for name, clf in classifiers.items():
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12) 

plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)

plt.show()

# Bar plot for comparison of metrics
results_df.plot(x='Classifier', y=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'], kind='bar', figsize=(12, 6))
plt.title('Classifier Comparison')
plt.ylabel('Metric Value')
plt.xticks(rotation=45)
plt.show()