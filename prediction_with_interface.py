import wx
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class StrokeRiskPredictor(wx.Frame):
    def __init__(self, parent, title):
        super(StrokeRiskPredictor, self).__init__(parent, title=title, size=(400, 400))
        
        self.panel = wx.Panel(self)
        self.grid_sizer = wx.GridBagSizer(10, 10)
        
        # Add header
        self.header = wx.StaticText(self.panel, label="Stroke Risk Prediction", style=wx.ALIGN_CENTER)
        self.grid_sizer.Add(self.header, pos=(0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL, border=10)
        
        # Define labels and input fields
        self.fields = {
            "Gender": wx.Choice(self.panel, choices=["Male", "Female"]),
            "Age": wx.TextCtrl(self.panel),
            "Hypertension": wx.Choice(self.panel, choices=["No", "Yes"]),
            "Heart Disease": wx.Choice(self.panel, choices=["No", "Yes"]),
            "Average Glucose Level": wx.TextCtrl(self.panel),
            "BMI": wx.TextCtrl(self.panel),
            "Ever Married": wx.Choice(self.panel, choices=["No", "Yes"]),
            "Residence Type": wx.Choice(self.panel, choices=["Rural", "Urban"]),
            "Work Type": wx.Choice(self.panel, choices=["children", "Govt_job", "Never_worked", "Private", "Self-employed"]),
            "Smoking Status": wx.Choice(self.panel, choices=["never smoked", "formerly smoked", "smokes"])
        }
        
        # Add fields to the sizer
        row = 1
        for label, field in self.fields.items():
            self.grid_sizer.Add(wx.StaticText(self.panel, label=label), pos=(row, 0), flag=wx.EXPAND | wx.ALL, border=5)
            self.grid_sizer.Add(field, pos=(row, 1), flag=wx.EXPAND | wx.ALL, border=5)
            row += 1
        
        # Add button and result display
        self.predict_button = wx.Button(self.panel, label="Predict Risk")
        self.result_display = wx.StaticText(self.panel, label="")
        
        self.grid_sizer.Add(self.predict_button, pos=(row, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL, border=10)
        self.grid_sizer.Add(self.result_display, pos=(row+1, 0), span=(1, 2), flag=wx.EXPAND | wx.ALL, border=10)
        
        self.panel.SetSizerAndFit(self.grid_sizer)
        
        self.predict_button.Bind(wx.EVT_BUTTON, self.on_predict)
        
        self.Bind(wx.EVT_CLOSE, self.on_close)
        
        self.rf_model, self.column_names = self.train_model()
    
    def train_model(self):
        # Load the dataset
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')

        # Handle missing values
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

        # Prepare data for training and testing
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

        # Optional: Evaluate the model (can be removed for production)
        y_pred = rf_model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        return rf_model, X_train.columns

    def on_predict(self, event):
        try:
            # Collect user input
            input_data = {
                'gender': [self.fields['Gender'].GetSelection()],
                'age': [float(self.fields['Age'].GetValue())],
                'hypertension': [self.fields['Hypertension'].GetSelection()],
                'heart_disease': [self.fields['Heart Disease'].GetSelection()],
                'avg_glucose_level': [float(self.fields['Average Glucose Level'].GetValue())],
                'bmi': [float(self.fields['BMI'].GetValue())],
                'ever_married': [self.fields['Ever Married'].GetSelection()],
                'Residence_type': [self.fields['Residence Type'].GetSelection()],
                'work_type': [self.fields['Work Type'].GetSelection()],
                'smoking_status': [self.fields['Smoking Status'].GetSelection()]
            }
            
            input_df = pd.DataFrame(input_data)
            input_df = input_df.reindex(columns=self.column_names, fill_value=0)
            
            prediction = self.rf_model.predict(input_df)[0]
            
            # Interpret and display the result
            result = "High risk of having a stroke." if prediction == 1 else "Low risk of having a stroke."
            self.result_display.SetLabel(result)
        except Exception as e:
            self.result_display.SetLabel(f"Error: {e}. Please enter valid input values.")

    def on_close(self, event):
        self.Destroy()

if __name__ == "__main__":
    app = wx.App(False)
    frame = StrokeRiskPredictor(None, title="Stroke Risk Prediction")
    frame.Show()
    app.MainLoop()
