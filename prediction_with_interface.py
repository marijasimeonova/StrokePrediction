import wx
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class StrokeRiskPredictor(wx.Frame):
    def __init__(self, parent, title):
        super(StrokeRiskPredictor, self).__init__(parent, title=title, size=(400, 800))
        
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour("#dad7cd")  # Background color
        
        self.grid_sizer = wx.GridBagSizer(10, 10)
        
        # Add header with increased font size
        self.header = wx.StaticText(self.panel, label="Stroke Risk Prediction", style=wx.ALIGN_CENTER)
        self.header.SetForegroundColour("#3a5a40")  # Text color
        header_font = wx.Font(24, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.header.SetFont(header_font)
        self.grid_sizer.Add(self.header, pos=(0, 0), span=(1, 2), flag=wx.ALIGN_CENTER | wx.ALL, border=10)
        
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
        
        # Apply input field background color
        for field in self.fields.values():
            field.SetBackgroundColour("#ffffff")  # Input fields color
            if isinstance(field, wx.Choice):
                field.SetForegroundColour("#3a5a40")  # Text color of dropdowns
        
        # Add fields to the sizer
        row = 1
        for label, field in self.fields.items():
            label_text = wx.StaticText(self.panel, label=label)
            label_text.SetForegroundColour("#3a5a40")  # Text color
            self.grid_sizer.Add(label_text, pos=(row, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)
            self.grid_sizer.Add(field, pos=(row, 1), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)
            row += 1
        
        # Add button and result display
        self.predict_button = wx.Button(self.panel, label="Predict Risk")
        self.predict_button.SetBackgroundColour("#80ED99")  # Button background color
        self.predict_button.SetForegroundColour("#344e41")  # Button text color
        self.predict_button.SetSize((300, 50))  # Increase button size
        
        # Bind button hover events
        self.predict_button.Bind(wx.EVT_ENTER_WINDOW, self.on_hover)
        self.predict_button.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)

        self.result_display = wx.StaticText(self.panel, label="", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.result_display.SetForegroundColour("#660708")  # Text color for the result
        
        self.grid_sizer.Add(self.predict_button, pos=(row, 0), span=(1, 2), flag=wx.ALIGN_CENTER | wx.ALL, border=10)
        self.grid_sizer.Add(self.result_display, pos=(row+1, 0), span=(1, 2), flag=wx.ALIGN_CENTER | wx.ALL, border=10)
        
        self.panel.SetSizerAndFit(self.grid_sizer)
        
        self.predict_button.Bind(wx.EVT_BUTTON, self.on_predict)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        
        self.rf_model, self.column_names = self.train_model()
        
        # Center the window on the screen
        self.Centre()
    
    def train_model(self):
        # (Training model code remains unchanged)
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

        y_pred = rf_model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        return rf_model, X_train.columns

    def on_predict(self, event):
        try:
            # Validate input fields
            age = self.fields['Age'].GetValue().strip()
            avg_glucose_level = self.fields['Average Glucose Level'].GetValue().strip()
            bmi = self.fields['BMI'].GetValue().strip()
        
            # Check if any field is left empty
            if not age or not avg_glucose_level or not bmi:
                self.result_display.SetForegroundColour("#FF0000")
                self.result_display.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))  # Smaller font for errors
                self.result_display.SetLabel("Error: All fields are required.")
                return
        
            # Convert inputs to float and validate ranges
            age = float(age)
            avg_glucose_level = float(avg_glucose_level)
            bmi = float(bmi)
        
            if not (0 <= age <= 150):
                self.result_display.SetForegroundColour("#FF0000")
                self.result_display.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))  # Smaller font for errors
                self.result_display.SetLabel("Error: Age must be between 0 and 150.")
                self.result_display.Wrap(300)
                self.result_display.SetWindowStyleFlag(wx.ALIGN_CENTER)
                return
        
            if not (0 <= avg_glucose_level <= 1000):
                self.result_display.SetForegroundColour("#FF0000")
                self.result_display.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))  # Smaller font for errors
                self.result_display.SetLabel("Error: Average Glucose Level must be between 0 and 1000.")
                self.result_display.Wrap(300)
                self.result_display.SetWindowStyleFlag(wx.ALIGN_CENTER)
                return
        
            if not (0 <= bmi <= 100):
                self.result_display.SetForegroundColour("#FF0000")
                self.result_display.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))  # Smaller font for errors
                self.result_display.SetLabel("Error: BMI must be between 0 and 100.")
                self.result_display.Wrap(300)
                self.result_display.SetWindowStyleFlag(wx.ALIGN_CENTER)
                return

            input_data = {
                'gender': [self.fields['Gender'].GetSelection()],
                'age': [age],
                'hypertension': [self.fields['Hypertension'].GetSelection()],
                'heart_disease': [self.fields['Heart Disease'].GetSelection()],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'ever_married': [self.fields['Ever Married'].GetSelection()],
                'Residence_type': [self.fields['Residence Type'].GetSelection()],
                'work_type': [self.fields['Work Type'].GetSelection()],
                'smoking_status': [self.fields['Smoking Status'].GetSelection()]
            }
        
            input_df = pd.DataFrame(input_data)
            input_df = input_df.reindex(columns=self.column_names, fill_value=0)
        
            prediction = self.rf_model.predict(input_df)[0]
        
            if prediction == 1:
                result = "High risk of having a stroke."
                self.result_display.SetForegroundColour("#660708")  # Dark red for high risk
            else:
                result = "Low risk of having a stroke."
                self.result_display.SetForegroundColour("#51cb20")  # Green for low risk
        
            self.result_display.SetFont(wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))  # Larger font for result
            self.result_display.SetLabel(result)
        
        except ValueError:
            self.result_display.SetForegroundColour("#FF0000")
            self.result_display.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))  # Smaller font for errors
            self.result_display.SetLabel("Error: Please enter valid numeric values.")
        except Exception as e:
            self.result_display.SetForegroundColour("#FF0000")
            self.result_display.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))  # Smaller font for errors
            self.result_display.SetLabel(f"Error: {e}. Please enter valid input values.")

    def on_close(self, event):
        self.Destroy()

    def on_hover(self, event):
        self.predict_button.SetBackgroundColour("#4a6f41")  # Darker green
        self.Refresh()

    def on_leave(self, event):
        self.predict_button.SetBackgroundColour("#80ED99")  # Original green
        self.Refresh()

if __name__ == "__main__":
    app = wx.App(False)
    frame = StrokeRiskPredictor(None, title="Stroke Risk Prediction")
    frame.Show()
    app.MainLoop()
