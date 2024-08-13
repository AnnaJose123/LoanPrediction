import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Load the dataset
train = pd.read_csv(r"D:\Pgm3\train_ctrUa4K.csv")

# Fill missing values
train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0])
train['Self_Employed'] = train['Self_Employed'].fillna(train['Self_Employed'].mode()[0])
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].median())
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].median())
train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].median())

# Check for any remaining NaNs
if train.isnull().sum().sum() > 0:
    raise ValueError("There are still NaNs in the dataset.")

# Handle outliers in CoapplicantIncome
for column in ['CoapplicantIncome']:
    Q1 = train[column].quantile(0.25)
    Q3 = train[column].quantile(0.75)
    iqr = Q3 - Q1
    lower = Q1 - 1.5 * iqr
    upper = Q3 + 1.5 * iqr
    median = train[column].median()
    train[column] = np.where((train[column] < lower) | (train[column] > upper), median, train[column])

# Drop Loan_ID before scaling and training
X = train.drop(columns=['Loan_Status', 'Loan_ID'], axis=1)
y = train['Loan_Status']

# Label encoding
label_encoders = {}

for column in ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)


with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
# Model training
model = LogisticRegression()
model.fit(X_scaled, y_encoded)

# Save the model and preprocessing tools
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and preprocessing tools saved successfully!")
