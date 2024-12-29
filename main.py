from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load the encoder, scaler, and model
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model_xgboost.pkl')

# Define the expected input data format using Pydantic
class LoanApplication(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str

# Prediction endpoint
@app.post("/predict")
def predict_loan_status(application: LoanApplication):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([application.dict()])

    # Define categorical columns for encoding
    categorical_columns = [
        'person_education', 
        'person_home_ownership', 
        'loan_intent', 
        'person_gender', 
        'previous_loan_defaults_on_file'
    ]
    
    # Apply one-hot encoding to categorical columns
    encoded_data = encoder.transform(input_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns), index=input_data.index)
    
    # Combine with non-categorical columns
    numerical_data = input_data.drop(columns=categorical_columns)
    combined_data = pd.concat([numerical_data, encoded_df], axis=1)
    
    # Apply scaling
    scaled_data = scaler.transform(combined_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]  # 0 or 1 for loan_status

    # Return prediction as JSON
    return {"loan_status": int(prediction)}  # Cast to int for JSON compatibility
