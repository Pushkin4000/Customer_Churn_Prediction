from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Churn What-If API")

# Load all artifacts
model = joblib.load("app/model.pkl")
threshold = joblib.load("app/threshold.pkl")
features = joblib.load("app/feature_names.pkl")  # This contains all columns (including dummies)
baseline = joblib.load("app/baseline.pkl")      # This is x_train.mean()
scaler = joblib.load("app/scaler.pkl")          # Fitted on specific numerical columns

# Define the user-facing features for the What-If tool
class PredictionInput(BaseModel):
    monthly_logins: float
    email_open_rate: float
    avg_session_time: float
    tenure_months: float
    total_revenue: float
    last_login_days_ago: float
    csat_score: float

@app.get("/")
def home():
    return {"message": "XGBoost Churn API is active"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # 1. Start with the baseline (mean of training data)
        # This fills in all the dummy columns (city_Berlin, etc.) with their average frequencies
        input_df = pd.DataFrame([baseline.values], columns=features)

        # 2. Update the 'What-If' features with user input
        user_input_dict = data.model_dump()
        for col, value in user_input_dict.items():
            if col in input_df.columns:
                input_df.at[0, col] = value

        # 3. Handle Scaling
        # According to your notebook, you only scaled specific columns (cols_to_scale).
        # We use 'scaler.feature_names_in_' to know exactly which ones to transform.
        cols_to_scale = scaler.feature_names_in_
        
        # Scale only the required numerical columns
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        # 4. Final Alignment
        # Ensure the column order is EXACTLY what XGBoost saw during training
        input_df = input_df[features]

        # 5. Prediction
        # XGBoost often prefers DMatrix or clean numpy arrays/DataFrames
        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > threshold else 0

        return {
            "churn_probability": float(round(prob, 4)),
            "churn_prediction": prediction,
            "threshold_used": float(threshold)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")