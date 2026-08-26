import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Churn What-If API")

# Allow browser clients hosted on a different origin to call this API.
# Set CORS_ALLOW_ORIGINS as a comma-separated list in production.
cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
if not cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_DIR = Path(__file__).resolve().parent
_artifacts = None
_artifacts_error = None

def _load_artifacts():
    global _artifacts, _artifacts_error
    if _artifacts is not None:
        return _artifacts
    if _artifacts_error is not None:
        raise RuntimeError(_artifacts_error)

    try:
        _artifacts = {
            "model": joblib.load(ARTIFACT_DIR / "model.pkl"),
            "threshold": joblib.load(ARTIFACT_DIR / "threshold.pkl"),
            "features": joblib.load(ARTIFACT_DIR / "feature_names.pkl"),
            "baseline": joblib.load(ARTIFACT_DIR / "baseline.pkl"),
            "scaler": joblib.load(ARTIFACT_DIR / "scaler.pkl"),
        }
        return _artifacts
    except Exception as exc:
        _artifacts_error = f"artifact load failed: {exc}"
        raise RuntimeError(_artifacts_error) from exc

# Define the user-facing features for the What-If tool
class PredictionInput(BaseModel):
    monthly_logins: float
    email_open_rate: float
    avg_session_time: float
    tenure_months: float
    total_revenue: float
    last_login_days_ago: float
    csat_score: float

# Warm the model at import time. On Vercel the whole module is imported during
# the cold start of the serverless function, so paying the joblib load here
# means the very first request that reaches a handler is already served warm.
try:
    _load_artifacts()
except Exception:  # pragma: no cover - surfaced per-request by /health, /predict
    # Don't let a boot-time failure poison the cache; let a real request retry.
    _artifacts_error = None


@app.get("/")
def home():
    return {"message": "XGBoost Churn API is active", "model_ready": _artifacts is not None}

@app.get("/health")
def health():
    """Readiness probe.

    Returns 200 only once the model artifacts are actually loaded, so a client
    that gets "ok" here can immediately POST /predict without a second cold
    start. On a serverless cold start this request is what pays the boot cost.
    """
    try:
        _load_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}")
    return {"status": "ok", "model_ready": True}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        artifacts = _load_artifacts()
        model = artifacts["model"]
        threshold = artifacts["threshold"]
        features = artifacts["features"]
        baseline = artifacts["baseline"]
        scaler = artifacts["scaler"]

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

