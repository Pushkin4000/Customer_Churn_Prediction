# Customer Churn Prediction

Interactive churn-risk prediction app with a FastAPI backend and a browser-based what-if simulator.

## Overview

This project lets you simulate customer behavior changes (logins, session time, CSAT, etc.) and instantly estimate churn probability using a trained XGBoost model.

- Backend: FastAPI inference API
- Frontend: single-page `main.html` dashboard
- Model artifacts: pre-trained `.pkl` files in `app/`
- Deploy target: Vercel (`vercel.json` included)

## Features

- Real-time what-if analysis with 7 high-impact customer features
- Probability + binary churn prediction using a saved decision threshold
- Baseline feature completion for non-exposed model columns
- Readiness endpoint plus frontend warm-up retries, so a serverless cold start never surfaces as a connection error
- Configurable CORS via environment variable

## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI, Pydantic |
| ML Inference | XGBoost, scikit-learn, pandas, joblib |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Vercel (`@vercel/python`, `@vercel/static`) |

## Project Structure

```text
Customer_Churn_Prediction/
|- app/
|  |- main.py
|  |- model.pkl
|  |- threshold.pkl
|  |- feature_names.pkl
|  |- baseline.pkl
|  |- scaler.pkl
|- main.html
|- requirements.txt
|- vercel.json
|- LICENSE
```

## Quickstart (Local)

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install uvicorn
```

### 3. Run the API

```powershell
uvicorn app.main:app --reload
```

API will be available at `http://127.0.0.1:8000`.

### 4. Open the frontend

Option A:
- Open `main.html` directly in your browser.

Option B:
- Serve the project as static files:

```powershell
python -m http.server 5500
```

Then open `http://127.0.0.1:5500/main.html`.

The UI auto-fills API endpoint as `http://127.0.0.1:8000` on localhost.

## API Reference

### `GET /`

Returns basic service status.

### `GET /health`

Readiness check. Returns `200` only once the model artifacts are loaded, and
`503` while they are not, so the frontend can use it to wait out a serverless
cold start before sending a prediction.

Response:

```json
{ "status": "ok", "model_ready": true }
```

### `POST /predict`

Predict churn probability from user behavior inputs.

Request body:

```json
{
  "monthly_logins": 12,
  "email_open_rate": 0.35,
  "avg_session_time": 18,
  "tenure_months": 24,
  "total_revenue": 4500,
  "last_login_days_ago": 7,
  "csat_score": 3.5
}
```

Response:

```json
{
  "churn_probability": 0.2143,
  "churn_prediction": 0,
  "threshold_used": 0.5
}
```

## Inference Flow

For each request, the API:

1. Loads model artifacts (`model`, `threshold`, `features`, `baseline`, `scaler`)
2. Starts from baseline (mean training profile)
3. Overwrites exposed what-if features with user inputs
4. Scales only columns expected by the saved scaler
5. Aligns final feature order to training-time columns
6. Runs `predict_proba` and applies stored threshold

## Configuration

### Environment Variables

- `CORS_ALLOW_ORIGINS`
  - Comma-separated list of allowed origins
  - Default: `*`
  - Example:

```powershell
$env:CORS_ALLOW_ORIGINS="https://your-frontend-domain.com,https://staging.example.com"
```

## Deployment (Vercel)

`vercel.json` is already configured to:

- Route `/predict` and `/health` to `app/main.py`
- Serve `main.html` at `/`

Deploy with Vercel CLI:

```powershell
vercel
vercel --prod
```

## License

This project is licensed under the MIT License. See `LICENSE`.
