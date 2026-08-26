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
- XGBoost compiled away for inference: ~2.8s cold start down to ~0.7s
- Configurable CORS via environment variable

## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI, Pydantic |
| ML Inference | numpy (model compiled from XGBoost - see below) |
| Training / tooling | XGBoost, scikit-learn, pandas, joblib (dev only) |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Vercel (`@vercel/python`, `@vercel/static`) |

## Project Structure

```text
Customer_Churn_Prediction/
|- app/
|  |- main.py
|  |- model_compiled.npz   <- what the API actually loads
|  |- model.pkl            <- training artifacts, kept as source of truth
|  |- threshold.pkl
|  |- feature_names.pkl
|  |- baseline.pkl
|  |- scaler.pkl
|- tools/
|  |- compile_model.py
|  |- verify_compiled_model.py
|- main.html
|- requirements.txt
|- requirements-dev.txt
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

To retrain or recompile the model you also need the dev extras:

```powershell
pip install -r requirements-dev.txt
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

1. Loads the compiled bundle (`app/model_compiled.npz`)
2. Starts from baseline (mean training profile)
3. Overwrites exposed what-if features with user inputs
4. Applies the MinMaxScaler's affine transform to the columns it was fitted on
5. Walks the decision trees over the training-time column order
6. Applies the sigmoid and the stored threshold

## Compiled Model

The serverless function does **not** import XGBoost. That import alone costs
~2.3s of cold start and transitively pulls in pandas, scikit-learn and scipy,
which is a lot of machinery for a model that is 16 trees of max depth 4.

`tools/compile_model.py` flattens the trained booster into plain numpy arrays
(`app/model_compiled.npz`, ~7 KB), so the runtime needs only `fastapi` and
`numpy`. Measured cold start went from ~2.8s to ~0.7s.

Two details the compiler has to get right, both covered by the verifier:

- The model was trained with early stopping (`best_iteration=15` of
  `n_estimators=1000`), so `predict_proba` evaluates only the first 16 trees
  while the booster dump contains all 66. Summing all of them changes the
  prediction.
- The margin intercept is calibrated against `predict_proba`, not against
  `booster.inplace_predict(predict_type="margin")` - the latter omits the base
  score that `predict_proba` applies.

### Regenerating after retraining

```powershell
pip install -r requirements-dev.txt
python tools/compile_model.py
python tools/verify_compiled_model.py
```

`verify_compiled_model.py` compares the compiled path against the original
pandas + scikit-learn + XGBoost pipeline over 5,000+ inputs spanning every
slider's full range plus out-of-range probes. It requires zero churn-verdict
flips and agreement to within 1e-6; the two paths currently agree to 6e-8
(XGBoost accumulates in float32, so they are not bit-identical).

The original `.pkl` artifacts stay in `app/` as the source of truth - they are
simply no longer loaded at runtime.

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
