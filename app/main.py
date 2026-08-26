import math
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
BUNDLE = ARTIFACT_DIR / "model_compiled.npz"

# The XGBoost model is 66 trees of max depth 4. Importing xgboost to evaluate
# that costs ~2.3s of cold start and pulls in pandas, scikit-learn and scipy;
# tools/compile_model.py flattens the same trees into plain arrays so this
# function needs nothing heavier than numpy. Regenerate after retraining.
_M = None
_load_error = None


def _load():
    global _M, _load_error
    if _M is not None:
        return _M
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        z = np.load(BUNDLE, allow_pickle=True)
        features = [str(f) for f in z["features"]]
        _M = {
            "baseline": z["baseline"],
            "features": features,
            # Column index for each what-if field the client may send.
            "exposed": {str(name): features.index(str(name)) for name in z["exposed"]},
            "scale_idx": z["scale_idx"],
            "scale_mul": z["scale_mul"],
            "scale_add": z["scale_add"],
            "threshold": float(z["threshold"]),
            "intercept": float(z["intercept"]),
            "feat": z["feat"], "thresh": z["thresh"],
            "left": z["left"], "right": z["right"],
            "miss": z["miss"], "leaf": z["leaf"],
        }
        _M["tree_ix"] = np.arange(_M["feat"].shape[0])
        return _M
    except Exception as exc:
        _load_error = f"artifact load failed: {exc}"
        raise RuntimeError(_load_error) from exc


def _margin(m, x):
    """Walk every tree at once; depth is bounded so this is a handful of steps."""
    ix = m["tree_ix"]
    node = np.zeros(ix.shape[0], dtype=np.int64)
    feat, thresh, left, right, miss = m["feat"], m["thresh"], m["left"], m["right"], m["miss"]

    for _ in range(feat.shape[1]):          # generous upper bound on depth
        f = feat[ix, node]
        active = f >= 0
        if not active.any():
            break
        vals = x[np.where(active, f, 0)]    # dummy index 0 where inactive
        nxt = np.where(vals < thresh[ix, node], left[ix, node], right[ix, node])
        nxt = np.where(np.isnan(vals), miss[ix, node], nxt)
        node = np.where(active, nxt, node)

    return m["intercept"] + float(m["leaf"][ix, node].sum())


class PredictionInput(BaseModel):
    monthly_logins: float
    email_open_rate: float
    avg_session_time: float
    tenure_months: float
    total_revenue: float
    last_login_days_ago: float
    csat_score: float


# Warm the bundle at import. It is ~16 KB, so this costs almost nothing and
# guarantees the function is ready the moment it can answer at all.
try:
    _load()
except Exception:
    _load_error = None  # let a real request retry rather than caching a boot failure


@app.get("/")
def home():
    return {"message": "XGBoost Churn API is active", "model_ready": _M is not None}


@app.get("/health")
def health():
    """Readiness probe: 200 only once the model is actually loaded."""
    try:
        _load()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}")
    return {"status": "ok", "model_ready": True}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        m = _load()

        # 1. Start from the baseline (mean training profile), which fills every
        #    column the UI does not expose - city dummies and the like.
        x = m["baseline"].copy()

        # 2. Overwrite the what-if features with the user's input.
        for col, value in data.model_dump().items():
            idx = m["exposed"].get(col)
            if idx is not None:
                x[idx] = value

        # 3. Apply the MinMaxScaler to exactly the columns it was fitted on.
        #    MinMaxScaler is affine, so this is the whole of its transform.
        si = m["scale_idx"]
        x[si] = x[si] * m["scale_mul"] + m["scale_add"]

        # 4. Evaluate the trees; column order is baked into the bundle.
        prob = 1.0 / (1.0 + math.exp(-_margin(m, x)))
        prediction = 1 if prob > m["threshold"] else 0

        return {
            "churn_probability": float(round(prob, 4)),
            "churn_prediction": prediction,
            "threshold_used": float(m["threshold"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
