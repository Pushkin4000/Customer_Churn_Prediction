"""Prove the numpy bundle predicts identically to the original XGBoost pipeline.

Runs the original path (pandas + MinMaxScaler + XGBClassifier.predict_proba)
and the shipped path (app.main) over the full domain of every UI slider, plus
boundary and adversarial cases, and compares probabilities.

    python tools/verify_compiled_model.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Domain of each what-if slider, taken from SLIDER_CONFIG in main.html.
RANGES = {
    "monthly_logins":     (0, 100),
    "email_open_rate":    (0, 1),
    "avg_session_time":   (0, 120),
    "tenure_months":      (0, 120),
    "total_revenue":      (0, 20000),
    "last_login_days_ago": (0, 365),
    "csat_score":         (1, 5),
}
KEYS = list(RANGES)


def original_predict(payload, art):
    """The pre-compilation inference path, reproduced verbatim."""
    model, scaler, baseline, features = art
    df = pd.DataFrame([baseline.values], columns=features)
    for col, value in payload.items():
        if col in df.columns:
            df.at[0, col] = value
    cols = scaler.feature_names_in_
    df[cols] = scaler.transform(df[cols])
    df = df[features]
    return float(model.predict_proba(df)[0][1])


def main():
    art = (
        joblib.load(ROOT / "app" / "model.pkl"),
        joblib.load(ROOT / "app" / "scaler.pkl"),
        joblib.load(ROOT / "app" / "baseline.pkl"),
        list(joblib.load(ROOT / "app" / "feature_names.pkl")),
    )

    from app.main import _load, _margin
    import math
    m = _load()

    def compiled_predict(payload):
        x = m["baseline"].copy()
        for col, value in payload.items():
            i = m["exposed"].get(col)
            if i is not None:
                x[i] = value
        si = m["scale_idx"]
        x[si] = x[si] * m["scale_mul"] + m["scale_add"]
        return 1.0 / (1.0 + math.exp(-_margin(m, x)))

    cases = []
    # 1. Defaults.
    cases.append({"monthly_logins": 12, "email_open_rate": 0.35, "avg_session_time": 18,
                  "tenure_months": 24, "total_revenue": 4500, "last_login_days_ago": 7,
                  "csat_score": 3.5})
    # 2. Every corner of the input box (all-min, all-max, and one-hot extremes).
    cases.append({k: RANGES[k][0] for k in KEYS})
    cases.append({k: RANGES[k][1] for k in KEYS})
    for k in KEYS:
        for bound in RANGES[k]:
            c = {j: (RANGES[j][0] + RANGES[j][1]) / 2 for j in KEYS}
            c[k] = bound
            cases.append(c)
    # 3. Dense random sweep of the whole domain.
    rng = np.random.default_rng(1234)
    for _ in range(4000):
        cases.append({k: float(rng.uniform(*RANGES[k])) for k in KEYS})
    # 4. Values outside the sliders, to probe split boundaries the UI can't reach.
    for _ in range(1000):
        cases.append({k: float(rng.uniform(-50, RANGES[k][1] * 2 + 10)) for k in KEYS})

    # XGBoost accumulates in float32, so the two paths agree to ~1e-7 rather
    # than bit-for-bit. A probability sitting within that margin of a 4-decimal
    # midpoint can round to either neighbouring bucket; that is a 0.01pp display
    # tie-break, not a disagreement about the model. Anything wider is a bug.
    TOL = 1e-6
    diffs, tie_breaks, real_round_diffs, verdict_mismatch = [], 0, 0, 0
    thr = m["threshold"]
    for c in cases:
        a = original_predict(c, art)
        b = compiled_predict(c)
        d = abs(a - b)
        diffs.append(d)
        if round(a, 4) != round(b, 4):
            adjacent = abs(round(a, 4) - round(b, 4)) <= 1e-4 + 1e-12
            if d < TOL and adjacent:
                tie_breaks += 1
            else:
                real_round_diffs += 1
                print(f"  UNEXPECTED: orig={a!r} new={b!r} diff={d:.2e}")
        if (a > thr) != (b > thr):
            verdict_mismatch += 1
            print(f"  VERDICT FLIP: orig={a!r} new={b!r} threshold={thr}")

    diffs = np.array(diffs)
    print(f"cases compared             : {len(cases)}")
    print(f"max abs probability diff   : {diffs.max():.3e}  (tolerance {TOL:.0e})")
    print(f"mean abs diff              : {diffs.mean():.3e}")
    print(f"churn verdict flips        : {verdict_mismatch}")
    print(f"4-decimal tie-breaks       : {tie_breaks}  (float32 noise at a midpoint)")
    print(f"unexplained rounding diffs : {real_round_diffs}")

    ok = diffs.max() < TOL and verdict_mismatch == 0 and real_round_diffs == 0
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
