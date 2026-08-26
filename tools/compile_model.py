"""Compile the trained XGBoost artifacts into a numpy-only bundle.

Runtime inference used to `import xgboost`, which costs ~2.3s of cold start and
transitively drags in pandas, scikit-learn and scipy. The model itself is 66
trees of max depth 4 - a few hundred float comparisons. This script flattens it
into plain arrays so the serverless function needs nothing but numpy.

Dev-time only: this is the one place that still needs the full ML stack. Run it
after retraining, then commit the regenerated app/model_compiled.npz.

    python tools/compile_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
OUT = APP / "model_compiled.npz"

# The 7 what-if features the UI exposes; everything else comes from baseline.
EXPOSED = [
    "monthly_logins", "email_open_rate", "avg_session_time", "tenure_months",
    "total_revenue", "last_login_days_ago", "csat_score",
]


def flatten_trees(booster, feature_names, n_trees_used):
    """Turn the booster's JSON dump into padded [n_trees, max_nodes] arrays.

    Only the first `n_trees_used` trees are kept. The model was trained with
    early stopping (best_iteration=15 of n_estimators=1000), so predict_proba
    evaluates iteration_range=(0, best_iteration + 1) while the dump contains
    every tree that was ever built. Summing all of them silently changes the
    prediction.
    """
    col = {name: i for i, name in enumerate(feature_names)}
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json")][:n_trees_used]

    parsed = []
    for tree in trees:
        nodes = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for child in n.get("children", []):
                walk(child)

        walk(tree)
        parsed.append(nodes)

    max_nodes = max(max(n) for n in parsed) + 1
    n_trees = len(parsed)

    feat = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    thresh = np.zeros((n_trees, max_nodes), dtype=np.float64)
    left = np.zeros((n_trees, max_nodes), dtype=np.int32)
    right = np.zeros((n_trees, max_nodes), dtype=np.int32)
    miss = np.zeros((n_trees, max_nodes), dtype=np.int32)
    leaf = np.zeros((n_trees, max_nodes), dtype=np.float64)

    for t, nodes in enumerate(parsed):
        for nid, n in nodes.items():
            if "leaf" in n:
                leaf[t, nid] = n["leaf"]
                continue
            feat[t, nid] = col[n["split"]]
            thresh[t, nid] = n["split_condition"]
            left[t, nid] = n["yes"]
            right[t, nid] = n["no"]
            miss[t, nid] = n["missing"]

    return dict(feat=feat, thresh=thresh, left=left, right=right, miss=miss, leaf=leaf)


def main():
    model = joblib.load(APP / "model.pkl")
    scaler = joblib.load(APP / "scaler.pkl")
    baseline = joblib.load(APP / "baseline.pkl")
    features = list(joblib.load(APP / "feature_names.pkl"))
    threshold = float(joblib.load(APP / "threshold.pkl"))

    booster = model.get_booster()

    # Match whatever predict_proba would use, tree for tree.
    best = getattr(model, "best_iteration", None)
    n_parallel = int(json.loads(booster.save_config())["learner"]["gradient_booster"]
                     ["gbtree_model_param"]["num_parallel_tree"])
    total = len(booster.get_dump())
    n_trees_used = total if best is None else (int(best) + 1) * n_parallel
    trees = flatten_trees(booster, features, n_trees_used)
    print(f"  trees built={total}  trees used by predict_proba={n_trees_used}")

    # MinMaxScaler is just an affine map: x * scale_ + min_.
    scale_cols = list(scaler.feature_names_in_)
    scale_idx = np.array([features.index(c) for c in scale_cols], dtype=np.int32)

    # Derive the margin intercept empirically. It must be calibrated against
    # model.predict_proba - the call the API actually made - because that
    # applies a base score that booster.inplace_predict(predict_type="margin")
    # leaves out. base_score semantics have also shifted between XGBoost
    # versions, so measuring beats trusting the config. Constancy is asserted.
    import pandas as pd
    rng = np.random.default_rng(0)
    probe = pd.DataFrame(
        rng.normal(size=(256, len(features))) * baseline.values.std() + baseline.values,
        columns=features,
    )
    proba = model.predict_proba(probe)[:, 1].astype(np.float64)
    assert proba.min() > 1e-9 and proba.max() < 1 - 1e-9, "probe saturated; widen the spread"
    logit = np.log(proba / (1.0 - proba))

    arr = probe.to_numpy()
    raw_leaves = np.array([_sum_leaves(trees, arr[i]) for i in range(len(probe))])

    intercept = logit - raw_leaves
    # XGBoost accumulates in float32, so expect ~1e-6 jitter here. Correctness
    # is asserted properly by verify_compiled_model.py, which compares final
    # probabilities against model.predict_proba over the full input domain.
    assert intercept.std() < 1e-4, f"intercept not constant: std={intercept.std()}"
    intercept = float(intercept.mean())

    np.savez_compressed(
        OUT,
        baseline=baseline.values.astype(np.float64),
        features=np.array(features, dtype=object),
        exposed=np.array(EXPOSED, dtype=object),
        scale_idx=scale_idx,
        scale_mul=scaler.scale_.astype(np.float64),
        scale_add=scaler.min_.astype(np.float64),
        threshold=np.float64(threshold),
        intercept=np.float64(intercept),
        **trees,
    )
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"  trees={trees['feat'].shape[0]}  max_nodes={trees['feat'].shape[1]}")
    print(f"  intercept={intercept:.12g}  threshold={threshold:.12g}")


def _sum_leaves(trees, x):
    """Reference (slow, scalar) tree walk used only for compile-time checks."""
    total = 0.0
    for t in range(trees["feat"].shape[0]):
        node = 0
        while trees["feat"][t, node] >= 0:
            f = trees["feat"][t, node]
            v = x[f]
            if np.isnan(v):
                node = trees["miss"][t, node]
            elif v < trees["thresh"][t, node]:
                node = trees["left"][t, node]
            else:
                node = trees["right"][t, node]
        total += trees["leaf"][t, node]
    return total


if __name__ == "__main__":
    main()
