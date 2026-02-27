"""FastAPI app: health check, drift report, and predict using models/model.pkl."""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.pkl")
METADATA_PATH = Path("models/metadata.json")
PSI_RETRAIN_THRESHOLD = 0.25
REFERENCE_PATH = Path(os.getenv("DRIFT_REFERENCE_PATH", "data/processed/reference.csv"))
CURRENT_PATH = Path(os.getenv("DRIFT_CURRENT_PATH", "data/processed/current.csv"))

# Load model at import time (best-effort; /predict will guard if missing)
try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception:  # pragma: no cover - defensive
    logger.exception("Failed to load model from %s", MODEL_PATH)
    model = None

# Load feature columns metadata
feature_cols: list[str] = []
if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        cols = meta.get("feature_cols") or []
        feature_cols = list(cols)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to load feature_cols from %s", METADATA_PATH)
        feature_cols = []
else:
    feature_cols = []


app = FastAPI(title="DriftGuard-ML")


class PredictRequest(BaseModel):
    """Request body: list of feature records."""

    data: list[dict[str, float | str]] = Field(
        ..., min_length=1, description="List of feature dicts"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index between expected and actual distributions."""
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = expected_arr[~np.isnan(expected_arr)]
    actual_arr = actual_arr[~np.isnan(actual_arr)]

    if len(expected_arr) == 0 or len(actual_arr) == 0:
        return float("nan")

    # Constant expected: single bin
    if np.ptp(expected_arr) == 0:
        c = float(np.min(expected_arr))
        expected_pct = np.array([1.0])
        in_bin = np.sum(np.abs(actual_arr - c) <= 1e-10) if len(actual_arr) > 0 else 0
        p = in_bin / len(actual_arr) if len(actual_arr) > 0 else 0.0
        actual_pct = np.array([max(1e-10, min(p, 1.0))])
    else:
        _, bin_edges = np.histogram(expected_arr, bins=buckets)
        if np.ptp(bin_edges) == 0:
            expected_pct = np.array([1.0])
            actual_pct = np.array([1.0])
        else:
            expected_counts, _ = np.histogram(expected_arr, bins=bin_edges)
            actual_counts, _ = np.histogram(actual_arr, bins=bin_edges)
            expected_pct = expected_counts / max(expected_counts.sum(), 1)
            actual_pct = actual_counts / max(actual_counts.sum(), 1)
            expected_pct = np.clip(expected_pct, 1e-10, 1.0)
            actual_pct = np.clip(actual_pct, 1e-10, 1.0)

    ratio = np.clip(actual_pct / expected_pct, 1e-10, 1e10)
    return float(np.sum((actual_pct - expected_pct) * np.log(ratio)))


def _ks_test(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Two-sample KS test, returning (statistic, pvalue)."""
    from scipy.stats import ks_2samp

    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = expected_arr[~np.isnan(expected_arr)]
    actual_arr = actual_arr[~np.isnan(actual_arr)]
    if len(expected_arr) == 0 or len(actual_arr) == 0:
        return float("nan"), float("nan")
    stat, pvalue = ks_2samp(expected_arr, actual_arr)
    return float(stat), float(pvalue)


def _detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    """Compute PSI and KS metrics per numeric feature."""
    results: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col]
        cur = current_df[col]
        psi_val = _psi(ref, cur)
        ks_stat, ks_pvalue = _ks_test(ref, cur)
        results[col] = {
            "psi": psi_val,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
        }
    return results


@app.get("/drift_report")
def drift_report() -> dict:
    """Return drift report JSON (max_psi, per_feature, status)."""
    try:
        if not REFERENCE_PATH.exists():
            raise FileNotFoundError(f"Reference file not found: {REFERENCE_PATH}")
        if not CURRENT_PATH.exists():
            raise FileNotFoundError(f"Current file not found: {CURRENT_PATH}")

        reference_df = pd.read_csv(REFERENCE_PATH)
        current_df = pd.read_csv(CURRENT_PATH)

        numeric_cols = [
            c
            for c in reference_df.columns
            if pd.api.types.is_numeric_dtype(reference_df[c]) and c in current_df.columns
        ]
        if not numeric_cols:
            return {"max_psi": None, "per_feature": {}, "status": "no_numeric_columns"}

        per_feature = _detect_drift(reference_df, current_df, numeric_cols)
        psi_values = [m["psi"] for m in per_feature.values() if not (m["psi"] != m["psi"])]
        max_psi = max(psi_values) if psi_values else None
        status = (
            "retrain"
            if max_psi is not None and max_psi >= PSI_RETRAIN_THRESHOLD
            else "ok"
        )
        return {"max_psi": max_psi, "per_feature": per_feature, "status": status}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to compute drift report")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute drift report: {e}",
        ) from e


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded; ensure models/model.pkl exists.",
        )
    if not feature_cols:
        raise HTTPException(
            status_code=503,
            detail="feature_cols not available; ensure models/metadata.json contains feature_cols.",
        )

    df = pd.DataFrame(request.data)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_columns": sorted(missing)},
        )

    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    predictions = (proba >= 0.5).astype(int).tolist()
    return {"predictions": predictions, "probabilities": proba.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
