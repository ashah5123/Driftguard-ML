"""CLI: load reference/current data, run drift detection, print JSON report and exit with 0/1/2."""

import json
import sys
from pathlib import Path

import pandas as pd

from data_preprocessing import load_data
from drift import detect_drift

REFERENCE_PATH = Path("data/processed/reference.csv")
CURRENT_PATH = Path("data/processed/current.csv")
PSI_RETRAIN_THRESHOLD = 0.25


def main() -> int:
    print("DriftGuard drift check")
    print("  Reference:", REFERENCE_PATH)
    print("  Current: ", CURRENT_PATH)

    if not REFERENCE_PATH.exists():
        print("[ERROR] Reference file not found:", REFERENCE_PATH)
        return 1
    if not CURRENT_PATH.exists():
        print("[ERROR] Current file not found:", CURRENT_PATH)
        return 1

    try:
        reference_df = load_data(REFERENCE_PATH)
        current_df = load_data(CURRENT_PATH)
    except Exception as e:
        print("[ERROR] Failed to load data:", e)
        return 1

    numeric_cols = [
        c for c in reference_df.columns
        if pd.api.types.is_numeric_dtype(reference_df[c])
        and c in current_df.columns
    ]
    if not numeric_cols:
        print("[WARN] No numeric columns in common; nothing to check.")
        report = {"max_psi": None, "per_feature": {}, "status": "no_numeric_columns"}
        print(json.dumps(report, indent=2))
        return 0

    print("  Numeric columns:", len(numeric_cols))

    per_feature = detect_drift(reference_df, current_df, numeric_cols)
    psi_values = [m["psi"] for m in per_feature.values() if not (m["psi"] != m["psi"])]
    max_psi = max(psi_values) if psi_values else None

    report = {
        "max_psi": max_psi,
        "per_feature": per_feature,
        "status": "retrain" if max_psi is not None and max_psi >= PSI_RETRAIN_THRESHOLD else "ok",
    }
    print("\n--- Drift report (JSON) ---")
    print(json.dumps(report, indent=2))

    if max_psi is None:
        print("\n[OK] No numeric features; no drift decision.")
        return 0
    if max_psi >= PSI_RETRAIN_THRESHOLD:
        print("\n[RETRAIN] max_psi =", round(max_psi, 4), ">=", PSI_RETRAIN_THRESHOLD, "-> trigger retraining.")
        return 2
    print("\n[OK] max_psi =", round(max_psi, 4), "<", PSI_RETRAIN_THRESHOLD, "-> no action.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
