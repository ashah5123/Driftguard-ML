#!/usr/bin/env bash
set -euo pipefail

BASELINE="data/processed/current_baseline.csv"
CURRENT="data/processed/current.csv"

if [[ ! -f "$CURRENT" ]]; then
  echo "Error: $CURRENT not found. Run 'make data-prep' (or ensure data/processed/current.csv exists) first." >&2
  exit 1
fi

if [[ ! -f "$BASELINE" ]]; then
  cp "$CURRENT" "$BASELINE"
fi
echo "== baseline ensured =="
cp "$BASELINE" "$CURRENT"

echo "== injecting drift =="
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed/current.csv")
if "distance" in df.columns:
    df["distance"] = df["distance"] * 1.8
if "dep_hour" in df.columns:
    df["dep_hour"] = (df["dep_hour"] + 6).clip(0, 23)
df.to_csv("data/processed/current.csv", index=False)
print("Drift injected. Rows:", len(df))
PY

echo "== pre-check =="
python src/check_drift.py || true

echo "== retrain =="
make train

echo "== restoring baseline =="
cp "$BASELINE" "$CURRENT"

echo "== post-check =="
python src/check_drift.py || true

echo "== done =="
