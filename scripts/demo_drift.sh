#!/usr/bin/env bash
set -euo pipefail

CURRENT_CSV="data/processed/current.csv"
BACKUP_CSV="data/processed/current.backup.csv"

if [[ ! -f "$CURRENT_CSV" ]]; then
  echo "Error: $CURRENT_CSV not found. Run 'make data-prep' (or ensure data/processed/current.csv exists) first." >&2
  exit 1
fi

echo "== backing up current.csv =="
cp "$CURRENT_CSV" "$BACKUP_CSV"

trap 'echo "== restoring backup =="; mv -f "$BACKUP_CSV" "$CURRENT_CSV" || true' EXIT

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

echo "== pre-retrain drift check (may return non-zero) =="
python src/check_drift.py || true

echo "== retraining model =="
make train

echo "== restoring backup =="
mv -f "$BACKUP_CSV" "$CURRENT_CSV" || true
trap - EXIT

echo "== post-retrain drift check (may return non-zero) =="
python src/check_drift.py || true

echo "== demo-drift done =="
