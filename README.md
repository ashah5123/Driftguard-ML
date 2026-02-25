# DriftGuard-ML

**DriftGuard-ML** is a production-style ML system that detects data drift and automates retraining. It runs a validation → drift-check → retrain → commit → serve loop entirely on GitHub Actions (and locally), with no paid services: sklearn/XGBoost pipelines, PSI + KS drift metrics, Great Expectations data checks, FastAPI serving, and versioned model artifacts in git.

---

## Key features

- **Data quality gate** — Great Expectations–backed checks on `current.csv` (required columns, binary target, no all-null columns, numeric diversity) before any drift or retrain step.
- **Drift detection** — Population Stability Index (PSI) and Kolmogorov–Smirnov tests on numeric features; configurable PSI threshold (default **0.25**) to trigger retraining.
- **Training pipeline** — ColumnTransformer (numeric + categorical), XGBoost classifier, 5-fold stratified CV, MLflow logging, joblib model + JSON metadata under `models/`.
- **Prediction API** — FastAPI with `/health` and `/predict` (list-of-records → probabilities), Pydantic validation, uvicorn; Dockerfile for containerized serve.
- **CI/CD** — Separate runtime vs dev dependencies; GitHub Actions for tests (CI) and scheduled validate → drift workflow (retrain gate).

---

## Architecture (ASCII)

```
  data/raw/          data/processed/              src/                    models/
  ---------          ----------------             ----                    -------
  (immutable)   →    reference.csv    ──┐
  (git-ignored)      current.csv      ──┼──► validate_data.py ──► check_drift.py
                                        │         (GE + checks)       (PSI/KS)
                                        │              │                    │
                                        │              ▼                    ▼
                                        │         pass? ──► train.py ──► model.pkl
                                        │                   (sklearn+XGB)   metadata.json
                                        │                        │              │
                                        └────────────────────────┼──────────────┘
                                                                 ▼
                                                        serve.py (FastAPI)
                                                                 │
                                                                 ▼
                                                        POST /predict → probabilities
```

---

## Auto-retrain loop

1. **Validate** — `validate_data.py` checks `data/processed/current.csv`: required columns (from reference or default), binary target, no all-null columns, numeric columns with ≥2 distinct values. Fails → workflow stops; no drift check or retrain.
2. **Drift** — `check_drift.py` compares `current.csv` to `reference.csv` (numeric columns only). Computes PSI and KS per feature; if **max_psi ≥ 0.25** → exit 2 (retrain trigger).
3. **Retrain** — Training runs on `current.csv` (e.g. `train.py`): preprocessing, XGBoost, CV AUC logged to MLflow, full-data fit, save `models/model.pkl` and `models/metadata.json`.
4. **Commit** — Updated `models/` can be committed (or PR’d) so model versions are tied to git history.
5. **Serve** — FastAPI loads `models/model.pkl`; `/predict` returns `predict_proba[:, 1]` for incoming records. Same artifact can be baked into the Docker image.

---

## PSI thresholds

**PSI (Population Stability Index)** measures distribution shift between reference and current:  
`PSI = Σ (current_pct - reference_pct) × ln(current_pct / reference_pct)` over bins.  
Larger values mean more shift. DriftGuard uses:

- **max_psi < 0.25** → no action (exit 0).
- **max_psi ≥ 0.25** → trigger retrain (exit 2).  
  This threshold is configurable in `src/check_drift.py` (`PSI_RETRAIN_THRESHOLD`). Common guidance: &lt; 0.1 stable, 0.1–0.25 moderate, ≥ 0.25 significant.

---

## Makefile commands

Run `make setup` once (creates `.venv`, installs `requirements.txt` + `requirements-dev.txt`). All other targets use the venv.

| Command       | Description |
|---------------|-------------|
| `make setup`  | Create venv and install runtime + dev deps |
| `make drift`  | Run drift check (`src/check_drift.py`) |
| `make train`  | Train on `data/processed/current.csv`, target `target`, MLflow `local_run` |
| `make serve`  | Start FastAPI with uvicorn (reload, 0.0.0.0:8000) |
| `make test`   | Pytest (quiet) |
| `make lint`   | Ruff check |
| `make format` | Black format |

---

## GitHub Actions workflows

| Workflow   | Trigger              | Steps |
|------------|----------------------|--------|
| **CI**     | Push/PR to main      | Install deps (runtime + dev), run `pytest tests/` |
| **Retrain**| Weekly cron + manual | Install runtime deps → `validate_data.py` → `check_drift.py`. Fails on validation or uses drift exit code; retrain/commit steps can be added downstream. |

---

## Quick start

```bash
make setup
make train    # needs data/processed/current.csv with target column
make serve   # then: curl http://localhost:8000/health && curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"data":[{"feature_a":1,"feature_b":2,"category":"A"}]}'
```

Docker: `docker build -t driftguard-ml . && docker run -p 8000:8000 driftguard-ml` (requires `models/model.pkl` in the build context).

---

## Roadmap

- **Retrain job in Actions** — Add a workflow step that runs `train.py` and commits (or opens a PR for) `models/` when drift triggers.
- **Reference snapshot from last train** — Auto-update `reference.csv` from the dataset used at last successful training so drift is always vs last production data.
- **Alerting on drift** — Optional step to post workflow summary (e.g. PSI report) to Slack/Discord or issue a GitHub issue when retrain is triggered.
- **Schema-enforced predict** — Pydantic model generated from training feature list so `/predict` rejects out-of-schema payloads early.
