## DriftGuard-ML

DriftGuard-ML is a **self-monitoring machine learning system** focused on **data/model drift detection** and **automated retraining**, designed to be production-ready while remaining fully free and self-hosted.

It assumes you already have a model in production and provides the scaffolding to:
- track data and prediction distributions over time,
- detect when behavior deviates from expectations,
- trigger retraining pipelines on a schedule via GitHub Actions,
- version and commit trained model artifacts in `models/`.

---

## Features

- **Drift detection hooks**: structure for tracking feature and prediction statistics so you can implement drift checks that fit your domain.
- **Automated retraining pipeline**: scheduled GitHub Actions workflow (cron-triggered) can run your retraining script and update committed model artifacts.
- **Committed models**: `models/` is intentionally **not** git-ignored so you can version control trained models and metadata.
- **Clean project layout**: separation of notebooks, raw data, processed data, source code, tests, and CI workflows.
- **Local-first & free**: no cloud services, no paid tools; everything runs locally or on free GitHub Actions runners.

---

## Repository structure

- `notebooks/`  
  Exploratory data analysis, experimentation, and prototyping notebooks.

- `data/raw/`  
  Immutable source datasets. **Git-ignored** so large/PII data never leaves your machine.

- `data/processed/`  
  Cleaned/feature-engineered datasets and intermediate artifacts. Also **git-ignored**.

- `src/`  
  Python source code for:
  - data loading and validation,
  - feature engineering,
  - model training/evaluation,
  - drift monitoring and alerting,
  - retraining pipelines.

- `models/`  
  Versioned model artifacts (e.g., serialized models, pickles, ONNX, metadata). **Tracked in git** so you can tie models to commits.

- `tests/`  
  Unit/integration tests for data utilities, training logic, and drift monitoring code.

- `.github/workflows/`  
  GitHub Actions workflows, including CI and scheduled retraining (cron-based).

---

## Data quality gate

Before drift checks (and in the retrain workflow), `data/processed/current.csv` is validated by `src/validate_data.py`:

- **Required columns** – All columns from `data/processed/reference.csv` must exist in current (or at least `feature1`, `feature2`, `target` if no reference).
- **Target binary** – The target column must contain only 0 and 1.
- **No all-null columns** – No column can be completely null.
- **Numeric diversity** – Every numeric column must have at least two distinct values (avoids degenerate drift/fit).

Validation uses Great Expectations minimally (programmatic expectations). Exit code **0** = pass, **1** = fail. In the scheduled retrain workflow, if the data quality gate fails, the job stops and drift check / retrain are not run.

Run locally: `python src/validate_data.py`.

---

## Local quickstart

All commands below assume a Unix-like shell (macOS/Linux). On Windows, adapt the activation command for your shell.

**Using Make (recommended):** from the repo root, run `make setup` once to create a venv and install dependencies. Then:

| Command    | Description |
|-----------|-------------|
| `make drift`  | Run drift check (reference vs current in `data/processed/`) |
| `make train`  | Train model on `data/processed/current.csv`, target `target`, MLflow experiment `local_run` |
| `make serve`  | Start FastAPI app with uvicorn (reload on file changes) |
| `make test`   | Run pytest (quiet) |
| `make lint`   | Run ruff check |
| `make format` | Run black formatter |

Manual setup (if not using Make):

- **1. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

- **2. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For linting and tests, also install dev tools: `pip install -r requirements-dev.txt`

- **3. Run tests**

```bash
pytest
```

- **4. Work with notebooks**

```bash
pip install notebook  # if not already in requirements
jupyter notebook notebooks/
```

You can place experimental EDA, feature exploration, and prototype training code under `notebooks/`, then promote stable logic into reusable modules under `src/`.

- **5. Run the prediction API locally**

From the repo root, with a trained model at `models/model.pkl` (e.g. after running `python src/train.py --input ...`):

```bash
python src/serve.py
```

The API listens on `http://0.0.0.0:8000`. Health check and predict:

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"feature_a": 1.0, "feature_b": 2.0, "category": "A"}, {"feature_a": 2.0, "feature_b": 3.0, "category": "B"}]}'
```

Response: `{"probabilities": [0.12, 0.34]}` (positive-class probabilities for each row). Shut down with Ctrl+C.

- **6. Run the API with Docker**

Build and run the FastAPI service in a container (ensure `models/model.pkl` exists so the image includes a trained model):

```bash
docker build -t driftguard-ml .
docker run -p 8000:8000 driftguard-ml
```

Then open `http://localhost:8000/health` or send `POST /predict` as above. Stop the container with Ctrl+C or `docker stop <container_id>`.

---

## Scheduled retraining with GitHub Actions (high level)

DriftGuard-ML is structured so you can use **GitHub Actions** (free tier) to automatically retrain your models on a schedule, without any cloud services:

- **1. Cron-based trigger**  
  A workflow in `.github/workflows/` is configured with a `schedule` block (cron expression). This tells GitHub Actions to start a run periodically (e.g., nightly or weekly) on your default branch.

- **2. Environment setup**  
  The workflow:
  - checks out the repository,
  - sets up Python (e.g., `python-version: '3.11'`),
  - installs dependencies via `pip install -r requirements.txt`.

- **3. Retraining pipeline execution**  
  A dedicated entrypoint script in `src/` (for example, `src/pipelines/retrain.py`) is called. That script is responsible for:
  - loading the latest raw/processed data (local paths or downloaded from wherever you store data),
  - training or fine-tuning the model,
  - evaluating metrics and drift statistics,
  - writing updated artifacts into `models/`.

- **4. Committing model updates**  
  The workflow can be configured to:
  - commit the updated contents of `models/` back to the repository, or
  - open a pull request that contains the new model artifacts and any metadata reports, so you can review and merge.

Because `models/` is tracked by git and `data/` is ignored, your repository stays lightweight while still preserving a full history of model versions and training code. Everything runs on GitHub’s free hosted runners—no additional cloud infrastructure or paid services required.

