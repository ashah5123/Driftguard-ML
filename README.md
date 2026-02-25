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

## Local quickstart

All commands below assume a Unix-like shell (macOS/Linux). On Windows, adapt the activation command for your shell.

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

