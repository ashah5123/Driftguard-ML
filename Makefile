.PHONY: setup drift train serve test lint format data-download data-prep

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

drift:
	$(PYTHON) src/check_drift.py

train:
	$(PYTHON) src/train.py --input data/processed/current.csv --target Delayed --mlflow_experiment local_run

serve:
	$(VENV)/bin/uvicorn src.serve:app --reload --host 0.0.0.0 --port 8000

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/black .

data-download:
	$(PYTHON) src/data/download_flights_sample.py

data-prep:
	$(PYTHON) src/data/preprocess_flights.py --reference-year 2021 --current-year 2022 --max-rows 2000000
