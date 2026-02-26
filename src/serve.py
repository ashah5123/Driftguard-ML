"""FastAPI app: health check and predict using models/model.pkl."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.pkl")
METADATA_PATH = Path("models/metadata.json")
model = None
feature_cols: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_cols
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        feature_cols[:] = meta.get("feature_cols", [])
        if not feature_cols:
            logger.error("models/metadata.json is missing 'feature_cols' or it is empty.")
            raise RuntimeError("models/metadata.json must contain a non-empty 'feature_cols' list.")
    else:
        logger.error("models/metadata.json not found.")
        raise RuntimeError("models/metadata.json not found; cannot load feature_cols.")
    yield
    model = None
    feature_cols[:] = []


app = FastAPI(title="DriftGuard-ML", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Request body: list of feature records."""

    data: list[dict[str, float | str]] = Field(..., min_length=1, description="List of feature dicts")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; ensure models/model.pkl exists.")
    df = pd.DataFrame(request.data)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_columns": sorted(list(missing))},
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
