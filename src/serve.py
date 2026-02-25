"""FastAPI app: health check and predict using models/model.pkl."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None
    yield
    model = None


app = FastAPI(title="DriftGuard-ML", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Request body: list of feature records."""

    data: list[dict[str, float | str]] = Field(..., min_length=1, description="List of feature dicts")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, list[float]]:
    if model is None:
        raise RuntimeError("Model not loaded; ensure models/model.pkl exists.")
    X = pd.DataFrame(request.data)
    proba = model.predict_proba(X)[:, 1]
    return {"probabilities": proba.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
