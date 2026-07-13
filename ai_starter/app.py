from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/model.joblib"))
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="AI Starter Inference API", version="1.0.0")
model: Any | None = None


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    prediction: str
    class_index: int
    probabilities: list[float]


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        model = None
        return
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded from '{MODEL_PATH}'. "
                "Train a model first with `python train.py`."
            ),
        )

    probabilities = model.predict_proba([payload.features])[0].tolist()
    class_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
    prediction = CLASS_NAMES[class_index]
    return PredictionResponse(
        prediction=prediction,
        class_index=class_index,
        probabilities=probabilities,
    )
