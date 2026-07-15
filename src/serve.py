"""FastAPI serving application.

Exposes a ``/predict`` endpoint that validates an incoming customer record
against :class:`~src.schemas.PredictionRequest`, runs the same stateless feature
engineering used in training, and returns the model's prediction together with a
confidence score. The model is loaded once and cached for the process lifetime.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import get_config
from src.feature_engineering import engineer_features
from src.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="AutoML Churn Prediction API",
    description="Serves churn predictions from a FLAML selected model.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    """Load and cache the trained model bundle from disk.

    Returns:
        A dict with the fitted ``model`` and the ordered ``feature_columns``.

    Raises:
        HTTPException: 503 if the model file has not been created yet.
    """

    config = get_config()
    model_path = Path(config.model_output_path)
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not found at {model_path}. Train it first with "
                "'python -m src.main train'."
            ),
        )
    with model_path.open("rb") as handle:
        return pickle.load(handle)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by orchestrators and the Docker health check."""

    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score a single customer record for churn.

    The request has already been validated against the schema by FastAPI before
    this handler runs, so the body is guaranteed to hold well typed feature
    values.

    Args:
        request: A validated customer record.

    Returns:
        The predicted label plus, for classification, the model's confidence.
    """

    bundle = load_model()
    model = bundle["model"]
    feature_columns: list[str] = bundle["feature_columns"]
    config = get_config()

    # Build a one row frame and apply the identical feature engineering.
    raw = pd.DataFrame([request.model_dump()])
    features = engineer_features(raw, target_column=config.target_column)

    # Align to the exact training column order.
    features = features.reindex(columns=feature_columns)

    prediction = model.predict(features)[0]

    probability = None
    if config.task_type == "classification":
        proba = model.predict_proba(features)[0]
        classes = list(model.classes_)
        probability = float(proba[classes.index(prediction)])

    return PredictionResponse(
        prediction=str(prediction),
        probability=probability,
        model_name=str(getattr(model, "best_estimator", "unknown")),
    )
