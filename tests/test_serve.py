"""Tests for the FastAPI serving endpoint using the httpx backed test client."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import get_config
from src.serve import app, load_model
from src.training import train_with_automl

WELL_FORMED_REQUEST = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.9,
    "TotalCharges": 450.5,
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a test client, training a quick model first if none exists."""

    config = get_config()
    if not Path(config.model_output_path).exists():
        # Train a small model on a short budget so the endpoint has something
        # to serve. The dataset ships with the repository.
        config.automl_time_budget_seconds = 10
        train_with_automl(config)
        load_model.cache_clear()
    return TestClient(app)


def test_predict_well_formed_request_returns_valid_response(client: TestClient) -> None:
    """A well formed request yields a 200 with a valid prediction payload."""

    response = client.post("/predict", json=WELL_FORMED_REQUEST)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in {"Yes", "No"}
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["model_name"], str)


def test_predict_malformed_request_returns_validation_error(
    client: TestClient,
) -> None:
    """An invalid category value is rejected with a 422 before reaching the model."""

    malformed = dict(WELL_FORMED_REQUEST)
    malformed["Contract"] = "Yearly"  # not a permitted contract type
    response = client.post("/predict", json=malformed)
    assert response.status_code == 422


def test_predict_missing_field_returns_validation_error(client: TestClient) -> None:
    """A request missing a required field is rejected with a 422."""

    incomplete = dict(WELL_FORMED_REQUEST)
    del incomplete["tenure"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
