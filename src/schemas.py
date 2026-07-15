"""Pydantic request and response schemas for the serving API.

``PredictionRequest`` mirrors the *raw* feature columns of the Telco churn
dataset (the identifier and the target are excluded). Incoming JSON is validated
against these types and constraints before it ever reaches the model, so
malformed input is rejected at the edge with a clear 422 error instead of
producing a confusing failure or a silently wrong prediction deeper in the
stack. The field names match the dataset columns exactly so a validated request
can be turned into a one row DataFrame with ``model_dump()``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """A single customer record to score for churn.

    Every field corresponds to a column produced by the data provider. String
    fields are constrained to their known category sets so an unexpected value
    (a common cause of production incidents) is rejected up front rather than
    being coerced or passed through to the model.
    """

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(
        description="1 if the customer is a senior citizen, else 0."
    )
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=1000, description="Months the customer has stayed.")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge.")
    TotalCharges: float = Field(ge=0, description="Total charged over the lifetime.")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """The model's verdict for a single customer.

    Attributes:
        prediction: The predicted class label (for churn, ``"Yes"`` or ``"No"``)
            or the point estimate for a regression task.
        probability: For classification, the model's confidence in the predicted
            label, in ``[0, 1]``. ``None`` for regression tasks.
        model_name: The FLAML selected model family that produced the
            prediction, for observability.
    """

    prediction: str
    probability: Optional[float] = Field(
        default=None, ge=0, le=1, description="Confidence for classification tasks."
    )
    model_name: str
