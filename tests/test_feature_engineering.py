"""Tests for the stateless feature engineering transformations."""

from __future__ import annotations

import pandas as pd

from src.feature_engineering import engineer_features


def _base_row(**overrides: object) -> dict[str, object]:
    """Return a complete raw Telco row, with optional field overrides."""

    row: dict[str, object] = {
        "customerID": "0001-AAAA",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "500.0",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_identifier_is_dropped() -> None:
    """The non predictive customerID column must be removed."""

    result = engineer_features(pd.DataFrame([_base_row()]))
    assert "customerID" not in result.columns


def test_missing_total_charges_is_imputed_from_tenure_and_monthly() -> None:
    """A blank TotalCharges is reconstructed as tenure * MonthlyCharges."""

    # A brand new customer: blank total, zero tenure.
    new_customer = _base_row(tenure=0, MonthlyCharges=70.0, TotalCharges=" ")
    result = engineer_features(pd.DataFrame([new_customer]))
    assert result.loc[0, "TotalCharges"] == 0.0

    # An existing customer with a blank total gets tenure * monthly.
    existing = _base_row(tenure=4, MonthlyCharges=25.0, TotalCharges=" ")
    result = engineer_features(pd.DataFrame([existing]))
    assert result.loc[0, "TotalCharges"] == 100.0


def test_avg_monthly_charges_derivation() -> None:
    """avg_monthly_charges equals TotalCharges / tenure for tenured customers."""

    row = _base_row(tenure=10, MonthlyCharges=50.0, TotalCharges="600.0")
    result = engineer_features(pd.DataFrame([row]))
    assert result.loc[0, "avg_monthly_charges"] == 60.0


def test_avg_monthly_charges_falls_back_for_zero_tenure() -> None:
    """Zero tenure customers use the current monthly charge as the average."""

    row = _base_row(tenure=0, MonthlyCharges=42.0, TotalCharges=" ")
    result = engineer_features(pd.DataFrame([row]))
    assert result.loc[0, "avg_monthly_charges"] == 42.0


def test_num_addon_services_counts_yes_only() -> None:
    """num_addon_services counts only the add-ons set to 'Yes'."""

    # Base row has OnlineSecurity, DeviceProtection, StreamingTV = Yes -> 3.
    result = engineer_features(pd.DataFrame([_base_row()]))
    assert result.loc[0, "num_addon_services"] == 3

    all_off = _base_row(
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
    )
    result = engineer_features(pd.DataFrame([all_off]))
    assert result.loc[0, "num_addon_services"] == 0


def test_is_new_customer_flag() -> None:
    """Customers within their first year are flagged as new."""

    new = engineer_features(pd.DataFrame([_base_row(tenure=12)]))
    tenured = engineer_features(pd.DataFrame([_base_row(tenure=13)]))
    assert new.loc[0, "is_new_customer"] == 1
    assert tenured.loc[0, "is_new_customer"] == 0


def test_categorical_gaps_filled_with_sentinel() -> None:
    """Missing categorical values are imputed with the 'Unknown' sentinel."""

    row = _base_row(InternetService=None)
    result = engineer_features(pd.DataFrame([row]))
    assert "Unknown" in list(result["InternetService"])


def test_target_column_passed_through_untouched() -> None:
    """The target column is preserved exactly when present."""

    result = engineer_features(pd.DataFrame([_base_row(Churn="Yes")]))
    assert result.loc[0, "Churn"] == "Yes"
