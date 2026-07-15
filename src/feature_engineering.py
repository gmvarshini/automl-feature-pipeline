"""Feature engineering for the Telco Customer Churn dataset.

The :func:`engineer_features` function is intentionally *stateless*: every
transformation is either a deterministic rule or a per-row computation, so the
exact same code path runs during training (on the full table) and during
serving (on a single incoming request). This eliminates train/serve skew, which
is a frequent source of silently degraded predictions in production.

Categorical encoding is deliberately left to the downstream FLAML AutoML layer,
which fits its own label encoders during training and persists them inside the
saved model object. That keeps a single fitted transformer as the source of
truth rather than re-deriving encodings at request time from a single row.
"""

from __future__ import annotations

import pandas as pd

# Identifier column that carries no predictive signal and must not leak into
# the model.
ID_COLUMN = "customerID"

# The six optional add-on services offered on top of internet connectivity.
# Each is "Yes", "No", or "No internet service".
ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """Transform raw Telco churn rows into model ready features.

    The function performs the following transformations, each chosen with the
    domain in mind:

    1. **Drop the identifier.** ``customerID`` is unique per row and would let a
       model memorize individuals instead of learning generalizable patterns.

    2. **Coerce ``TotalCharges`` to numeric.** The raw file stores this column
       as text and encodes brand new customers (tenure of zero months) as an
       empty string, which becomes ``NaN`` under numeric coercion.

    3. **Impute missing ``TotalCharges``.** A missing total is reconstructed as
       ``tenure * MonthlyCharges``. For a zero tenure customer this correctly
       yields ``0``, which is more faithful than a blanket mean or median.

    4. **Impute any remaining gaps.** Residual numeric gaps are filled with
       ``0`` and residual categorical gaps with the sentinel ``"Unknown"``.
       Both rules are stateless, so a single request row is imputed identically
       to the training table.

    5. **Derived feature ``avg_monthly_charges``.** Total lifetime spend divided
       by tenure (falling back to ``MonthlyCharges`` for zero tenure). This
       captures a customer's *realized* average bill, which can diverge from the
       current monthly charge after plan changes and is a strong churn signal.

    6. **Derived feature ``num_addon_services``.** A count of how many of the six
       optional add-on services the customer subscribes to. More add-ons means a
       deeper, stickier relationship with the provider and typically lower churn.

    7. **Derived feature ``is_new_customer``.** A flag for customers within their
       first year (tenure of twelve months or fewer), who churn at a markedly
       higher rate than tenured customers.

    Args:
        df: Raw input rows. May include the ``target_column`` (training) or omit
            it (serving). Must contain the standard Telco churn feature columns.
        target_column: Name of the label column. It is passed through untouched
            when present and ignored when absent.

    Returns:
        A new :class:`pandas.DataFrame` with engineered features. Categorical
        columns are returned as pandas ``category`` dtype so the downstream
        AutoML layer can encode them consistently; the target column, if
        present, is left exactly as received.
    """

    out = df.copy()

    # 1. Drop the identifier if it is present.
    if ID_COLUMN in out.columns:
        out = out.drop(columns=[ID_COLUMN])

    # 2. Coerce TotalCharges to numeric; blank strings become NaN.
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    # 3. Reconstruct missing TotalCharges from tenure and monthly charge.
    if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(out.columns):
        reconstructed = out["tenure"] * out["MonthlyCharges"]
        out["TotalCharges"] = out["TotalCharges"].fillna(reconstructed)

    # 5. Derived: realized average monthly spend over the customer lifetime.
    if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(out.columns):
        tenure_safe = out["tenure"].where(out["tenure"] > 0, other=1)
        avg = out["TotalCharges"] / tenure_safe
        # For zero tenure customers, fall back to the current monthly charge.
        out["avg_monthly_charges"] = avg.where(out["tenure"] > 0, out["MonthlyCharges"])

    # 6. Derived: number of optional add-on services subscribed.
    present_addons = [c for c in ADDON_SERVICE_COLUMNS if c in out.columns]
    if present_addons:
        out["num_addon_services"] = (out[present_addons] == "Yes").sum(axis=1)

    # 7. Derived: flag for first year customers.
    if "tenure" in out.columns:
        out["is_new_customer"] = (out["tenure"] <= 12).astype(int)

    # 4. Impute residual gaps. Numeric columns get 0, object columns get a
    # sentinel category. Done last so derived features are imputed too.
    for column in out.columns:
        if column == target_column:
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].fillna(0)
        else:
            out[column] = out[column].fillna("Unknown").astype("category")

    return out
