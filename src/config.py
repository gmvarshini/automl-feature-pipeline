"""Application configuration.

Centralizes all runtime settings in a single, validated Pydantic Settings
object so that the feature engineering, training, and serving modules read
their parameters from one place rather than scattering environment lookups
throughout the codebase.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Runtime configuration loaded from environment variables or a ``.env`` file.

    Attributes:
        target_column: Name of the column in the dataset that the model learns
            to predict. Loaded from the ``TARGET_COLUMN`` environment variable.
        task_type: The kind of supervised learning problem. Either
            ``"classification"`` or ``"regression"``. Loaded from ``TASK_TYPE``.
            The value drives both the FLAML search space and how predictions are
            returned by the serving layer (probabilities for classification,
            point estimates for regression).
        automl_time_budget_seconds: Wall clock time budget, in seconds, granted
            to the FLAML AutoML search. Larger budgets let FLAML evaluate more
            model families and hyperparameter configurations. Loaded from
            ``AUTOML_TIME_BUDGET_SECONDS`` and defaults to ``60``.
        model_output_path: Filesystem path where the trained model pickle is
            written by the training module and read by the serving module.
            Loaded from ``MODEL_OUTPUT_PATH`` and defaults to
            ``"./models/model.pkl"``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow attribute names with the ``model_`` prefix (model_output_path)
        # without colliding with Pydantic's protected namespace.
        protected_namespaces=(),
    )

    target_column: str = "Churn"
    task_type: Literal["classification", "regression"] = "classification"
    automl_time_budget_seconds: int = 60
    model_output_path: str = "./models/model.pkl"


def get_config() -> AppConfig:
    """Build and return an :class:`AppConfig` instance.

    Returns:
        A fully populated configuration object with values resolved from the
        environment and the ``.env`` file.
    """

    return AppConfig()
