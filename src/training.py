"""AutoML training with FLAML.

This module loads the raw dataset, applies the stateless feature engineering
from :mod:`src.feature_engineering`, and hands the result to FLAML, which
searches across several model families (LightGBM, XGBoost, random forests,
extra trees, and more) within a fixed time budget to select the best
configuration. The fitted model and a JSON metadata record are written to disk.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from src.config import AppConfig, get_config
from src.feature_engineering import engineer_features

DATA_PATH = Path("data/dataset.csv")


def _evaluate(
    automl: AutoML,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> dict[str, float]:
    """Score the fitted model on the untouched test holdout.

    Args:
        automl: The fitted FLAML ``AutoML`` object.
        x_test: Engineered test features never seen during the search.
        y_test: True test labels.
        task_type: ``"classification"`` or ``"regression"``.

    Returns:
        A mapping of metric name to value for the test holdout.
    """

    predictions = automl.predict(x_test)
    if task_type == "classification":
        metrics = {"accuracy": float(accuracy_score(y_test, predictions))}
        # ROC AUC needs probabilities and is only defined for binary targets.
        try:
            proba = automl.predict_proba(x_test)
            if proba.shape[1] == 2:
                positive_label = automl.classes_[1]
                metrics["roc_auc"] = float(
                    roc_auc_score((y_test == positive_label).astype(int), proba[:, 1])
                )
        except (AttributeError, ValueError):
            pass
        return metrics

    return {
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }


def train_with_automl(config: AppConfig | None = None) -> dict[str, Any]:
    """Run the end to end AutoML training pipeline.

    The routine carves the data into a training portion and a test holdout that
    FLAML never sees. FLAML is then given only the training portion and performs
    its own internal resampling (k fold cross validation) to score every
    candidate configuration during the search.

    **How the search avoids overfitting to the validation set.** Two safeguards
    work together. First, FLAML does not judge a configuration on a single fixed
    validation split; it uses cross validation, so a config must perform well
    across several rotating folds rather than getting lucky on one partition.
    This penalizes configurations that merely memorize a particular slice of the
    data. Second, the accuracy we *report* is computed on the separate test
    holdout that was removed before the search began and was never used for
    model selection or hyperparameter tuning. The cross validation score guides
    the search; the test score is the honest, unbiased estimate of how the model
    will behave on genuinely new customers.

    Args:
        config: Application configuration. When ``None``, it is loaded from the
            environment and ``.env`` file via :func:`src.config.get_config`.

    Returns:
        The metadata dictionary describing the selected model and its
        performance, which is also written to disk next to the model file.
    """

    config = config or get_config()

    # Load and engineer.
    raw = pd.read_csv(DATA_PATH)
    engineered = engineer_features(raw, target_column=config.target_column)

    y = engineered[config.target_column]
    x = engineered.drop(columns=[config.target_column])
    feature_columns = list(x.columns)

    # Hold out a test set the search will never see. Stratify for classification
    # so both splits keep the same class balance.
    stratify = y if config.task_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Configure and run the FLAML search.
    automl = AutoML()
    metric = "roc_auc" if config.task_type == "classification" else "r2"
    automl.fit(
        X_train=x_train,
        y_train=y_train,
        task=config.task_type,
        metric=metric,
        time_budget=config.automl_time_budget_seconds,
        eval_method="cv",
        n_splits=5,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
        seed=42,
        verbose=1,
    )

    test_metrics = _evaluate(automl, x_test, y_test, config.task_type)

    # The FLAML search score is 1 - best_loss for maximization metrics.
    validation_score = float(1.0 - automl.best_loss)

    metadata: dict[str, Any] = {
        "task_type": config.task_type,
        "target_column": config.target_column,
        "selected_model": automl.best_estimator,
        "best_config": automl.best_config,
        "search_metric": metric,
        "validation_score": validation_score,
        "validation_note": (
            "Cross validated score over 5 folds used by FLAML to select the "
            "model. Not used as the final performance estimate."
        ),
        "test_metrics": test_metrics,
        "time_budget_seconds": config.automl_time_budget_seconds,
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_columns": feature_columns,
    }

    # Persist the fitted model and metadata.
    model_path = Path(config.model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"model": automl, "feature_columns": feature_columns}, handle)

    metadata_path = model_path.with_suffix(".meta.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    return metadata
