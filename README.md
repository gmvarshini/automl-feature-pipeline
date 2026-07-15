# AutoML Feature Pipeline

A containerized, end to end feature engineering and AutoML pipeline with a
FastAPI model serving layer. It trains a customer churn classifier by searching
across several model families with FLAML, then serves predictions through a
validated REST endpoint.

## Dataset

The project uses the **Telco Customer Churn** dataset, a well known real world
tabular dataset of 7,043 telecom customers. Each row describes a customer's
account (tenure, contract type, payment method), the services they subscribe to
(phone, internet, streaming, and various add-ons), and their billing figures.
The target column is `Churn` (`Yes` or `No`), making this a binary
classification problem.

- Source: IBM sample data, mirrored at
  `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- Stored locally at `data/dataset.csv`.

## Feature engineering

All transformations live in `src/feature_engineering.py` and are deliberately
**stateless**: each is either a deterministic rule or a per-row computation, so
the identical code runs on the full training table and on a single request at
serving time. This removes train/serve skew, a frequent and hard to diagnose
source of degraded production predictions.

The transformations are:

1. **Drop `customerID`.** A unique identifier carries no generalizable signal
   and would encourage memorization.
2. **Coerce `TotalCharges` to numeric.** The raw file stores it as text and
   encodes brand new customers as an empty string.
3. **Impute missing `TotalCharges`** as `tenure * MonthlyCharges`, which
   correctly yields `0` for zero tenure customers rather than a misleading mean.
4. **Impute residual gaps** with `0` for numeric columns and an `"Unknown"`
   sentinel for categorical columns, both stateless rules.
5. **Derived `avg_monthly_charges`** (`TotalCharges / tenure`, falling back to
   `MonthlyCharges` for zero tenure). This captures the customer's realized
   average bill, which can diverge from the current monthly charge after plan
   changes and is a strong churn signal.
6. **Derived `num_addon_services`**, a count of the six optional add-on services
   the customer subscribes to. More add-ons means a deeper, stickier
   relationship and typically lower churn.
7. **Derived `is_new_customer`**, a flag for customers in their first year, who
   churn at a markedly higher rate.

Categorical encoding is intentionally delegated to the FLAML AutoML layer, which
fits its own encoders during training and persists them inside the saved model.
This keeps a single fitted transformer as the source of truth rather than
re-deriving encodings from a single request row.

## AutoML selection process and results

Training lives in `src/training.py`. The flow is:

1. Load the data and apply feature engineering.
2. Split off a 20 percent **test holdout**, stratified by class, that the search
   never sees.
3. Hand the remaining 80 percent to FLAML with a configurable time budget.
   FLAML searches across LightGBM, XGBoost, random forest, and extra trees,
   using **5 fold cross validation** internally to score each candidate
   configuration and optimizing ROC AUC.
4. Evaluate the selected model on the untouched test holdout and record both the
   cross validated search score and the honest test metrics.

The fitted model is written to `models/model.pkl` and a human readable
`models/model.meta.json` records the selected model type, its configuration, and
its performance.

Representative results from a short 20 second budget run:

| Metric                  | Value |
| ----------------------- | ----- |
| Selected model          | LightGBM (`lgbm`) |
| Cross validated ROC AUC | ~0.85 |
| Test accuracy           | ~0.80 |
| Test ROC AUC            | ~0.85 |

A longer budget (the default is 60 seconds, configurable via
`AUTOML_TIME_BUDGET_SECONDS`) lets FLAML explore more configurations.

## Configuration

Settings are loaded by `src/config.py` (a Pydantic Settings class) from
environment variables or a `.env` file. Copy `.env.example` to `.env` to start:

| Variable                      | Default               | Meaning |
| ----------------------------- | --------------------- | ------- |
| `TARGET_COLUMN`               | `Churn`               | Column to predict. |
| `TASK_TYPE`                   | `classification`      | `classification` or `regression`. |
| `AUTOML_TIME_BUDGET_SECONDS`  | `60`                  | FLAML search budget. |
| `MODEL_OUTPUT_PATH`           | `./models/model.pkl`  | Where the model is saved and loaded. |

## Running locally

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                       # install dependencies
uv run python -m src.main train   # feature engineering + AutoML training
uv run uvicorn src.serve:app --reload   # serve on http://localhost:8000
```

Run the tests and linter with:

```bash
uv run pytest
uv run ruff check .
```

## Train and serve with Docker Compose

The `Dockerfile` builds a slim Python 3.11 image, installs UV, syncs
dependencies, and serves the API on port 8000. `docker-compose.yml` mounts the
`models/` directory as a volume so a model trained on the host is served without
rebuilding the image.

The image also installs the `libgomp1` system package. LightGBM and XGBoost
link against the OpenMP runtime (`libgomp.so.1`) at import time, and the slim
base image does not ship it. Without this package, unpickling the FLAML selected
model inside the `/predict` handler fails with `libgomp.so.1: cannot open shared
object file`, returning a `500` while `/health` still reports `200` because it
never loads the model. See commit `53aef60` for the fix.

```bash
# 1. Train a model on the host (writes models/model.pkl).
uv run python -m src.main train

# 2. Build and start the serving container.
docker compose up --build
```

The API is then available at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

To train inside the container instead, run:

```bash
docker compose run --rm api uv run python -m src.main train
```

## Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 450.5
  }'
```

Example response:

```json
{
  "prediction": "Yes",
  "probability": 0.74,
  "model_name": "lgbm"
}
```

## Key concepts for learning

**Why the search uses a validation holdout rather than the test set directly.**
During the AutoML search, FLAML tries many model families and hyperparameter
configurations and repeatedly asks "which one scores best?" If it scored those
configurations on the test set, the very act of picking the winner would bleed
information from the test set into model selection: after enough tries, some
configuration looks great on that specific data by chance rather than genuine
skill, and the reported number becomes optimistic. To prevent this, FLAML scores
candidates using cross validation folds carved from the training data only, so
selection pressure never touches the test set. Cross validation also averages
performance across several rotating folds, so a configuration must generalize
rather than get lucky on one partition. The test holdout is opened exactly once,
after the search is finished, to give an unbiased estimate of real world
performance.

**Why Pydantic validation on the API request matters.** The `/predict` endpoint
validates every incoming request against `PredictionRequest` before the model is
ever called. Category fields are restricted to their known value sets and
numeric fields to sensible ranges. In production, malformed or unexpected input
is one of the most common causes of incidents: a missing field, a typo in a
category, or a string where a number was expected can otherwise crash the
handler or, worse, silently produce a nonsensical prediction that downstream
systems trust. Validating at the edge turns these failures into an immediate,
descriptive `422` error that tells the caller exactly what was wrong, which is
far safer and easier to debug than a corrupted prediction.
