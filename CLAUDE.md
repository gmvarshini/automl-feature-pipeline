# CLAUDE.md

Notes for working on this repo with Claude Code.

## Project

A customer churn model for the IBM Telco dataset (7,043 customers). The pipeline
cleans the data and adds three features, lets FLAML search LightGBM, XGBoost,
random forest and extra trees, and serves the best model through a FastAPI
`/predict` endpoint. Everything runs in Docker with Docker Compose.

## Layout

- `src/config.py`: settings (target column, task type, time budget, model path)
- `src/feature_engineering.py`: stateless cleaning and derived features
- `src/training.py`: 80/20 stratified split, FLAML search with 5-fold CV, test metrics
- `src/schemas.py`: Pydantic request and response models
- `src/serve.py`: FastAPI app with `/health` and `/predict`
- `src/main.py`: `train` command
- `tests/`: feature engineering tests and API tests
- `Dockerfile`, `docker-compose.yml`: container setup

## Commands

```bash
uv sync                                    # install
uv run python -m src.main train            # train, writes models/model.pkl
uv run uvicorn src.serve:app --reload      # serve on localhost:8000
uv run pytest                              # tests
uv run ruff check .                        # lint
docker compose up --build                  # run in a container
```

## Rules

- `engineer_features` must stay stateless. The same function runs in training
  and on every request, so no values learned from the training data go in there.
- The 20% test split is only used once, after the search. Model selection uses
  cross validation on the training part only.
- Any new input field must be added to `PredictionRequest` with strict types or
  allowed values, so bad requests get a 422.
- System libraries needed at runtime go in the Dockerfile. LightGBM needs
  `libgomp1`, and the slim image does not include it.
- Trained models stay out of Git. `models/` is mounted as a volume.
- Type hints and Google style docstrings on all functions.
- Plain English in docs and comments. No em dashes.

## CI

`.github/workflows/ci.yml` has two jobs. `test` runs ruff and pytest. `container`
trains a quick model, starts the API with Docker Compose and sends a real request
to `/predict`. This catches problems that `/health` alone misses, like the
missing `libgomp1` library.
