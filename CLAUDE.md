You are building a containerized feature engineering and AutoML pipeline with model serving. Follow these steps in order.
Step 1: Initialize the project
Create the directory structure above, initialize Git, and create a `.gitignore` excluding `.env`, `__pycache__`, `.venv`, and trained model artifacts larger than a reasonable size, with a `.gitkeep` in `models/`.
Step 2: Set up UV
Run `uv init` and `uv venv`. Add dependencies with `uv add scikit-learn lightgbm flaml pandas pydantic pydantic-settings fastapi uvicorn python-dotenv` and dev dependencies `uv add --dev pytest ruff httpx`.
Step 3: Source a dataset
Use a real world tabular dataset suited to either classification or regression, such as housing prices, customer churn, or loan approval data, with a clear target column. Save it as `data/dataset.csv` and document its source in the README.
Step 4: Build the configuration module
In `src/config.py`, define a Pydantic Settings class `AppConfig` loading `TARGET_COLUMN`, `TASK_TYPE` (either "classification" or "regression"), `AUTOML_TIME_BUDGET_SECONDS` (default 60), and `MODEL_OUTPUT_PATH` (default "./models/model.pkl"). Include full docstrings and type hints, plus `.env.example` and `.env`.
Step 5: Build the feature engineering module
In `src/feature_engineering.py`, write a function `engineer_features` that handles missing value imputation, encodes categorical variables appropriately, and creates at least two derived features based on domain reasoning about the chosen dataset. Include a docstring explaining each transformation and its rationale.
Step 6: Build the training module
In `src/training.py`, write a function `train_with_automl` that uses FLAML to search across multiple model types within the configured time budget, using a proper train validation holdout to select the best model, and saves the trained model along with a metadata file recording which model type was selected and its validation performance. Include a docstring explaining how the search avoids overfitting to the validation set.
Step 7: Build request schemas and serving
In `src/schemas.py`, define a Pydantic model `PredictionRequest` matching the feature columns used at training time with appropriate types and validation constraints, and a `PredictionResponse` model. In `src/serve.py`, build a FastAPI application with a `/predict` endpoint that loads the saved model, validates incoming requests against the schema, and returns predictions with a confidence or probability where applicable.
Step 8: Build the main entry point
In `src/main.py`, write a script that runs feature engineering and training end to end when invoked with a `train` argument, separate from the serving application.
Step 9: Write the Dockerfile
Write a `Dockerfile` using a slim Python 3.11 base image, installing UV, copying the project, running `uv sync`, and exposing the FastAPI serving application on port 8000 via uvicorn. Write a `docker-compose.yml` that mounts the models directory as a volume.
Step 10: Write tests, README, and finalize
In `tests/test_feature_engineering.py`, write pytest tests verifying missing value imputation and derived feature calculations against known inputs. Write a FastAPI test using httpx verifying the `/predict` endpoint returns a valid response for a well formed request and a validation error for a malformed one. Write a README explaining the dataset, feature engineering choices, the AutoML selection process and results, how to train and serve with Docker Compose, and an example curl command demonstrating a prediction request. Do not use em dashes. Run `ruff check`, fix issues, verify the Docker build succeeds, commit with a descriptive message, and push to a new GitHub repository named `automl-feature-pipeline`. Report the repository URL.
---
Key Concepts to Highlight for Learning
Include a short section explaining why the AutoML search uses a validation holdout rather than the test set directly, and why Pydantic validation on the API request matters for a production serving endpoint, since input validation failures are a common real world source of production incidents.