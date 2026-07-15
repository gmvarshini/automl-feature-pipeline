FROM python:3.11-slim

# LightGBM and XGBoost link against the OpenMP runtime at import time, so the
# slim base needs libgomp1. Without it, unpickling a FLAML/LightGBM model at
# serve time fails with "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install UV, the fast dependency manager, from its official static binary.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first so the sync layer is cached across code edits.
COPY pyproject.toml uv.lock ./

# Install dependencies into an in-project virtual environment.
RUN uv sync --frozen --no-dev

# Copy the rest of the project.
COPY . .

# Serve on port 8000.
EXPOSE 8000

# Launch the FastAPI serving app with uvicorn through the UV managed venv.
CMD ["uv", "run", "--no-dev", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
