FROM python:3.11-slim

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
