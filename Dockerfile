FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache .

# -------------------------

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY src ./src
COPY artifacts ./artifacts

RUN mkdir -p artifacts logs mlruns data

ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

EXPOSE 8000

CMD ["sh", "-c", "uvicorn housing_price.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

