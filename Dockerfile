FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
 && uv pip install --system --no-cache \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    scikit-learn \
    joblib

COPY src ./src
COPY artifacts ./artifacts

ENV PYTHONPATH=/app/src
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "housing_price.api:app", "--host", "0.0.0.0", "--port", "8000"]
