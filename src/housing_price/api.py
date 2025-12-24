"""
api.py
-------
API module for House Price Prediction.

This module exposes a FastAPI application to serve a trained model for predicting
house prices. It includes health checks and a /predict endpoint.
"""

import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from housing_price import config


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan_handler(fastapi_app: FastAPI):
    """Lifespan handler to load the model on startup."""
    model_path = os.path.join(config.artifacts_path, config.BEST_MODEL)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")
    fastapi_app.state.MODEL = joblib.load(model_path)
    yield  # application runs
    # Cleanup if necessary


app = FastAPI(
    title="House Price Prediction API",
    version="0.2.0",
    description="Predict house prices using a trained ML model",
    lifespan=lifespan_handler,
)


# Request schema
class HouseFeatures(BaseModel):
    """
    Schema for house features sent in prediction requests.

    Ensures that exactly one ocean proximity flag is True.
    """

    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    income_cat: float

    rooms_per_household: float
    bedrooms_per_room: float
    population_per_household: float

    ocean_proximity_INLAND: bool = Field(False)
    ocean_proximity_ISLAND: bool = Field(False)
    ocean_proximity_NEAR_BAY: bool = Field(False)
    ocean_proximity_NEAR_OCEAN: bool = Field(False)
    ocean_proximity_LESS_THAN_1H_OCEAN: bool = Field(False)

    @model_validator(mode="after")
    def validate_ocean_proximity(self):
        """Ensure exactly one ocean proximity option is True."""
        flags = [
            self.ocean_proximity_INLAND,
            self.ocean_proximity_ISLAND,
            self.ocean_proximity_NEAR_BAY,
            self.ocean_proximity_NEAR_OCEAN,
            self.ocean_proximity_LESS_THAN_1H_OCEAN,
        ]
        if sum(flags) != 1:
            raise ValueError("Exactly one ocean_proximity option must be True")
        return self


# Root
@app.get("/")
def root():
    """Root endpoint for quick status check."""
    return {"message": "House Price Prediction API is running"}


# Health
@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


# Prediction
@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Predict house price based on input features.

    Args:
        features (HouseFeatures): Input JSON schema containing house features.

    Returns:
        dict: Contains prediction value and model used.
    """
    try:
        model = app.state.MODEL
        feature_order = config.FEATURE_ORDER[:-1]  # Exclude target column
        input_df = pd.DataFrame([features.model_dump()])[feature_order]
        input_df = input_df.astype(int, errors="ignore")

        prediction = model.predict(input_df)[0]

        return {
            "prediction": float(prediction),
            "model": config.BEST_MODEL,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
