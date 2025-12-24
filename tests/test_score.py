"""
test_score.py
--------------
pytest script for score.py

"""

import numbers
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.housing_price.score import predict_result


@pytest.fixture
def sample_test_df():
    """pytest function for creating sample data"""
    np.random.seed(42)
    n = 100

    data = {
        "median_income": np.random.uniform(0.5, 6.5, size=n),
        "median_house_value": np.random.randint(50000, 500000, size=n),
        "total_rooms": np.random.randint(1, 20, size=n),
        "total_bedrooms": np.random.randint(1, 10, size=n),
        "population": np.random.randint(1, 15, size=n),
        "households": np.random.randint(1, 10, size=n),
    }

    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


@pytest.fixture
def test_csv_path(tmp_path, sample_test_df):
    """pytest function to save train data in temp path"""
    file = tmp_path / "test.csv"
    sample_test_df.to_csv(file, index=False)
    return str(file)


@pytest.fixture
def trained_model_path(tmp_path, sample_test_df):
    """pytest function to create a dummy model and save to temp path"""
    x = sample_test_df.drop("median_house_value", axis=1)
    y = sample_test_df["median_house_value"]

    model = LinearRegression()
    model.fit(x, y)

    model_file = tmp_path / "lr.pkl"
    joblib.dump(model, model_file)

    return str(model_file)


@pytest.fixture
def mock_logger():
    """pytest funtion to test logger"""
    logger = MagicMock()
    logger.info = MagicMock()
    return logger


def test_predict_result(test_csv_path, trained_model_path, mock_logger):
    """pytest function to test prediction (score.py)"""
    df, metrics = predict_result(trained_model_path, test_csv_path, mock_logger)

    expected_keys = {"mse", "rmse", "mae"}

    assert (
        set(metrics.keys()) == expected_keys
    ), "Metric dictionary keys do not match expected keys"

    for key, value in metrics.items():
        assert isinstance(
            value, numbers.Number
        ), f"Value for {key} must be numeric, got {type(value)}"

    assert isinstance(df, pd.DataFrame)

    assert "lr_predictions" in df.columns
