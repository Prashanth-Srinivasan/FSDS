"""
test_train.py
--------------
pytest script for train.py

"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from src.housing_price.train import train_model


@pytest.fixture
def sample_train_df():
    """pytest function to create sample data"""
    np.random.seed(42)
    n = 10000

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
def train_csv_path(tmp_path, sample_train_df):
    """pytest function to save train data in temp location"""
    file_path = tmp_path / "train.csv"
    sample_train_df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def mock_logger():
    """pytest funtion to test logger"""
    logger = MagicMock()
    logger.info = MagicMock()
    return logger


def test_train_model_lr(train_csv_path, mock_logger):
    """
    test case for lr model validation
    """
    model = train_model(train_csv_path, "lr", mock_logger)

    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")


def test_train_model_dtr(train_csv_path, mock_logger):
    """
    test case for dtr model validation
    """
    model = train_model(train_csv_path, "dtr", mock_logger)

    assert isinstance(model, DecisionTreeRegressor)
    assert hasattr(model, "feature_importances_")


def test_train_model_rfr_rs(train_csv_path, mock_logger):
    """
    test case for rfr_rs model validation
    """
    model = train_model(train_csv_path, "rfr_rs", mock_logger)

    assert isinstance(model, RandomizedSearchCV)
    assert hasattr(model, "best_estimator_")


def test_train_model_rfr_gs(train_csv_path, mock_logger):
    """
    test case for rfr_gs model validation
    """
    model = train_model(train_csv_path, "rfr_gs", mock_logger)

    assert isinstance(model, GridSearchCV)
    assert hasattr(model, "best_estimator_")


def test_train_model_invalid_name(train_csv_path, mock_logger):
    """
    test case for missing model input
    """
    model = train_model(train_csv_path, "invalid_model", mock_logger)
    assert model is None
