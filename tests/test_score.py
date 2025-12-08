import pandas as pd
import pytest
import joblib
from unittest.mock import MagicMock

from sklearn.linear_model import LinearRegression

from src.housing_price.score import predict_result


@pytest.fixture
def sample_test_df():
    return pd.DataFrame(
        {
            "median_income": [3.5, 4.2, 2.3],
            "total_rooms": [100, 200, 150],
            "median_house_value": [150000, 180000, 120000],
        }
    )


@pytest.fixture
def test_csv_path(tmp_path, sample_test_df):
    file = tmp_path / "test.csv"
    sample_test_df.to_csv(file, index=False)
    return str(file)


@pytest.fixture
def trained_model_path(tmp_path, sample_test_df):
    x = sample_test_df.drop("median_house_value", axis=1)
    y = sample_test_df["median_house_value"]

    model = LinearRegression()
    model.fit(x, y)

    model_file = tmp_path / "lr.pkl"
    joblib.dump(model, model_file)

    return str(model_file)


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = MagicMock()
    return logger


def test_predict_result(test_csv_path, trained_model_path, mock_logger):
    df = predict_result(trained_model_path, test_csv_path, mock_logger)

    assert isinstance(df, pd.DataFrame)

    assert "lr_predictions" in df.columns
