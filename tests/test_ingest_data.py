"""
test_ingest_data.py
--------------
pytest script for ingest_data.py

"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.housing_price import config
# fmt: off
from src.housing_price.ingest_data import data_train_test_split, load_housing_data

# fmt: on


@pytest.mark.local_only
def test_data_file_exists():
    """pytest function for raw data path"""
    assert os.path.exists(
        config.housing_raw_path
    ), f"Config path missing: {config.housing_raw_path}"


@pytest.mark.local_only
def test_loaded_data_has_rows():
    """pytest function for loading raw data"""
    df = load_housing_data()
    assert len(df) > 0


@pytest.fixture
def sample_housing_data():
    """pytest function for creating sample data"""
    np.random.seed(42)
    n = 10000

    data = {
        "longitude": np.random.randint(-123, -120, size=n),
        "latitude": np.random.randint(30, 40, size=n),
        "housing_median_age": np.random.randint(20, 50, size=n),
        "median_income": np.random.uniform(0.5, 6.5, size=n),
        "median_house_value": np.random.randint(50000, 500000, size=n),
        "total_rooms": np.random.randint(1, 20, size=n),
        "total_bedrooms": np.random.randint(1, 10, size=n),
        "population": np.random.randint(1, 15, size=n),
        "households": np.random.randint(1, 10, size=n),
        "ocean_proximity": np.random.choice(
            ["NEAR BAY", "NEAR OCEAN", "ISLAND", "INLAND", "<1H OCEAN"], size=n
        ),
    }

    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


@pytest.fixture(autouse=True)
def cleanup_files():
    """pytest function for cleaning up temp files"""
    for path in [config.train_housing_path, config.test_housing_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
    yield
    for path in [config.train_housing_path, config.test_housing_path]:
        if os.path.exists(path):
            shutil.rmtree(path)


def test_data_split(sample_housing_data):  # noqa: F811
    """pytest function for train test split (ingest_data.py)"""
    train_set, test_set, strat_train_set, strat_test_set = data_train_test_split(
        sample_housing_data
    )

    # Check dataframes are not empty
    assert not train_set.empty
    assert not test_set.empty
    assert not strat_train_set.empty
    assert not strat_test_set.empty

    # Check expected columns
    expected_cols = set(sample_housing_data.columns).union(
        {"rooms_per_household", "bedrooms_per_room", "population_per_household"}
    )
    expected_cols.discard("ocean_proximity")
    # Include dummy columns for ocean_proximity after get_dummies
    expected_cat_cols = {
        "ocean_proximity_ISLAND",
        "ocean_proximity_INLAND",
        "ocean_proximity_NEAR OCEAN",
        "ocean_proximity_NEAR BAY",
    }

    print(expected_cols)
    print(train_set.columns)

    for df in [train_set, test_set, strat_train_set, strat_test_set]:

        missing_col_info = set(expected_cols) - set(df.columns)
        assert expected_cols.issubset(
            df.columns
        ), f"Missing expected columns in dataframe: {missing_col_info}"

        actual_cat_cols = {c for c in df.columns if c.startswith("ocean_proximity_")}

        expected_min_dummy = len(expected_cat_cols) - 1
        actual_dummy = len(actual_cat_cols)

        assert (
            len(actual_cat_cols) >= len(expected_cat_cols) - 1
        ), f"Expected at least {expected_min_dummy} dummy cols, found {actual_dummy}"

    # Check files exist
    for fname in ["train.csv", "test.csv", "strat_train.csv", "strat_test.csv"]:
        assert os.path.exists(
            os.path.join(config.train_housing_path, fname)
        ) or os.path.exists(os.path.join(config.test_housing_path, fname))
