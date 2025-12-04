import os
import pandas as pd
import numpy as np
import pytest
import shutil

from src import config
from src.ingest_data import load_housing_data, data_train_test_split


def test_data_file_exists():
    assert os.path.exists(
        config.housing_raw_path
    ), f"Config path missing: {config.housing_raw_path}"


def test_loaded_data_has_rows():
    df = load_housing_data()
    assert len(df) > 0


@pytest.fixture
def sample_housing_data():
    categories = ["NEAR BAY", "NEAR OCEAN", "ISLAND"]
    np.random.seed(42)
    n = 10000

    data = {
        "median_income": np.random.uniform(0.5, 6.5, size=n),
        "median_house_value": np.random.randint(50000, 500000, size=n),
        "total_rooms": np.random.randint(1, 20, size=n),
        "total_bedrooms": np.random.randint(1, 10, size=n),
        "population": np.random.randint(1, 15, size=n),
        "households": np.random.randint(1, 10, size=n),
        "ocean_proximity": np.random.choice(
            ["NEAR BAY", "NEAR OCEAN", "ISLAND", "INLAND"], size=n
        ),
    }

    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


@pytest.fixture(autouse=True)
def cleanup_files():
    for path in [config.train_housing_path, config.test_housing_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
    yield
    for path in [config.train_housing_path, config.test_housing_path]:
        if os.path.exists(path):
            shutil.rmtree(path)


def test_data_split(sample_housing_data):
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
        assert expected_cols.issubset(
            df.columns
        ), f"Missing expected columns in dataframe: {set(expected_cols) - set(df.columns)}"

        actual_cat_cols = {c for c in df.columns if c.startswith("ocean_proximity_")}

        assert (
            len(actual_cat_cols) >= len(expected_cat_cols) - 1
        ), f"Expected at least {len(expected_cat_cols)-1} dummy columns, found {len(actual_cat_cols)}"

    # Check files exist
    for fname in ["train.csv", "test.csv", "strat_train.csv", "strat_test.csv"]:
        assert os.path.exists(
            os.path.join(config.train_housing_path, fname)
        ) or os.path.exists(os.path.join(config.test_housing_path, fname))
