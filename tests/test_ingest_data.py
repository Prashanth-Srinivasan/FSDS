import os

import pytest
from src import config
from src.ingest_data import load_housing_data


def test_data_file_exists():
    assert os.path.exists(
        config.housing_raw_path
    ), f"Config path missing: {config.housing_raw_path}"


def test_loaded_data_has_rows():
    df = load_housing_data()
    assert len(df) > 0
