"""
ingest_data.py
--------------
This Script will download / load (../data/housing_raw) the housing data.
The data is then processeed and split into train and test set.
The data is finally saved to data/housing_processed folder.

Inputs (via argparse arguments)
-------------------------------
-tr / --train_data_path : str
    Directory where the processed training dataset will be saved.
    Default: config.train_housing_path

-ts / --test_data_path : str
    Directory where the processed test dataset will be saved.
    Default: config.test_housing_path

-v / --verbose : str
    Whether to also print logs to the console.
    Allowed: Y, N
    Default: "Y"

-ll / --log_level : str
    Logging verbosity level.
    Allowed: DEBUG, INFO, CRITICAL, ERROR, WARNING
    Default: "INFO"

-lp / --log_file_path : str
    Path to the directory where the log file will be stored.
    Default: config.log_path


Outputs
-------
All files are saved inside: ../data/processed

The script will generate train and test files:
- /train/train.csv
- /train/strat_train.csv
- /test/test.csv
- /test/strat_test.csv

Usage
-----
Run from terminal:

    python ingest_data.py # fix

"""

import argparse
import os
import tarfile
from datetime import datetime
from urllib.error import URLError
import numpy as np
import pandas as pd
from six.moves import urllib  # type: ignore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from src.housing_price import config
from src.housing_price.utils import get_logger


def fetch_housing_data(
    housing_url=config.HOUSING_URL,
    housing_path=config.housing_raw_path,
):
    """
    This function takes parameters housing_url and housing_path
    Downloads the data from housing_url, extracts the .tgz file and saves
    it to housing_path

    Args:
        housing_url (str, optional):URL from which housing data is downloaded.
        Defaults: config.HOUSING_URL.
        housing_path (str, optional): output path to store the downloaded data.
        Defaults: config.housing_raw_path.

    Return:
        None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=config.housing_raw_path):
    """
    This function use to load the data and returns csv

    Args:
        housing_path (str, optional):
        path to housing csv file (raw)

        Defaults to config.housing_raw_path.

    Returns:
        csv: dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def data_train_test_split(housing_data):
    """
    Preprocessing of data and split into train and test dataset
    traget_variable: median_house_value

    Args:
        housing_data (dataframe, optional): csv file to process.
        Defaults to None.
    Returns:
        train_set,test_set: Dataframes
    """

    housing_data["income_cat"] = pd.cut(
        housing_data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    housing_labels = housing_data["median_house_value"].copy()
    housing_data = housing_data.drop("median_house_value", axis=1)

    imputer = SimpleImputer(strategy="median")

    housing_num = housing_data.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    temp_x = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        temp_x, columns=housing_num.columns, index=housing_data.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing_data[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    housing_final = housing_prepared.join(housing_labels)

    train_data, test_data = train_test_split(
        housing_final, test_size=0.2, random_state=42
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(
        housing_final, housing_final["income_cat"]
    ):
        strat_train_data = housing_final.loc[train_index]
        strat_test_data = housing_final.loc[test_index]

    os.makedirs(config.train_housing_path, exist_ok=True)
    os.makedirs(config.test_housing_path, exist_ok=True)
    train_data.to_csv(
        os.path.join(config.train_housing_path, "train.csv"),
        index=False,
    )
    test_data.to_csv(
        os.path.join(config.test_housing_path, "test.csv"),
        index=False,
    )
    strat_train_data.to_csv(
        os.path.join(config.train_housing_path, "strat_train.csv"),
        index=False,
    )
    strat_test_data.to_csv(
        os.path.join(config.test_housing_path, "strat_test.csv"),
        index=False,
    )

    return train_data, test_data, strat_train_data, strat_test_data


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tr",
        "--train_data_path",
        help="test data path",
        default=config.train_housing_path,
    )
    parser.add_argument(
        "-ts",
        "--test_data_path",
        help="test data path",
        default=config.test_housing_path,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="whether to output logs?",
        default="Y",
        choices=["Y", "N", "y", "n"],
    )
    parser.add_argument(
        "-ll",
        "--log_level",
        help="level of logs that required",
        default="INFO",
        choices=["DEBUG", "INFO", "CRITICAL", "ERROR", "WARNING"],
    )
    parser.add_argument(
        "-lp",
        "--log_file_path",
        help="Log file path",
        default=config.log_path,
    )

    args = parser.parse_args()

    logger = get_logger("ingest_data.py", args.log_file_path, console=True)

    print("passed args")

    try:
        logger.info("fetching housing data from %s", config.HOUSING_URL)
        fetch_housing_data()
        logger.info("fetching data completed")
    except URLError as e:
        logger.warning(
            "Download failed, switching to local copy. Error: %s",
            e,
            exc_info=True,
        )

    housing = load_housing_data()
    logger.info("train_data_path: %s", args.train_data_path)
    logger.info("test_data_path: %s", args.test_data_path)
    logger.info("fetched data size %s", housing.shape)
    logger.info("starting train-test split with test size 0.2")
    train_set, test_set, strat_train_set, strat_test_set = data_train_test_split(
        housing
    )
    logger.info(
        "completed train-test split with train_size %s and test_size %s",
        train_set.shape,
        test_set.shape,
    )

    end = datetime.now()
    exec_time = round((end - start).seconds, 4)
    logger.info(
        "execution time for ingest_data script %s s",
        exec_time,
    )
