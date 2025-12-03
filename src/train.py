"""
train.py
--------------
This Script will train models on the provided training data.
The data is then processeed and split into train and test set.
The data is finally saved to data/housing_processed folder.

Inputs (via argparse arguments)
-------------------------------

-tr / --train_data_path : str
    Path to the processed training dataset file.
    Default: <config.train_housing_path>/train.csv

-m / --model_name : str
    Name of the model to train.
    Allowed: lr, dtr, rfr_rs, rfr_gs
    Default: "lr"

-of / --output_folder : str
    Directory where the trained model artifacts will be saved.
    Default: config.artifacts_path

-lp / --log_file_path : str
    Directory where the log file should be stored.
    Default: config.log_path

-v / --verbose : str
    Whether logs should also be printed.
    Allowed: Y, N
    Default: "Y"

-ll / --log_level : str
    Logging verbosity level.
    Allowed: DEBUG, INFO, CRITICAL, ERROR, WARNING
    Default: "INFO"

Outputs
-------
All files are saved inside: ../artifacts

The script will generate model pickle files

Usage
-----
Run from terminal:

    python train.py # fix

"""

import argparse
import os
from scipy.stats import randint
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.utils import get_logger


def train_model(train_data_set_path, model_name, logger):
    """This function is use to train model on the given data and
    returns trained_model pickle files to artifacts.

    Args:
        train_data_set_path (str):
        train data set file path (.csv).

        model_name (str):
        name that you need to pass
        [lr,dtr,rfr_rs,rfr_gs]
        Defaults to None.

    Returns:
        trained_model pickle files
    """

    train_data = pd.read_csv(train_data_set_path)
    y_label = train_data["median_house_value"]
    x_label = train_data.drop("median_house_value", axis=1)

    logger.info("train data extracted {}".format(train_data.shape))
    logger.info("x_data {}, Y_data {}".format(x_label.shape, y_label.shape))

    if x_label.shape[0] == y_label.shape[0]:
        logger.info("Passed data check")
    else:
        logger.info("failed data check")

    logger.info("model_selected : {}".format(model_name))

    if model_name == "lr":
        lr = LinearRegression()
        logger.info("Linear model learning starts")
        lr.fit(x_label, y_label)
        logger.info("linear model learning end")
        return lr
    elif model_name == "dtr":
        dtr = DecisionTreeRegressor(random_state=42)
        logger.info("decision tree model train initiated")
        dtr.fit(x_label, y_label)
        logger.info("decision tree model train end")
        return dtr
    elif model_name == "rfr_rs":
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        logger.info("random forest - random search model train start")
        rnd_search.fit(x_label, y_label)
        logger.info("random forest - random search model train end")
        return rnd_search
    elif model_name == "rfr_gs":
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {
                "n_estimators": [3, 10, 30],
                "max_features": [2, 4, 6, 8],
            },
            # then try 6 (2×3) combinations with bootstrap set as False
            {
                "bootstrap": [False],
                "n_estimators": [3, 10],
                "max_features": [2, 3, 4],
            },
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        logger.info("random forest - grid search model train start")
        grid_search.fit(x_label, y_label)
        logger.info("random forest - grid search model train end")
        return grid_search
    else:
        return None


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tr",
        "--train_data_path",
        help="train data path ",
        default=os.path.join(config.train_housing_path, "train.csv"),
    )
    parser.add_argument(
        "-m",
        "--model_name",
        help="(lr),(dtr),(rfr_rs),(rfr_gs)",
        default="lr",
        choices=["lr", "dtr", "rfr_rs", "rfr_gs"],
    )
    parser.add_argument(
        "-of",
        "--output_folder",
        help="path to output model folder",
        default=config.artifacts_path,
    )
    parser.add_argument(
        "-lp",
        "--log_file_path",
        help="Log file path",
        default=config.log_path,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="whether to output logs",
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

    args = parser.parse_args()

    logger = get_logger("train.py", args.log_file_path, console=True)

    logger.info("Data Training Starts")
    logger.info("train_file_loc:{}".format(args.train_data_path))
    logger.info("model_selection:{}".format(args.model_name))

    model = train_model(args.train_data_path, args.model_name, logger)

    logger.info("data training end")

    final_model = args.model_name + "_" + str(datetime.now().date()) + ".pkl"

    joblib.dump(model, os.path.join(args.output_folder, final_model))
    logger.info(
        "model_output_loc:{}".format(os.path.join(args.output_folder, final_model))
    )
    logger.info("model saved completed")
    end = datetime.now()
    logger.info(
        "execution time for ingest_data script {}s".format(
            round((end - start).seconds, 4)
        )
    )
