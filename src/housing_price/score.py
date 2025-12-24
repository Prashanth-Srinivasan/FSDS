"""
score.py
--------------
This script is use to predict the model evaluation on test data.

Inputs (via argparse arguments)
-------------------------------
-ts / --test_data_path : str
    Full path to the test dataset CSV file.
    Default: config.train_housing_path/test.csv

-m / --model_name : str
    Name of the model to train.
    Allowed: lr, dtr, rfr_rs, rfr_gs
    Default: "lr"

-of / --model_folder : str
    Directory where the trained model will be saved.
    Default: config.artifacts_path

-lp / --log_file_path : str
    Path to the directory where the log file will be stored.
    Default: config.log_path

-v / --verbose : str
    Whether logs should also be printed.
    Allowed: Y, N, y, n
    Default: "Y"

-ll / --log_level : str
    Logging verbosity level.
    Allowed: DEBUG, INFO, CRITICAL, ERROR, WARNING
    Default: "INFO"


Outputs
-------
All prediction files are saved inside: ../output

The script will generate preditction files:
- lr_<date>_predictions.csv
- dtr_<date>_predictions.csv
- rfr_rs_<date>_predictions.csv
- rfr_gs_<date>_predictions.csv

Usage
-----
Run from terminal:

    python score.py # fix

"""

import argparse
import os
from datetime import datetime

import joblib
import mlflow  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.housing_price import config
from src.housing_price.utils import get_logger


def predict_result(model_path, test_data_path, logger_fn):
    """This function is use to predict result on the test_data and
    returns predicted_result

    Args:
        test_data_path (str, optional):
        test data set file path (.csv).

        model_path (str):
        trained model file.

        logger (log_obj):
        it will help to log data in the log_file.

    Returns:
        predicted_data:Dataframe
    """
    data = pd.read_csv(test_data_path)
    y_test = data["median_house_value"]
    x_test = data.drop("median_house_value", axis=1)

    logger_fn.info("data extracted %s", data.shape)

    if x_test.shape[0] == y_test.shape[0]:
        logger_fn.info("test data check passed")
    else:
        logger_fn.info("test data check failed")

    model = joblib.load(model_path)

    model_name = model_path.split("/")[-1].split(".pkl")[0]

    logger_fn.info("model picked: %s", model_name)

    pred_result = model.predict(x_test)
    logger_fn.info("prediction done for x_test")

    if "rfr" in model_name:
        cvres = model.cv_results_
        best_index = np.argmax(cvres["mean_test_score"])
        result_mse = -cvres["mean_test_score"][best_index]
        result_rmse = np.sqrt(result_mse)
        result_mae = mean_absolute_error(y_test, pred_result)
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            logger_fn.info(
                "model_name: %s, rmse_score: %s, params: %s",
                model_name,
                np.sqrt(-mean_score),
                params,
            )

    else:
        result_mse = mean_squared_error(y_test, pred_result)
        result_rmse = np.sqrt(result_mse)
        result_mae = mean_absolute_error(y_test, pred_result)
        logger_fn.info(
            "model_name: %s, rmse_score: %s, mae_score: %s",
            model_name,
            result_rmse,
            result_mae,
        )

    metrics_dic = {"mse": result_mse, "rmse": result_rmse, "mae": result_mae}
    data[model_name + "_predictions"] = pred_result
    return data, metrics_dic


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ts",
        "--test_data_path",
        help="test data path ",
        default=os.path.join(config.test_housing_path, "test.csv"),
    )
    parser.add_argument(
        "-m",
        "--model_name",
        help="model name required",
    )
    parser.add_argument(
        "-of",
        "--model_folder",
        help="path to model folder",
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

    parent_id = os.environ.get("PARENT_MLFLOW_RUN_ID")
    with mlflow.start_run(run_id=parent_id, nested=True):

        mlflow.log_param("test_data_path", args.test_data_path)

        logger = get_logger("score.py", args.log_file_path, console=True)

        logger.info("Scoring Starts")
        logger.info("log_file_location: %s", args.log_file_path)
        logger.info("test_file_loc: %s", args.test_data_path)
        logger.info("model_folder_location: %s", args.model_folder)
        logger.info("model_selection: %s", args.model_name)
        logger.info("output_path: %s", config.output_path)

        predicted_data, metrics = predict_result(
            os.path.join(args.model_folder, args.model_name),
            args.test_data_path,
            logger,
        )

        # for metric_name, metric_value in metrics.items():
        #     mlflow.log_metric(metric_name, metric_value)
        metric_key = list(metrics.keys())
        metric_value = list(metrics.values())
        mlflow.log_metric(metric_key[0], metric_value[0])
        mlflow.log_metric(metric_key[1], metric_value[1])
        mlflow.log_metric(metric_key[2], metric_value[2])

        os.makedirs(config.output_path, exist_ok=True)

        pred_file_path = os.path.join(
            config.output_path,
            args.model_name.split(".pkl")[0] + "_predictions.csv",
        )

        predicted_data.to_csv(
            os.path.join(
                config.output_path,
                args.model_name.split(".pkl")[0] + "_predictions.csv",
            ),
            index=False,
        )

        logger.info("prediction result saved: %s", pred_file_path)

        logger.info("Scoring Ends")
        end = datetime.now()
        exec_time = round((end - start).seconds, 4)
        logger.info("execution time for ingest_data script %s s", exec_time)
