"""
main.py
-------

This script orchestrates the full machine learning pipeline for housing data.
It runs data ingestion, model training, and scoring sequentially using
subprocess calls to the respective scripts. The entire pipeline is tracked
under a single parent MLflow run, with nested child runs for ingestion,
training, and scoring.

Inputs (via argparse arguments)
-------------------------------
-m / --model_name : str
    Model to train and score.
    Allowed: lr, dtr, rfr_rs, rfr_gs
    Default: "lr"

-n / --run_name : str
    Name of the parent MLflow run under which all child runs will be nested.
    Default: full_pipeline_<current_timestamp>


Outputs
-------
The script orchestrates the following:
1. Data ingestion via ingest_data.py
2. Model training via train.py
3. Model scoring via score.py

All trained model artifacts and prediction outputs are saved in their
respective folders as configured in train.py and score.py.

Usage
-----
Run from terminal:

    python src/housing_price/main.py -m lr
"""

import argparse
import os
import subprocess
from datetime import datetime

import mlflow  # type: ignore

from src.housing_price import config
from src.housing_price.utils import get_logger

logger = get_logger("main.py", config.log_path, console=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model_name",
    help="(lr),(dtr),(rfr_rs),(rfr_gs)",
    default="lr",
    choices=["lr", "dtr", "rfr_rs", "rfr_gs"],
)
parser.add_argument(
    "-n",
    "--run_name",
    help="Provide the run name",
    default=f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
args = parser.parse_args()
model_to_run = args.model_name

experiment_name = args.run_name

existing_exp = mlflow.get_experiment_by_name(experiment_name)

if existing_exp is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment: {experiment_name}, id: {experiment_id}")
else:
    experiment_id = existing_exp.experiment_id
    print(f"Using existing experiment: {experiment_name}, id: {experiment_id}")

with mlflow.start_run(
    run_name=f"{model_to_run}_{experiment_name}",
    experiment_id=experiment_id,
) as parent_run:
    parent_run_id = parent_run.info.run_id
    os.environ["PARENT_MLFLOW_RUN_ID"] = parent_run_id
    logger.info("Parent MLflow run started: %s", parent_run.info.run_id)

    # Ingest Data
    logger.info("Starting data ingestion")
    subprocess.run(["python", "-m", "src.housing_price.ingest_data"], check=True)
    logger.info("Data ingestion completed")

    # Train
    logger.info("Training model: %s", model_to_run)
    subprocess.run(
        ["python", "-m", "src.housing_price.train", "-m", model_to_run], check=True
    )
    logger.info("%s training completed", model_to_run)

    # Score
    logger.info("Scoring model: %s", model_to_run)
    subprocess.run(
        [
            "python",
            "-m",
            "src.housing_price.score",
            "-m",
            f"{model_to_run}.pkl",
        ],
        check=True,
    )
    logger.info("%s scoring completed", model_to_run)

    logger.info("Pipeline completed successfully")
