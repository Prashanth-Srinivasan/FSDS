# Price Elasticity Modeling & MLflow Tracking

This project implements an end-to-end workflow for training, evaluating, and registering machine learning models for price elasticity estimation.  
It includes:

- Data loading and preprocessing  
- Model training  
- Model evaluation  
- MLflow experiment tracking  
- MLflow Model Registry integration  
- A scoring pipeline for inference  
- A project-structured Python package with `uv` for environment management  

--------------------------------------------------

## Project Structure
```
.
├── .flake8
├── .gitignore
├── .gitlab-ci.yml
├── .isort.cfg
├── .python-version
├── Notebooks
│   └── nonstandardcode.ipynb
├── README.md
├── artifacts
├── data
│   ├── processed
│   │   ├── test
│   │   └── train
│   └── raw
├── logs
├── outputs
├── pyproject.toml
├── src
│   ├── __init__.py
│   └── housing_price
│       ├── __init__.py
│       ├── config.py
│       ├── ingest_data.py
│       ├── main.py
│       ├── score.py
│       ├── train.py
│       └── utils.py
├── tests
│   ├── test_ingest_data.py
│   ├── test_score.py
│   └── test_train.py
└── uv.lock
```
--------------------------------------------------

## Features

### Data Ingestion
- Reads raw housing dataset.
- Performs stratified train–test split based on income category.
- Saves processed datasets into the artifacts directory.

### Model Training
- Supports the following algorithms:
- Linear Regression (lr)
- Decision Tree Regressor (dtr)
- Random Forest (RandomizedSearch) (rfr_rs)
- Random Forest (GridSearch) (rfr_gs)
- Logs parameters, metrics, and artifacts to MLflow.
- Optionally registers the model in MLflow Model Registry.

### Model Scoring
- Loads the trained model from artifacts.
- Predicts on the test dataset.
- Computes evaluation metrics (RMSE, MAE, MSE).
- Logs metrics to MLflow.
- Saves prediction files to the output folder.

### Testing
- Includes Pytest-based unit tests.
- Supports marking tests (example: @pytest.mark.local_only).
- Provides fixtures for sample housing datasets.

### CI/CD Readiness
- Supports linting, formatting, and testing in automated pipelines.
- Compatible with GitLab CI/CD or GitHub Actions.

--------------------------------------------------
## Run Individual Modules (**uv**)

### Ingest Data
```
uv run src/housing_price/ingest_data.py
```
### Training
```
uv run src/housing_price/train.py \
    --train_data_path data/processed/train.csv \
    --model_name lr \
    --save_dir artifacts
```
### Scoring
```
uv run src/housing_price/score.py \
    --test_data_path data/processed/test.csv \
    --model_name lr.pkl \
    --model_folder artifacts
```
### Run Entire Pipeline
```
uv run src/housing_price/main.py \
    --model_name lr \
    --run_name run_001
```
--------------------------------------------------