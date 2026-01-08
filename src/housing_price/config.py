"""
Config.py
---------
This is a config file to store all static variables/locations
that we need in this project.

"""

import os

DOWNLOAD_PATH = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_PATH + "datasets/housing/housing.tgz"
housing_raw_path = os.path.join(os.getcwd(), "data", "raw")
housing_processed_path = os.path.join(os.getcwd(), "data", "processed")
reference_housing_path = os.path.join(housing_processed_path, "drift_reference")
train_housing_path = os.path.join(housing_processed_path, "train")
test_housing_path = os.path.join(housing_processed_path, "test")
log_path = os.path.join(os.getcwd(), "logs")
artifacts_path = os.path.join(os.getcwd(), "artifacts")
output_path = os.path.join(os.getcwd(), "outputs")
BEST_MODEL = "lr.pkl"
FEATURE_ORDER = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "income_cat",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "ocean_proximity_LESS_THAN_1H_OCEAN",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR_BAY",
    "ocean_proximity_NEAR_OCEAN",
    "median_house_value",
]
DRIFT_THRESHOLD = 0.25
