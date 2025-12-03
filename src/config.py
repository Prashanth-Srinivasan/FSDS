"""
Config.py
---------
This is a config file to store all static variables/locations that we need in this project.

"""

import os

download_path = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
housing_url = download_path + "datasets/housing/housing.tgz"
housing_raw_path = os.path.join(os.getcwd(), "data", "raw")
housing_processed_path = os.path.join(os.getcwd(), "data", "processed")
train_housing_path = os.path.join(housing_processed_path, "train")
test_housing_path = os.path.join(housing_processed_path, "test")
log_path = os.path.join(os.getcwd(), "logs")
artifacts_path = os.path.join(os.getcwd(), "artifacts")
output_path = os.path.join(os.getcwd(), "outputs")
