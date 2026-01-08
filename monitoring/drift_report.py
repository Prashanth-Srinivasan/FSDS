"""
This script performs data drift monitoring for the housing price model
using EvidentlyAI.

The script loads:
- Reference dataset
- Current dataset

It computes feature-level data drift metrics using EvidentlyAI's
DataDriftPreset and generates an HTML drift report.

If the overall drift score exceeds a predefined threshold, the script:
- Logs an alert message
- Exits with a non-zero status code to fail the CI/CD pipeline

Outputs:
- HTML drift report saved under the "reports" directory

This script is designed to be executed:
- Automatically in GitLab CI/CD as a monitoring gate before deployment

Usage
-----
Run from terminal:

    python monitoring/drift_report.py

"""

import os
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from housing_price import config  # type: ignore

# Paths
REFERENCE_PATH = os.path.join(config.reference_housing_path, "drift_reference.csv")
CURRENT_PATH = os.path.join(config.train_housing_path, "train.csv")
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "data_drift_report.html")
DRIFT_THRESHOLD = config.DRIFT_THRESHOLD


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    reference_df = pd.read_csv(REFERENCE_PATH)
    current_df = pd.read_csv(CURRENT_PATH)

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    report_result = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    report_result.save_html(REPORT_PATH)
    print(f"Drift report saved at: {REPORT_PATH}")

    report_dict = report_result.dict()

    drifted_ratio = report_dict["metrics"][0]["value"]["share"]

    print(f"Drifted features ratio: {drifted_ratio:.2f}")

    if drifted_ratio > DRIFT_THRESHOLD:
        print("ALERT: Data drift threshold exceeded!")
        sys.exit(1)
    else:
        print("Drift within acceptable limits")


if __name__ == "__main__":
    main()
