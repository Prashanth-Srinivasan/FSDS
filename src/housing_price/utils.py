"""
utils.py
--------------
This script contains utility functions that are used across the modules.

"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str, level: str = "INFO", console: bool = True):
    """
    Create and return a logger that writes to a file and optionally to console.

    Parameters
    ----------
    name : str
        Name of the logger (e.g. __name__ or script name).
    log_dir : str, optional
        Directory where log files are saved (default = "logs").
    level : str, optional
        Logging level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    console : bool, optional
        If True = log to console + file.
        If False = log only to file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d}.log")

    level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
