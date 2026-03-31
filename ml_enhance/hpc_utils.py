import logging
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class Files:
    SPLITS_FILE: Path
    RESULTS_FILE: Path


class Score:
    """Class that provides evaluation metrics for model scoring."""

    def __init__(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        self.y_true = y_true
        self.y_pred = y_pred

    @property
    def r2(self) -> float:
        return r2_score(self.y_true, self.y_pred)

    @property
    def MSE(self) -> float:
        return mean_squared_error(self.y_true, self.y_pred)


class LoggerWriter:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self) -> None:
        pass


def setup_logger(DATA_PATH: Path, filename: str) -> logging.Logger:
    # Create log directory
    log_dir = DATA_PATH / "logs"
    log_dir.mkdir(exist_ok=True)

    assert filename.endswith(".log"), "filename should end with '.log'."

    # Log filename
    log_filename: Path = log_dir / filename

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    )

    # Main logger
    logger = logging.getLogger(__name__)

    logging.captureWarnings(True)

    sys.stdout = LoggerWriter(logger, logging.INFO)

    logger.info(f"Logging to: {log_filename}")

    return logger


def process_job(
    FILES: Files, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, fold_id: int, logger: logging.Logger
) -> None:
    X_features: list[str] = X.columns.to_list()

    logger.info(f"X consists of {len(X_features)} features: {X_features}")

    X_train, X_test, y_train, y_test = custom_train_test_split(FILES.SPLITS_FILE, fold_id, X, y)

    begin_time = time.time()

    scores: dict[str, Any] = process_fold(X_train, X_test, y_train, y_test, model, logger=logger)

    logger.info(f"Calculation finished in {time.time() - begin_time}s")

    save_results(scores, FILES.RESULTS_FILE)
    logger.info(f"Results saved to {FILES.RESULTS_FILE!s}")


def process_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: BaseEstimator,
    *,
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info("Begin model fit")
    # Fit
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    logger.info(f"Model fit took: {fit_time}s")

    # Score
    start_score = time.time()
    y_pred: np.ndarray = model.predict(X_test)
    predict_time = time.time() - start_score

    logger.info(f"Model predict took: {predict_time}s")

    y_train_pred: np.ndarray = model.predict(X_train)

    train_score = Score(y_train, y_train_pred)
    test_score = Score(y_test, y_pred)

    logger.info(f"Train r2 score: {train_score.r2}")
    logger.info(f"Test r2 score: {test_score.r2}")

    return {
        "model": model,
        "train_r2": train_score.r2,
        "test_r2": test_score.r2,
        "train_MSE": train_score.MSE,
        "test_MSE": test_score.MSE,
    }


def save_results(results: dict[str, Any], file: Path) -> None:
    file.parent.mkdir(exist_ok=True)
    assert file.suffix == ".pkl", f"File must be '.pkl', but is {file.suffix}"

    with file.open("wb") as f:
        pickle.dump(results, f)


def custom_train_test_split(
    splits_file: Path, fold_id: int, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    with splits_file.open("rb") as f:
        splits = pickle.load(f)

    train_idx, test_idx = splits[fold_id]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test
