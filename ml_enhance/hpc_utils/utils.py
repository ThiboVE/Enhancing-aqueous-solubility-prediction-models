import logging
import pickle
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score


class Files:
    def __init__(self, running_file: str, filename: str) -> None:
        self.running_file: Path = Path(running_file)
        self.filename: str = filename + "_rerun"

        self.base = self.running_file.parent

        self.output_dir = Path("/data/gent/489/vsc48953/ML_enhance") / (self.running_file.stem + "_rerun") / "results"
        self.log_dir = self.base / "logs"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_df_file(self, df_file_name: str) -> Path:
        return self.base.parent / df_file_name

    @property
    def SPLITS_FILE(self) -> Path:
        return Path("../splits.pkl")

    @property
    def RDKIT_FILE(self) -> Path:
        return self.base.parent / "rdkit_feature_names.json"

    @property
    def LOG_FILE(self) -> Path:
        return self.log_dir / f"{self.filename}.log"

    @property
    def LIGHTNING_LOG_DIR(self) -> Path:
        return self.log_dir / f"{self.filename}_log"

    @property
    def RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_results.pkl"

    @property
    def RESULTS_FILE_JSON(self) -> Path:
        return self.output_dir / f"{self.filename}_results.json"

    @property
    def RESULTS_FILE_MODEL(self) -> Path:
        return self.output_dir / f"{self.filename}_model.pt"

    @property
    def PFI_RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_PFI_results.csv"

    @property
    def SHAP_RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_SHAP_results.csv"


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Class that can remove features with a correlation higher than the given threshold value from a pd.DataFrame.

    The class is designed to work in a sklearn.pipeline.Pipeline object.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold: float = threshold
        self.to_drop_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = np.arange(X.shape[1])
            X = pd.DataFrame(X)

        corr_matrix = X.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.to_drop_ = np.array([column for column in upper.columns if any(upper[column] >= self.threshold)])
        self.support_ = np.array([column not in self.to_drop_ for column in upper.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        return X.drop(columns=self.to_drop_, errors="ignore").to_numpy()

    def get_support(self, *, indices: bool = False) -> np.ndarray:
        """Return boolean mask or indices of kept features."""
        mask = self.support_
        return np.where(mask)[0] if indices else mask

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        """This is the magic method that lets the pipeline track names."""
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features)[self.get_support()]


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


def setup_logger(log_file: Path) -> logging.Logger:
    assert log_file.suffix == ".log", "filename should end with '.log'."

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )

    # Main logger
    logger = logging.getLogger(__name__)

    logging.captureWarnings(True)

    # Attach handlers to Optuna logger as well
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.INFO)
    # Avoid duplicate logs if handlers already exist
    if not optuna_logger.handlers:
        optuna_logger.addHandler(logging.FileHandler(log_file, mode="a"))
        optuna_logger.addHandler(logging.StreamHandler())

    sys.stdout = LoggerWriter(logger, logging.INFO)

    logger.info(f"Logging to: {log_file}")

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
