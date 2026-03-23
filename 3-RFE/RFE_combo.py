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
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Class that can remove features with a correlation higher than the given threshold value from a pd.DataFrame.

    The class is designed to work in a sklearn.pipeline.Pipeline object.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold: float = threshold
        self.to_drop_: Iterable[str] | None = None

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

    logger.info(f"Logging to: {log_filename}")

    return logger


def process_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: BaseEstimator,
    *,
    logger: logging.Logger,
) -> dict[str, Any]:
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
    suffix: str = file.suffix
    assert suffix == ".pkl", f"File must be '.pkl', but is {suffix}"

    with file.open("wb") as f:
        pickle.dump(results, f)


def train_test_split(
    splits_file: Path, fold_id: int, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    with splits_file.open("rb") as f:
        splits = pickle.load(f)

    train_idx, test_idx = splits[fold_id]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def filter_X(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(
        [
            "avg_atomic_quadrupole_principal_invariant_3",  # quadrupole principal invariant 3 features correlate highly with the invariant 2 features, so can drop them
            "max_atomic_quadrupole_principal_invariant_3",
            "molecular_quadrupole_principal_invariant_3",
            "avg_atomic_dipole_dipole_interaction",  # the dipole dipole interaction between atoms would physically not be that influential on the solubility, can drop it
        ],
        axis=1,
    )


def main() -> None:
    fold_id = int(sys.argv[1])

    n_cpus = 5

    BASE: Path = Path(__file__).parent
    file_name: str = Path(__file__).stem + f"_{fold_id}"

    df_file: Path = BASE.parent / "processed_dataset_wo_metals_w_even_more_qm2.csv"
    splits_file: Path = BASE / "splits.pkl"
    result_file: Path = BASE / "results" / (file_name + "_results.pkl")
    log_file: str = file_name + ".log"

    logger = setup_logger(BASE, log_file)

    pl_huber = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("remove_corr", CorrelationFilter(threshold=0.95)),
            ("transform", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scale", StandardScaler()),
            ("predict", HuberRegressor(epsilon=2.0, alpha=0.01, max_iter=1000)),
        ]
    )

    # gridsearch_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rfe_kf = KFold(n_splits=5, shuffle=True, random_state=40)

    # param_grid = {
    #     "estimator__predict__epsilon": [1.1, 1.35, 1.5, 2.0],
    #     "estimator__predict__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    # }

    def get_coef(estimator: BaseEstimator) -> np.ndarray:
        return estimator.named_steps["predict"].coef_

    rfe = RFECV(pl_huber, cv=rfe_kf, scoring="r2", n_jobs=n_cpus, importance_getter=get_coef, verbose=12)

    # search = GridSearchCV(estimator=rfe, param_grid=param_grid, cv=gridsearch_kf, scoring="r2", n_jobs=1, verbose=12)

    df = pd.read_csv(df_file)
    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    X = filter_X(X)
    y = df["solubility"]

    X_train, X_test, y_train, y_test = train_test_split(splits_file, fold_id, X, y)

    begin_time = time.time()

    scores: dict[str, Any] = process_fold(X_train, X_test, y_train, y_test, rfe, logger=logger)

    logger.info(f"Calculation finished in {time.time() - begin_time}s")

    save_results(scores, result_file)
    logger.info(f"Results saved to {result_file!s}")


if __name__ == "__main__":
    main()
