import logging
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from ml_enhance import CorrelationFilter


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
) -> tuple[BaseEstimator, Score, Score]:
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

    return model, train_score, test_score


def custom_cross_validate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: BaseCrossValidator,
    logger: logging.Logger,
) -> dict[str, Any]:
    p_process_fold = partial(process_fold, logger=logger)

    models: list[BaseEstimator] = []
    train_r2s: list[float] = []
    test_r2s: list[float] = []
    train_MSEs: list[float] = []
    test_MSEs: list[float] = []
    for idx, (train_idxs, test_idxs) in enumerate(outer_cv.split(X, y), start=1):
        logger.info(f"=== OUTER FOLD {idx}/{outer_cv.get_n_splits()} ===")
        X_train, X_test = X.iloc[train_idxs], X.iloc[test_idxs]
        y_train, y_test = y.iloc[train_idxs], y.iloc[test_idxs]

        model_copy = clone(model)

        return_model, train_scores, test_scores = p_process_fold(X_train, X_test, y_train, y_test, model_copy)

        models.append(return_model)

        train_r2s.append(train_scores.r2)
        train_MSEs.append(train_scores.MSE)

        test_r2s.append(test_scores.r2)
        test_MSEs.append(test_scores.MSE)

    return {
        "estimator": np.array(models),
        "train_r2": np.array(train_r2s),
        "test_r2": np.array(test_r2s),
        "train_MSE": np.array(train_MSEs),
        "test_MSE": np.array(test_MSEs),
    }


def save_results(results: dict[str, Any], file: Path) -> None:
    suffix: str = file.suffix
    assert suffix == ".pkl", f"File must be '.pkl', but is {suffix}"

    with file.open("wb") as f:
        pickle.dump(results, f)


def main() -> None:
    df = pd.read_csv("data/processed_dataset_wo_metals_w_even_more_qm2.csv")
    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    y = df["solubility"]

    X = X.drop(
        [
            "avg_atomic_quadrupole_principal_invariant_3",  # quadrupole principal invariant 3 features correlate highly with the invariant 2 features, so can drop them
            "max_atomic_quadrupole_principal_invariant_3",
            "molecular_quadrupole_principal_invariant_3",
            "avg_atomic_dipole_dipole_interaction",  # the dipole dipole interaction between atoms would physically not be that influential on the solubility, can drop it
        ],
        axis=1,
    )

    logger = setup_logger(Path(), "test.log")

    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

    model = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("remove_corr", CorrelationFilter(threshold=0.95)),
            ("transform", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scale", StandardScaler()),
            ("predict", HuberRegressor(epsilon=2.0, alpha=0.01, max_iter=1000)),
        ]
    )

    begin_time = time.time()

    scores: dict[str, Any] = custom_cross_validate(model, X, y, outer_kf, logger)

    logger.info(f"Calculation finished in {time.time() - begin_time}s")

    result_file = Path("test_scores.pkl")
    save_results(scores, result_file)
    logger.info(f"Results saved to {result_file!s}")


if __name__ == "__main__":
    main()
