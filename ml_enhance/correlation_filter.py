from collections.abc import Iterable
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Class that can remove features with a correlation higher than the given threshold value from a pd.DataFrame.

    The class is designed to work in a sklearn.pipeline.Pipeline object.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold: float = threshold
        self.to_drop_: list[str] = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Self:
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
