from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ml_enhance import get_topology_features

SelectionMode = Literal["two_stage", "full"]


class FeatureImportance:
    """Aggregate feature importance across outer CV folds.

    Computes:
    - Selection frequency (top-N presence)
    - Mean importance (when selected)
    """

    def __init__(self, results_df: pd.DataFrame, *, provided_FI: dict[int, pd.Series] | None = None) -> None:
        if "estimator" not in results_df.columns:
            raise ValueError("results_df must contain 'estimator' column")

        self.df: pd.DataFrame = results_df
        self.n_outer_folds: int = len(results_df)
        self.provided_FI: dict[int, pd.Series] | None = provided_FI

        if self.provided_FI is not None:
            missing = set(self.df.index) - set(self.provided_FI.keys())
            if missing:
                raise ValueError(f"Missing FI for folds: {missing}")

    def get_feature_importance(
        self,
        num_features: int = 20,
        *,
        mode: SelectionMode = "two_stage",
        weight_by_score: bool = False,
    ) -> pd.DataFrame:
        frequency: dict[str, int] = defaultdict(int)
        importance_dict: dict[str, list[float]] = defaultdict(list)

        for fold_id, row in self.df.iterrows():
            if self.provided_FI is not None:
                fi_series = self.provided_FI[fold_id]

            else:
                estimator = row["estimator"]
                model, features, support = self._unwrap_estimator(estimator)

                importances = self._get_importance(model)

                if features is None:
                    features = np.array([f"f{j}" for j in range(len(importances))])

                if support is not None:
                    # RFECV selected features only
                    features = features[support]

                fi_series = pd.Series(importances, index=features)

            if mode == "two_stage":
                selected = fi_series.abs().sort_values(ascending=False).head(num_features)
            elif mode == "full":
                selected = fi_series.abs().sort_values(ascending=False)
            else:
                raise ValueError("Unknown mode")

            weight: float = 1.0
            if weight_by_score and "test_r2" in row.index:
                weight = float(row["test_r2"])

            for feat in selected.index:
                frequency[feat] += 1
                importance_dict[feat].append(fi_series[feat] * weight)

        results: list[dict[str, float | str]] = [
            {
                "feature": feat,
                "frequency": frequency[feat] / self.n_outer_folds,
                "mean_importance": np.mean(importance_dict[feat]).astype(float),
                "std_importance": np.std(importance_dict[feat]).astype(float),
            }
            for feat in frequency
        ]

        df = pd.DataFrame(results)
        df["score"] = df["frequency"] * df["mean_importance"].abs()
        df["std_score"] = df["frequency"] * df["std_importance"].abs()

        self.fi_df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        return self.fi_df

    def plot(
        self,
        num_features: int = 20,
        *,
        save_fig: bool = False,
        fig_name: str | None = None,
        color: str = "tab:blue",
    ) -> None:
        """Plot the feature importance.

        params:
            num_features (int): the top "n" features that are plotted
            save_fig (bool): flag that tells the function whether the figure should be saved or not
            fig_name (str): file name of the figure to be saved
            color (str): name of the color used for the QM features
        """
        topology_features: list[str] = get_topology_features()

        num_features = min(num_features, self.fi_df.shape[0])

        try:
            df = self.fi_df.head(num_features)

        except AttributeError:
            print("Feature importance is not calculated, try running 'get_feature_importance' first.")

        colors = [color if feature not in topology_features else "grey" for feature in df["feature"]]
        alphas = [1 if feature not in topology_features else 0.6 for feature in df["feature"]]

        plt.figure(figsize=(8, 6))
        bars = plt.barh(df["feature"], df["score"].to_numpy(), color=colors, xerr=df["std_score"].to_numpy(), capsize=3)

        for bar, alpha in zip(bars, alphas, strict=True):
            bar.set_alpha(alpha)

        legend_elements = [
            Patch(facecolor=color, label="QM feature"),
            Patch(facecolor="grey", label="Topological feature", alpha=0.6),
        ]

        ax = plt.gca()
        yticks = ax.get_yticklabels()

        for tick in yticks:
            tick.set_size(12)
            if tick.get_text() not in topology_features:
                tick.set_fontweight("bold")
                tick.set_color(color)

        plt.gca().invert_yaxis()

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.xlabel("Weighted Feature importance", fontsize=16)
        plt.title(f"Top {num_features} most important features", fontsize=16)
        plt.legend(handles=legend_elements, frameon=False, loc="lower right", fontsize=12)
        plt.tight_layout()

        if save_fig:
            assert fig_name is not None, "No name for the figure is provided."
            plt.savefig(fig_name, dpi=300)

        plt.show()

    def _unwrap_estimator(self, estimator: BaseEstimator) -> tuple[BaseEstimator, np.ndarray | None, np.ndarray | None]:
        """Returns:

        - model
        - feature names
        - support mask (if RFECV)
        """
        support = None

        # Handle SearchCV
        if hasattr(estimator, "best_estimator_"):
            estimator = estimator.best_estimator_

        # Handle RFECV
        if hasattr(estimator, "estimator_") and hasattr(estimator, "support_"):
            support = estimator.support_
            estimator = estimator.estimator_

        # Handle Pipeline
        if isinstance(estimator, Pipeline):
            try:
                features = estimator[:-1].get_feature_names_out()
            except Exception:
                features = None
            model = estimator[-1]
        else:
            model = estimator
            features = None

        return model, features, support

    def _get_importance(self, model: BaseEstimator) -> np.ndarray:
        """Extract feature importance from model."""
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_)

        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            return coef.ravel()

        raise ValueError(f"Model of type {type(model)} does not provide feature importance")
