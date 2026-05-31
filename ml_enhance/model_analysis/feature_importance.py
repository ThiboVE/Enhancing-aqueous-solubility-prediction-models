import re
from collections import defaultdict
from typing import Literal

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
        # if "estimator" not in results_df.columns:
        #     raise ValueError("results_df must contain 'estimator' column")

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

    def get_FI_from_shap(self, num_features: int = 20, *, drop_formal_charge: bool = False) -> pd.DataFrame:
        df = self.df.groupby("feature")["mean_abs_shap"].agg(["mean", "std"])

        if drop_formal_charge:
            df = df.drop("formal_charge")

        df = df.reset_index().rename({"mean": "score", "std": "std_score"}, axis=1)
        self.fi_df = df.sort_values("score", ascending=False).head(num_features)
        return self.fi_df

    def plot(
        self, ax, num_features: int = 10, *, color: str = "tab:blue", title: str | None = None, shap: bool = False
    ) -> None:
        """Plot the feature importance on a given axis.

        params:
            ax (matplotlib.axes.Axes): axis to plot on
            num_features (int): top "n" features to plot
            color (str): color for QM features
        """
        num_features = min(num_features, self.fi_df.shape[0])

        try:
            df = self.fi_df.head(num_features)
        except AttributeError:
            raise RuntimeError("Feature importance not calculated. Run 'get_feature_importance' first.")

        pattern = "|".join(get_topology_features())
        topology_features = [feature for feature in df["feature"] if re.search(pattern, feature)] + [
            "in_ring?",
            "bond_type",
            "atomic_number",
            "degree",
            "number_of_hydrogens",
            "chiral_tag",
            "stereochemistry",
            "hybridization",
            "formal_charge",
            "conjugated?",
            "aromaticity",
            "mass",
            "Atomic structure",
            "Bond structure",
        ]

        colors = ["grey" if feature in topology_features else color for feature in df["feature"]]
        alphas = [0.6 if feature in topology_features else 1 for feature in df["feature"]]

        bars = ax.barh(
            df["feature"],
            df["score"].to_numpy(),
            color=colors,
            xerr=df["std_score"].to_numpy(),
            capsize=3,
        )

        for bar, alpha in zip(bars, alphas, strict=True):
            bar.set_alpha(alpha)

        legend_elements = [
            Patch(facecolor=color, label="QM feature"),
            Patch(facecolor="grey", label="Topological feature", alpha=0.6),
        ]

        for tick in ax.get_yticklabels():
            tick.set_size(14)
            if tick.get_text() not in topology_features:
                tick.set_fontweight("bold")
                tick.set_color(color)

        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        xlabel = "mean(|SHAP value|)" if shap else "Feature importance"
        ax.set_xlabel(xlabel, fontsize=18)
        if title is not None:
            ax.set_title(title, fontsize=20)
        # ax.legend(handles=legend_elements, frameon=False, loc="lower right", fontsize=12)

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
