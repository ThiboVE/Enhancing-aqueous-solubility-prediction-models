from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline


class PlotOptions(NamedTuple):
    """NamedTuple for storing the settings of the plot_scaled_linreg_result function.

    - n_cols:      number of columns for drawing grid of scatter plots
    - point_size:  scatter plot point size
    - point_alpha: scatter plot point opacity
    - line_width:  line of best fit width
    - line_alpha:  line of best fit opacity
    """

    n_cols: int = 2
    point_size: float = 0.5
    point_alpha: float = 0.05
    line_width: int = 2
    line_alpha: float = 0.8


def plot_scaled_linreg_result(
    X: pd.DataFrame, y: pd.Series, pipeline: Pipeline, plot_opts: PlotOptions = PlotOptions()
) -> None:
    """Display line of best fit from a Scaler->LinearRegression pipeline per predictor variable.

    - X:           pandas.DataFrame with explanatory variables
    - y:           pandas.Series with target variable
    - pipeline:    sklearn pipeline with a 'scale' and 'predict' slot
    - n_cols:      number of columns for drawing grid of scatter plots
    - point_size:  scatter plot point size
    - point_alpha: scatter plot point opacity
    - line_width:  line of best fit width
    - line_alpha:  line of best fit opacity
    """
    ## Scale the data using provided scaler
    X_scaled = pipeline[:-1].transform(X)

    ## Create subplot for each predictor
    n_plots = X.shape[1]
    n_rows = int(np.ceil(n_plots / plot_opts.n_cols))
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=plot_opts.n_cols, sharey=True, figsize=(plot_opts.n_cols * 3, n_rows * 4)
    )
    fig.text(0, 0.5, y.name, va="center", rotation="vertical")

    ## Iterate over predictors
    for idx in range(n_plots):
        idx_row = (idx) // plot_opts.n_cols
        idx_col = (idx) % plot_opts.n_cols
        feature_name = X.columns[idx]
        ax = axs[idx_row, idx_col] if n_rows > 1 else axs[idx]
        ax.set_xlabel(feature_name)
        ax.set_ylim(-14, 3)

        ## Extract coefficients
        a = pipeline["predict"].coef_[idx]  # slope
        b = pipeline["predict"].intercept_  # intercept
        x = X_scaled[:, idx]

        ## Plot the scaled data points
        ax.scatter(x, y, color="blue", alpha=plot_opts.point_alpha, s=plot_opts.point_size)

        ## Plot the line of best fit
        line_points = a * x + b
        ax.plot(x, line_points, color="mediumvioletred", linewidth=plot_opts.line_width, alpha=plot_opts.line_alpha)

    plt.show()


class StatisticalComparison:
    """Class that encompasses statistical methods to compare two different models."""

    def __init__(self, model1_scores: np.ndarray, model2_scores: np.ndarray) -> None:
        if model1_scores.shape != model2_scores.shape:
            raise ValueError("scores1 and scores2 must have the same shape")

        self.scores1 = model1_scores
        self.scores2 = model2_scores

    def wilcoxon_fold_differences(self) -> dict[str, float]:
        """Compute Wilcoxon signed-rank test for fold-wise differences.

        Parameters
        ----------
        None

        Returns:
        -------
        dict with keys:
            'wilcoxon_stat' : Wilcoxon signed-rank statistic
            'p_value_w' : two-sided p-value
            'mean_diff' : mean difference across folds
        """
        fold_diffs: np.ndarray = self.scores1 - self.scores2
        mean_diff: float = np.mean(fold_diffs)

        wilcoxon_stat, p_value_w = stats.wilcoxon(fold_diffs)

        return {"wilcoxon_stat": wilcoxon_stat, "p_value_w": p_value_w, "mean_diff": mean_diff}

    def nadeau_bengio_corrected_t_test(self, n_train: int, n_test: int) -> dict[str, float]:
        """Compute Nadeau & Bengio (2003) corrected paired t-test for repeated k-fold CV.

        Parameters:
        -----------
        n_train : int
            Number of training samples in each fold.
        n_test : int
            Number of test samples in each fold.

        Returns:
        --------
        t_stat : float
            Corrected t-statistic.
        p_value : float
            Two-sided p-value.
        mean_diff : float
            Mean difference of scores1 - scores2.
        """
        # Fold-wise differences
        d: np.ndarray = self.scores1 - self.scores2
        mean_diff = np.mean(d)
        var_d = np.var(d, ddof=1)

        # Nadeau & Bengio corrected variance
        n_folds = len(d)
        corrected_var = var_d / n_folds * (1 + n_test / n_train)

        # Corrected t-statistic
        t_stat = mean_diff / np.sqrt(corrected_var)

        # Degrees of freedom
        df = n_folds - 1
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

        return {"t_stat": t_stat, "p_value": p_value, "mean_diff": mean_diff}
