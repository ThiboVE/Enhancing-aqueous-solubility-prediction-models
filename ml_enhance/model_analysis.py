from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
