import numpy as np
import pandas as pd
from scipy import stats


class StatisticalComparison:
    """Class that encompasses statistical methods to compare two different models."""

    def __init__(self, model1_scores: np.ndarray, model2_scores: np.ndarray, n_tot: int) -> None:
        if model1_scores.shape != model2_scores.shape:
            raise ValueError("scores1 and scores2 must have the same shape")

        self.scores1 = np.abs(model1_scores)
        self.scores2 = np.abs(model2_scores)

        self.n_train = n_tot * 4 // 5
        self.n_test = n_tot - self.n_train

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

    def nadeau_bengio_corrected_t_test(self) -> dict[str, float]:
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
        mean_diff: float = np.mean(d)
        var_d = np.var(d, ddof=1)

        # Nadeau & Bengio corrected variance
        n_folds = len(d)
        corrected_var = var_d / n_folds * (1 + self.n_test / self.n_train)

        # Corrected t-statistic
        t_stat = mean_diff / np.sqrt(corrected_var)

        # Degrees of freedom
        df = n_folds - 1
        p_value: float = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

        return {"t_stat": t_stat, "p_value": p_value, "mean_diff": mean_diff}


def compare(
    combo_df: pd.DataFrame, topo_df: pd.DataFrame, qm_df: pd.DataFrame | None = None, n_tot: int = 8763
) -> None:
    dfs = [combo_df, topo_df, qm_df] if qm_df is not None else [combo_df, topo_df]
    assert all("name" in df.columns for df in dfs), "all dataframes should contain 'name' column."
    metrics = ["r2", "MSE"]
    for metric in metrics:
        comparator = StatisticalComparison(
            combo_df[f"test_{metric}"].to_numpy(), topo_df[f"test_{metric}"].to_numpy(), n_tot
        )

        ttest_result = comparator.nadeau_bengio_corrected_t_test()
        wilcoxon_result = comparator.wilcoxon_fold_differences()

        for df in dfs:
            print(
                f"{df['name'][0]} mean {metric}: {np.abs(df[f'test_{metric}']).mean()} (Train: {np.abs(df[f'train_{metric}']).mean()})"
            )

        print("Mean improvement:", ttest_result["mean_diff"])
        print(
            f"T-test p-value: {ttest_result['p_value']} ->{' not' if ttest_result['p_value'] > 0.05 else ''} statistically significant"
        )
        print(
            f"Wilcoxon p-value: {wilcoxon_result['p_value_w']} ->{' not' if wilcoxon_result['p_value_w'] > 0.05 else ''} statistically significant"
        )
        print("\n")
