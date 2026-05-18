import numpy as np
import pandas as pd

# ── RBF expansion ─────────────────────────────────────────────────────────────


def rbf_expand(
    X: np.ndarray,
    feature_names: list[str],
    n_centers: int = 10,
    v: float = -3.0,
) -> tuple[np.ndarray, list[str]]:
    delta = 6.0 / (n_centers - 1)
    centers = v + delta * np.arange(n_centers)

    diff = X[:, :, np.newaxis] - centers[np.newaxis, np.newaxis, :]
    rbf = np.exp(-(diff**2) / delta)

    expanded = rbf.reshape(X.shape[0], -1)
    expanded_names = [f"{feat}_{k}" for feat in feature_names for k in range(n_centers)]

    return expanded, expanded_names


def apply_rbf(
    df: pd.DataFrame,
    feature_names: list[str],
    n_centers: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    expanded, expanded_names = rbf_expand(df[feature_names].to_numpy(), feature_names, n_centers)
    expanded_df = pd.DataFrame(expanded, columns=expanded_names, index=df.index)
    return pd.concat([df.drop(columns=feature_names), expanded_df], axis=1), expanded_names
