from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── scaling ───────────────────────────────────────────────────────────────────


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler().fit(train_df[feature_cols])

    norm_train_df = train_df.copy()
    norm_val_df = val_df.copy()

    norm_train_df[feature_cols] = scaler.transform(norm_train_df[feature_cols])
    norm_val_df[feature_cols] = scaler.transform(norm_val_df[feature_cols])

    return norm_train_df, norm_val_df, scaler


def scale_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "solubility",
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    train_df, val_df, scaler = scale_features(train_df, val_df, [target_col])
    return train_df, val_df, scaler


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


# ──────────────────────────────────────────────────────────────────────────────


def apply_rbf(
    df: pd.DataFrame,
    feature_names: list[str],
    n_centers: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    expanded, expanded_names = rbf_expand(df[feature_names].to_numpy(), feature_names, n_centers)
    expanded_df = pd.DataFrame(expanded, columns=expanded_names, index=df.index)
    return pd.concat([df.drop(columns=feature_names), expanded_df], axis=1), expanded_names


def build_lookup(df: pd.DataFrame | None, group_col: str, feature_cols: list[str]) -> dict[str, pd.DataFrame] | None:
    if df is None:
        return None

    return df.groupby(group_col)[feature_cols].apply(lambda x: x.to_numpy(dtype=float).copy()).to_dict()


def split_df_by_ids(df: pd.DataFrame, ids: Iterable[int], id_col: str = "id") -> pd.DataFrame:
    return df.set_index(id_col).loc[ids].reset_index()


def subset_features(
    features: dict[str, pd.DataFrame | None],
    smiles: list[str],
) -> dict[str, pd.DataFrame | None]:
    return {
        key: df[df["original_smiles"].isin(smiles)].reset_index(drop=True) if df is not None else None
        for key, df in features.items()
    }
