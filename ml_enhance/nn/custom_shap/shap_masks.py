import numpy as np


def mask_extra_features(
    arr: np.ndarray | None,
    keep_features: list[bool],
    n_rbf: int = 10,
) -> np.ndarray | None:
    """Zero out RBF features in V_f or E_f based on keep_features mask.

    Parameters
    ----------
    arr:
        Feature array of shape (n_atoms_or_bonds, n_features * n_rbf).
    keep_features:
        One bool per feature. False means zero out that feature's n_rbf columns.
    n_rbf:
        Number of columns per feature.
    """
    if arr is None:
        return None

    masked = arr.copy()
    for group_idx, keep in enumerate(keep_features):
        if not keep:
            start = group_idx * n_rbf
            end = start + n_rbf
            masked[:, start:end] = 0.0
    return masked


def mask_mol_features(
    x_d: np.ndarray | None,
    keep_features: list[bool],
) -> np.ndarray | None:
    """Zero out individual molecule-level features based on keep_features mask."""
    if x_d is None:
        return None

    masked = x_d.copy()
    for i, keep in enumerate(keep_features):
        if not keep:
            masked[i] = 0
    return masked
