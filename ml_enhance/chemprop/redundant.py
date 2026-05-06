from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ml_enhance import parallelize
from ml_enhance.chemprop import Config, build_datasets

# ── CV loop ───────────────────────────────────────────────────────────────────


def run_inner_loop(
    outer_train_files: np.ndarray,
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    inner_cv: KFold,
    config: Config,
) -> list[dict[str, MoleculeDataset | StandardScaler]]:
    inner_folds: list[dict[str, MoleculeDataset | StandardScaler]] = []

    for inner_tr_idxs, inner_val_idxs in inner_cv.split(outer_train_files):
        inner_train_files = outer_train_files[inner_tr_idxs]
        inner_val_files = outer_train_files[inner_val_idxs]

        train_dataset, val_dataset, target_scaler = build_datasets(
            inner_train_files,
            inner_val_files,
            target_df,
            all_features,
            use_custom_atom_featurizer=config.use_custom_atom_featurizer,
            use_custom_bond_featurizer=config.use_custom_bond_featurizer,
            n_rbf_centers=config.n_rbf_centers,
            target_col=config.target_col,
        )

        inner_folds.append(
            {
                "train": train_dataset,
                "val": val_dataset,
                "target_scaler": target_scaler,
            }
        )

    return inner_folds


def build_and_save_fold(
    outer_fold: tuple[int, tuple[np.ndarray, np.ndarray]],
    used_files: np.ndarray,
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    inner_cv: KFold,
    config: Config,
    output_dir: Path,
) -> None:
    outer_idx, (tr_idxs, tst_idxs) = outer_fold

    outer_train_files = used_files[tr_idxs]
    outer_test_files = used_files[tst_idxs]

    inner_folds = run_inner_loop(outer_train_files, target_df, all_features, inner_cv, config)

    outer_train_dataset, outer_test_dataset, outer_target_scaler = build_datasets(
        outer_train_files,
        outer_test_files,
        target_df,
        all_features,
        use_custom_atom_featurizer=config.use_custom_atom_featurizer,
        use_custom_bond_featurizer=config.use_custom_bond_featurizer,
        n_rbf_centers=config.n_rbf_centers,
        target_col=config.target_col,
    )

    torch.save(
        {
            "inner_folds": inner_folds,
            "outer_train": outer_train_dataset,
            "outer_test": outer_test_dataset,
            "outer_target_scaler": outer_target_scaler,
        },
        output_dir / f"outer_fold_{outer_idx}.pt",
    )

    print(f"Saved outer fold {outer_idx}")


def run_outer_loop(
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    used_files: np.ndarray,
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    inner_cv: KFold,
    config: Config,
    output_dir: Path,
    n_jobs: int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    p_build_and_save_fold = partial(
        build_and_save_fold,
        used_files=used_files,
        target_df=target_df,
        all_features=all_features,
        inner_cv=inner_cv,
        config=config,
        output_dir=output_dir,
    )

    parallelize(
        p_build_and_save_fold,
        enumerate(outer_splits),
        n_jobs=n_jobs,
    )
