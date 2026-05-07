from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from sklearn.preprocessing import StandardScaler

from ml_enhance.nn.config import (
    Config,
    atom_features,
    bond_features,
    mol_features,
)
from ml_enhance.nn.features import (
    apply_rbf,
    build_lookup,
    scale_features,
    scale_target,
    split_df_by_ids,
    subset_features,
)
from ml_enhance.nn.featurizer import get_featurizer

# ── datapoint construction ────────────────────────────────────────────────────

type FeatureTransform = Callable[
    [np.ndarray | None, np.ndarray | None, np.ndarray | None],
    tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
]


def _lookup_features(
    smiles: str,
    atom_lookup: dict[str, np.ndarray] | None,
    bond_lookup: dict[str, np.ndarray] | None,
    mol_lookup: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    return (
        atom_lookup.get(smiles) if atom_lookup else None,
        bond_lookup.get(smiles) if bond_lookup else None,
        mol_lookup.get(smiles) if mol_lookup else None,
    )


def make_datapoints(
    target_df: pd.DataFrame,
    atom_lookup: dict[str, np.ndarray] | None,
    bond_lookup: dict[str, np.ndarray] | None,
    mol_lookup: dict[str, np.ndarray] | None,
    target_col: str = "solubility",
    *,
    transform_features: FeatureTransform | None = None,
) -> list[MoleculeDatapoint]:
    datapoints = []

    for row in target_df.itertuples(index=False):
        smiles = str(row.smiles)

        y = np.array([getattr(row, target_col)], dtype=float)

        V_f, E_f, x_d = _lookup_features(
            smiles,
            atom_lookup,
            bond_lookup,
            mol_lookup,
        )

        V_f_transformed, E_f_transformed, x_d_transformed = (
            V_f,
            E_f,
            x_d if transform_features is None else transform_features(V_f, E_f, x_d),
        )

        datapoints.append(
            MoleculeDatapoint.from_smi(
                smi=smiles,
                y=y,
                V_f=V_f_transformed,
                E_f=E_f_transformed,
                x_d=x_d_transformed,
                keep_h=True,
            )
        )

    return datapoints


# ── dataset construction ────────────────────────────────────────────────────


def build_datasets(
    train_ids: Iterable[int],
    val_ids: Iterable[int],
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    *,
    config: Config,
) -> tuple[MoleculeDataset, MoleculeDataset, StandardScaler]:
    train_target_df = split_df_by_ids(target_df, train_ids)
    val_target_df = split_df_by_ids(target_df, val_ids)

    train_smiles = train_target_df["smiles"].tolist()
    val_smiles = val_target_df["smiles"].tolist()

    train_features = subset_features(all_features, train_smiles)
    val_features = subset_features(all_features, val_smiles)

    train_atom_df = train_features["atoms"] if config.use_atom_features else None
    val_atom_df = val_features["atoms"] if config.use_atom_features else None

    train_bond_df = train_features["bonds"] if config.use_bond_features else None
    val_bond_df = val_features["bonds"] if config.use_bond_features else None

    train_mol_df = train_features["mols"] if config.use_mol_features else None
    val_mol_df = val_features["mols"] if config.use_mol_features else None

    atom_rbf_cols: list[str] = []
    bond_rbf_cols: list[str] = []

    if (train_atom_df is not None) and (val_atom_df is not None):
        train_atom_df, val_atom_df, _ = scale_features(train_atom_df, val_atom_df, atom_features)
        train_atom_df, atom_rbf_cols = apply_rbf(train_atom_df, atom_features, config.n_rbf_centers)
        val_atom_df, _ = apply_rbf(val_atom_df, atom_features, config.n_rbf_centers)

    if (train_bond_df is not None) and (val_bond_df is not None):
        train_bond_df, val_bond_df, _ = scale_features(train_bond_df, val_bond_df, bond_features)
        train_bond_df, bond_rbf_cols = apply_rbf(train_bond_df, bond_features, config.n_rbf_centers)
        val_bond_df, _ = apply_rbf(val_bond_df, bond_features, config.n_rbf_centers)

    if (train_mol_df is not None) and (val_mol_df is not None):
        raise NotImplementedError
        train_mol_df, val_mol_df, _ = scale_features(train_mol_df, val_mol_df, mol_features)

    train_target_df, val_target_df, target_scaler = scale_target(train_target_df, val_target_df, config.target_col)

    train_atom_lookup = (
        build_lookup(train_atom_df, "original_smiles", atom_rbf_cols) if train_atom_df is not None else None
    )
    val_atom_lookup = build_lookup(val_atom_df, "original_smiles", atom_rbf_cols) if val_atom_df is not None else None

    train_bond_lookup = (
        build_lookup(train_bond_df, "original_smiles", bond_rbf_cols) if train_bond_df is not None else None
    )
    val_bond_lookup = build_lookup(val_bond_df, "original_smiles", bond_rbf_cols) if val_bond_df is not None else None

    train_mol_lookup = (
        None
        if train_mol_df is None
        else train_mol_df.groupby("original_smiles")[mol_features]
        .apply(lambda x: np.atleast_1d(x.to_numpy(dtype=np.float32).squeeze()))
        .to_dict()
    )

    val_mol_lookup = (
        None
        if val_mol_df is None
        else val_mol_df.groupby("original_smiles")[mol_features]
        .apply(lambda x: np.atleast_1d(x.to_numpy(dtype=np.float32).squeeze()))
        .to_dict()
    )

    train_datapoints = make_datapoints(
        train_target_df,
        train_atom_lookup,
        train_bond_lookup,
        train_mol_lookup,
        config.target_col,
    )

    val_datapoints = make_datapoints(
        val_target_df,
        val_atom_lookup,
        val_bond_lookup,
        val_mol_lookup,
        config.target_col,
    )

    featurizer = get_featurizer(
        use_custom_atom_featurizer=config.use_custom_atom_featurizer,
        use_custom_bond_featurizer=config.use_custom_bond_featurizer,
        extra_atom_fdim=len(atom_rbf_cols) if train_atom_df is not None else 0,
        extra_bond_fdim=len(bond_rbf_cols) if train_bond_df is not None else 0,
    )
    return (
        MoleculeDataset(train_datapoints, featurizer=featurizer),
        MoleculeDataset(val_datapoints, featurizer=featurizer),
        target_scaler,
    )
