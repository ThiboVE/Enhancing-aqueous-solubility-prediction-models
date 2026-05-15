from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

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
from ml_enhance.nn.featurizer.featurizer import get_featurizer

type FeatureTransformFn = Callable[
    ...,
    tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
]


@dataclass
class FeatureScalers:
    atom: StandardScaler | None
    bond: StandardScaler | None
    mol: StandardScaler | None
    target: StandardScaler | None


# ── datapoint construction ────────────────────────────────────────────────────


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


def make_datapoint(
    mol_data: NamedTuple,
    atom_lookup: dict[str, np.ndarray] | None,
    bond_lookup: dict[str, np.ndarray] | None,
    mol_lookup: dict[str, np.ndarray] | None,
    target_col: str = "solubility",
    *,
    transform_features: FeatureTransformFn | None = None,
) -> MoleculeDatapoint:
    smiles = str(mol_data.smiles)

    y = np.array([getattr(mol_data, target_col)], dtype=float)

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

    return MoleculeDatapoint.from_smi(
        smi=smiles,
        y=y,
        V_f=V_f_transformed,
        E_f=E_f_transformed,
        x_d=x_d_transformed,
        keep_h=True,
    )


def make_datapoints(
    target_df: pd.DataFrame,
    atom_lookup: dict[str, np.ndarray] | None,
    bond_lookup: dict[str, np.ndarray] | None,
    mol_lookup: dict[str, np.ndarray] | None,
    target_col: str = "solubility",
    *,
    transform_features: FeatureTransformFn | None = None,
) -> list[MoleculeDatapoint]:
    p_make_datapoint = partial(
        make_datapoint,
        atom_lookup=atom_lookup,
        bond_lookup=bond_lookup,
        mol_lookup=mol_lookup,
        target_col=target_col,
        transform_features=transform_features,
    )

    return [p_make_datapoint(row) for row in target_df.itertuples(index=False)]


# ── dataset construction ────────────────────────────────────────────────────


def build_datasets(
    train_ids: Iterable[int],
    val_ids: Iterable[int],
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    *,
    config: Config,
    datapoints_builder: Callable[..., list[MoleculeDatapoint]] = make_datapoints,
) -> tuple[MoleculeDataset, MoleculeDataset, FeatureScalers]:
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

    atom_scaler = None
    bond_scaler = None
    mol_scaler = None

    if (train_atom_df is not None) and (val_atom_df is not None):
        train_atom_df, val_atom_df, atom_scaler = scale_features(train_atom_df, val_atom_df, atom_features)
        train_atom_df, atom_rbf_cols = apply_rbf(train_atom_df, atom_features, config.n_rbf_centers)
        val_atom_df, _ = apply_rbf(val_atom_df, atom_features, config.n_rbf_centers)

    if (train_bond_df is not None) and (val_bond_df is not None):
        train_bond_df, val_bond_df, bond_scaler = scale_features(train_bond_df, val_bond_df, bond_features)
        train_bond_df, bond_rbf_cols = apply_rbf(train_bond_df, bond_features, config.n_rbf_centers)
        val_bond_df, _ = apply_rbf(val_bond_df, bond_features, config.n_rbf_centers)

    if (train_mol_df is not None) and (val_mol_df is not None):
        train_mol_df, val_mol_df, mol_scaler = scale_features(train_mol_df, val_mol_df, mol_features)
        raise NotImplementedError

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

    train_datapoints = datapoints_builder(
        train_target_df,
        train_atom_lookup,
        train_bond_lookup,
        train_mol_lookup,
        config.target_col,
    )

    val_datapoints = datapoints_builder(
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

    feature_scaler = FeatureScalers(
        atom_scaler,
        bond_scaler,
        mol_scaler,
        target_scaler,
    )

    return (
        MoleculeDataset(train_datapoints, featurizer=featurizer),
        MoleculeDataset(val_datapoints, featurizer=featurizer),
        feature_scaler,
    )
