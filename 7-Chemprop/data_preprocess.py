import re
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.featurizers import MultiHotAtomFeaturizer, MultiHotBondFeaturizer, SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from scipy.constants import (
    Avogadro,  # 1/mol
    Boltzmann,  # in J/K
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ml_enhance import QuantumFPFileLoader, parallelize


@dataclass
class Config:
    use_atom_features: bool = True
    use_bond_features: bool = False
    use_mol_features: bool = False
    use_custom_atom_featurizer: bool = True
    use_custom_bond_featurizer: bool = False
    n_rbf_centers: int = 10
    target_col: str = "solubility"


atomic_features: list[str] = [
    "atomic_fukui_minus",
    "atomic_fukui_plus",
    "atomic_dipole_norm",
    "atomic_quadrupole_principal_invariant_2",
    "atomic_quadrupole_principal_invariant_3",
    "atomic_polarizability_mean",
    "atomic_polarizability_anisotropy",
    "atomic_sasa",
    "partial_charge",
]

bond_features: list[str] = [
    "bond_length",
    "atomic_charge_quadrupole_interaction",
]  # TODO: fill in bond feature column names
mol_features: list[str] = [
    "molecular_dipole_norm",
]  # TODO: fill in molecular feature column names


def filter_files(files: Iterable[Path], used_ids: list[int]) -> list[Path]:
    used_files: list[Path] = []

    for file in files:
        matches = re.findall(r"\d+", file.name)
        file_id = int(matches[1])

        if file_id in used_ids:
            used_files.append(file)

    return used_files


def stream_conformer_df(
    file: Path,
    loader: QuantumFPFileLoader,
) -> Generator[pd.DataFrame, None, None]:
    for df in loader.stream_conformer_dataframe(file):
        df["gibbs_free_energy_300K"] = df["gibbs_free_energy"].map(lambda x: x[1][1])
        yield df


# ── boltzmann averaging ───────────────────────────────────────────────────────


def boltzmann_weights(G: np.ndarray, T: float = 300.0) -> np.ndarray:
    k_B: float = Boltzmann * Avogadro * 0.000239005736
    delta_G = G - G.min()
    factors = np.exp(-delta_G / (k_B * T))
    return factors / factors.sum()


# ── feature extraction ────────────────────────────────────────────────────────


def reorder_atom_features(molecule_df: pd.DataFrame) -> pd.DataFrame:
    """The atom features are ordered according to the map number of each atom (atom.GetAtomMapNum()), however, Chemprop uses the atom index (atom.GetIdx()) to order features.

    These two (map number and index) are not always the same, so the atoms in the DataFrame need to be reordered.
    """
    mol = Chem.MolFromSmiles(
        molecule_df.loc[0, "original_smiles"], sanitize=False
    )  # sanitize=False => keep the H atoms

    mapnum_order = [atom.GetAtomMapNum() - 1 for atom in mol.GetAtoms()]

    return molecule_df.iloc[mapnum_order]


def get_bond_mapping(smiles: str) -> dict[tuple[int, int], int]:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    return {
        (bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()): bond.GetIdx()
        for bond in mol.GetBonds()
    }


def filter_bond_features(df: pd.DataFrame) -> pd.DataFrame:
    """Some of the bond features (such as the interaction features) contain more pairs than there are bonds.

    Filter them out.
    """
    bond_atom_pairs = get_bond_mapping(df.loc[0, "original_smiles"]).keys()

    def filter_list(lst: list[int | float]) -> list[int | float]:
        return [x for x in lst if (x[0], x[1]) in bond_atom_pairs]

    for col in bond_features:
        df[col] = df[col].apply(filter_list)

    return df


def get_bond_idx(smiles: str, begin_atom_idxs: np.ndarray, end_atom_idxs: np.ndarray) -> np.ndarray:
    """The atoms are denoted with their map indices, which are not used in chemprop. Chemprop uses the bond index and iterates over the bond indices from 0 onward.

    => Provide a mapping between atom map index pairs and the bond index, e.g. (1, 2): 1
    """
    mapping = get_bond_mapping(smiles)

    return np.array(
        [mapping[(begin_idx, end_idx)] for begin_idx, end_idx in zip(begin_atom_idxs, end_atom_idxs, strict=True)]
    )


def extract_atom_features(df: pd.DataFrame) -> pd.DataFrame:
    selected_df = df[["original_smiles", "output_smiles", "gibbs_free_energy_300K", *atomic_features]]
    exploded_df = selected_df.explode(atomic_features)

    G = selected_df["gibbs_free_energy_300K"].unique()
    weights = boltzmann_weights(G)

    arr = np.array(exploded_df[atomic_features].values.tolist())  # (n_conformers * n_atoms, n_features, 2)
    atom_map_idx = arr[:, 0, 0].astype(int)
    values = arr[:, :, 1].astype(float)

    unique_atom_idxs = pd.unique(atom_map_idx)

    n_conformers = len(weights)
    n_atoms = len(unique_atom_idxs)
    n_features = len(atomic_features)

    atom_matrix = values.reshape(n_conformers, n_atoms, n_features)
    averages = np.einsum("i,ijk->jk", weights, atom_matrix)

    result = pd.DataFrame(averages, columns=atomic_features)
    result.insert(0, "atom_map_idx", unique_atom_idxs)
    result.insert(0, "original_smiles", selected_df["original_smiles"].iloc[0])

    return reorder_atom_features(result)


def extract_bond_features(df: pd.DataFrame) -> pd.DataFrame:
    selected_df = df[["original_smiles", "output_smiles", "gibbs_free_energy_300K", *bond_features]]
    selected_df = filter_bond_features(selected_df)
    exploded_df = selected_df.explode(bond_features)

    # exploded_df["smiles_equal"] = exploded_df["original_smiles"] == exploded_df["output_smiles"]

    G = selected_df["gibbs_free_energy_300K"].unique()
    weights = boltzmann_weights(G)

    arr = np.array(exploded_df[bond_features].values.tolist())  # (n_conformers * n_bonds, n_features, 3)

    begin_atom_map_idx = arr[:, 0, 0].astype(int)
    end_atom_map_idx = arr[:, 0, 1].astype(int)
    values = arr[:, :, -1].astype(float)

    bond_idx = get_bond_idx(exploded_df["original_smiles"].values[0], begin_atom_map_idx, end_atom_map_idx)

    unique_bond_idxs = pd.unique(bond_idx)

    n_conformers = len(weights)
    n_bonds = len(unique_bond_idxs)
    n_features = len(bond_features)

    bond_matrix = values.reshape(n_conformers, n_bonds, n_features)
    averages = np.einsum("i,ijk->jk", weights, bond_matrix)

    result = pd.DataFrame(averages, columns=bond_features)
    result.insert(0, "bond_idx", unique_bond_idxs)
    result.insert(0, "original_smiles", selected_df["original_smiles"].iloc[0])

    return result.sort_values("bond_idx")


def extract_mol_features(df: pd.DataFrame) -> pd.DataFrame:
    selected_df = df[["original_smiles", "output_smiles", "gibbs_free_energy_300K", *mol_features]]

    G = selected_df["gibbs_free_energy_300K"].unique()
    weights = boltzmann_weights(G)

    arr = selected_df[mol_features].to_numpy()  # (n_conformers, n_features)

    averages = np.einsum("i,ij->j", weights, arr)

    result = pd.DataFrame(averages, columns=mol_features)
    result.insert(0, "original_smiles", selected_df["original_smiles"].iloc[0])

    return result


# ── file processing ───────────────────────────────────────────────────────────


def process_files(
    files: Iterable[Path],
    loader: QuantumFPFileLoader,
    *,
    use_atom_features: bool = True,
    use_bond_features: bool = False,
    use_mol_features: bool = False,
) -> dict[str, pd.DataFrame | None]:
    all_atoms: list[pd.DataFrame] = []
    all_bonds: list[pd.DataFrame] = []
    all_mols: list[pd.DataFrame] = []

    if use_atom_features or use_bond_features or use_mol_features:
        for file in files:
            for sdf in stream_conformer_df(file, loader):
                if use_atom_features:
                    all_atoms.append(extract_atom_features(sdf))
                if use_bond_features:
                    all_bonds.append(extract_bond_features(sdf))
                if use_mol_features:
                    all_mols.append(extract_mol_features(sdf))

    return {
        "atoms": pd.concat(all_atoms, ignore_index=True) if all_atoms else None,
        "bonds": pd.concat(all_bonds, ignore_index=True) if all_bonds else None,
        "mols": pd.concat(all_mols, ignore_index=True) if all_mols else None,
    }


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


def apply_rbf(
    df: pd.DataFrame,
    feature_names: list[str],
    n_centers: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    expanded, expanded_names = rbf_expand(df[feature_names].to_numpy(), feature_names, n_centers)
    expanded_df = pd.DataFrame(expanded, columns=expanded_names, index=df.index)
    return pd.concat([df.drop(columns=feature_names), expanded_df], axis=1), expanded_names


# ── datapoint construction ────────────────────────────────────────────────────


def make_datapoints(
    target_df: pd.DataFrame,
    atom_df: pd.DataFrame | None,
    bond_df: pd.DataFrame | None,
    mol_df: pd.DataFrame | None,
    atom_rbf_cols: list[str],
    bond_rbf_cols: list[str],
    target_col: str = "solubility",
) -> list[MoleculeDatapoint]:
    datapoints = []

    for smiles in target_df["smiles"]:
        y = target_df.loc[target_df["smiles"] == smiles, target_col].to_numpy()

        V_f = (
            atom_df[atom_df["original_smiles"] == smiles][atom_rbf_cols].to_numpy(dtype=float)
            if atom_df is not None
            else None
        )
        E_f = (
            bond_df[bond_df["original_smiles"] == smiles][bond_rbf_cols].to_numpy(dtype=float)
            if bond_df is not None
            else None
        )
        x_d = (
            np.atleast_1d(mol_df[mol_df["original_smiles"] == smiles][mol_features].to_numpy(dtype=float).squeeze())
            if mol_df is not None
            else None
        )

        datapoints.append(MoleculeDatapoint.from_smi(smi=smiles, y=y, V_f=V_f, E_f=E_f, x_d=x_d, keep_h=True))

    return datapoints


# ── fold building ─────────────────────────────────────────────────────────────


def get_featurizer(
    *,
    use_custom_atom_featurizer: bool = False,
    use_custom_bond_featurizer: bool = False,
    extra_atom_fdim: int = 0,
    extra_bond_fdim: int = 0,
) -> SimpleMoleculeMolGraphFeaturizer:
    if use_custom_atom_featurizer:
        # These represent all atoms present in the processed dataset "processed_dataset_wo_metals_w_even_more_qm2.csv"
        atomic_nums = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 35, 53]
        degrees = [0, 1, 2, 3, 4, 5]
        formal_charges = []
        chiral_tags = [0, 1, 2, 3]
        num_Hs = [0, 1, 2, 3, 4]
        hybridizations = [
            HybridizationType.S,
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP2D,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ]

        atom_featurizer = MultiHotAtomFeaturizer(
            atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations
        )
    else:
        atom_featurizer = MultiHotAtomFeaturizer.v2()

    if use_custom_bond_featurizer:
        raise NotImplementedError

    bond_featurizer = MultiHotBondFeaturizer()

    return SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        extra_atom_fdim=extra_atom_fdim,
        extra_bond_fdim=extra_bond_fdim,
    )


def get_smiles_from_files(
    files: Iterable[Path],
    target_df: pd.DataFrame,
) -> list[str]:
    ids = [int(re.findall(r"\d+", file.name)[1]) for file in files]
    return target_df[target_df["id"].isin(ids)]["smiles"].tolist()


def subset_features(
    features: dict[str, pd.DataFrame | None],
    smiles: list[str],
) -> dict[str, pd.DataFrame | None]:
    return {
        key: df[df["original_smiles"].isin(smiles)].reset_index(drop=True) if df is not None else None
        for key, df in features.items()
    }


def build_datasets(
    train_files: np.ndarray,
    val_files: np.ndarray,
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    *,
    use_custom_atom_featurizer: bool = False,
    use_custom_bond_featurizer: bool = False,
    n_rbf_centers: int = 10,
    target_col: str = "solubility",
) -> tuple[MoleculeDataset, MoleculeDataset, StandardScaler]:
    train_smiles = get_smiles_from_files(train_files, target_df)
    val_smiles = get_smiles_from_files(val_files, target_df)

    train_features = subset_features(all_features, train_smiles)
    val_features = subset_features(all_features, val_smiles)

    train_atom_df = train_features["atoms"]
    val_atom_df = val_features["atoms"]

    train_bond_df = train_features["bonds"]
    val_bond_df = val_features["bonds"]

    train_mol_df = train_features["mols"]
    val_mol_df = val_features["mols"]

    train_target_df = target_df[target_df["smiles"].isin(train_smiles)]
    val_target_df = target_df[target_df["smiles"].isin(val_smiles)]

    atom_rbf_cols: list[str] = []
    bond_rbf_cols: list[str] = []

    if (train_atom_df is not None) and (val_atom_df is not None):
        train_atom_df, val_atom_df, _ = scale_features(train_atom_df, val_atom_df, atomic_features)
        train_atom_df, atom_rbf_cols = apply_rbf(train_atom_df, atomic_features, n_rbf_centers)
        val_atom_df, _ = apply_rbf(val_atom_df, atomic_features, n_rbf_centers)

    if (train_bond_df is not None) and (val_bond_df is not None):
        train_bond_df, val_bond_df, _ = scale_features(train_bond_df, val_bond_df, bond_features)
        train_bond_df, bond_rbf_cols = apply_rbf(train_bond_df, bond_features, n_rbf_centers)
        val_bond_df, _ = apply_rbf(val_bond_df, bond_features, n_rbf_centers)

    if (train_mol_df is not None) and (val_mol_df is not None):
        train_mol_df, val_mol_df, _ = scale_features(train_mol_df, val_mol_df, mol_features)

    train_target_df, val_target_df, target_scaler = scale_target(train_target_df, val_target_df, target_col)

    train_datapoints = make_datapoints(
        train_target_df,
        train_atom_df,
        train_bond_df,
        train_mol_df,
        atom_rbf_cols,
        bond_rbf_cols,
        target_col,
    )
    val_datapoints = make_datapoints(
        val_target_df,
        val_atom_df,
        val_bond_df,
        val_mol_df,
        atom_rbf_cols,
        bond_rbf_cols,
        target_col,
    )

    featurizer = get_featurizer(
        use_custom_atom_featurizer=use_custom_atom_featurizer,
        use_custom_bond_featurizer=use_custom_bond_featurizer,
        extra_atom_fdim=len(atom_rbf_cols) if train_atom_df is not None else 0,
        extra_bond_fdim=len(bond_rbf_cols) if train_bond_df is not None else 0,
    )
    return (
        MoleculeDataset(train_datapoints, featurizer=featurizer),
        MoleculeDataset(val_datapoints, featurizer=featurizer),
        target_scaler,
    )


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    parallelize(
        build_and_save_fold,
        enumerate(outer_splits),
        n_jobs=5,
        used_files=used_files,
        target_df=target_df,
        all_features=all_features,
        inner_cv=inner_cv,
        config=config,
        output_dir=output_dir,
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    target_df = pd.read_csv("../data/processed_dataset_wo_metals_w_even_more_qm2.csv")
    used_ids: list[int] = target_df["id"].apply(round).to_list()

    qfp_loader = QuantumFPFileLoader(Path("../data/QuantumFP/QFP_output"))
    filelist: list[Path] = qfp_loader.list_output_files()
    used_files = np.array(filter_files(filelist, used_ids))

    outer_splits = pd.read_pickle("../hpc_splits.pkl")
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    config = Config()

    all_features = process_files(
        used_files,
        qfp_loader,
        use_atom_features=config.use_atom_features,
        use_bond_features=config.use_bond_features,
        use_mol_features=config.use_mol_features,
    )

    run_outer_loop(
        outer_splits,
        used_files,
        target_df,
        all_features,
        inner_cv,
        config,
        output_dir=Path("../data/folds_no_added"),
    )


if __name__ == "__main__":
    main()
