from collections.abc import Generator, Iterable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.constants import (
    Avogadro,  # 1/mol
    Boltzmann,  # in J/K
)

from ml_enhance import QuantumFPFileLoader, RDKitFeatureCalculator, parallelize
from ml_enhance.nn import atom_features, bond_features, mol_features


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
        return [x for x in lst if (x[0], x[1]) in bond_atom_pairs or (x[1], x[0]) in bond_atom_pairs]

    for col in bond_features:
        if col in df.columns:
            df[col] = df[col].apply(filter_list)
        else:
            print(f"{col} not present in dataframe.")

    return df


def get_bond_idx(smiles: str, begin_atom_idxs: np.ndarray, end_atom_idxs: np.ndarray) -> np.ndarray:
    """The atoms are denoted with their map indices, which are not used in chemprop. Chemprop uses the bond index and iterates over the bond indices from 0 onward.

    => Provide a mapping between atom map index pairs and the bond index, e.g. (1, 2): 1
    """
    mapping = get_bond_mapping(smiles)

    return np.array(
        [mapping[(begin_idx, end_idx)] for begin_idx, end_idx in zip(begin_atom_idxs, end_atom_idxs, strict=True)]
    )


def extract_atom_features(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    arr = np.array(df[atom_features].values.tolist())  # shape: (n_conformers, n_features, n_atoms, 2)
    arr = arr.transpose(0, 2, 1, 3)  # shape: (n_conformers, n_atoms, n_features, 2)
    atom_map_idx = arr[0, :, 0, 0].astype(int)
    values = arr[:, :, :, 1].astype(float)

    unique_atom_idxs = pd.unique(atom_map_idx)

    n_conformers = len(weights)
    n_atoms = len(unique_atom_idxs)
    n_features = len(atom_features)

    atom_matrix = values.reshape(n_conformers, n_atoms, n_features)
    averages = np.einsum("i,ijk->jk", weights, atom_matrix)  # (n_atoms, n_features)

    result = pd.DataFrame(averages, columns=atom_features)
    result.insert(0, "atom_map_idx", unique_atom_idxs)
    result.insert(0, "original_smiles", df["original_smiles"].iloc[0])

    return reorder_atom_features(result)


def extract_bond_features(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    df = filter_bond_features(df)

    arr = np.array(df[bond_features].values.tolist())  # shape: (n_conformers, n_features, n_bonds, 2)
    arr = arr.transpose(0, 2, 1, 3)  # shape: (n_conformers, n_bonds, n_features, 2)
    begin_atom_map_idx = arr[0, :, 0, 0].astype(int)
    end_atom_map_idx = arr[0, :, 0, 1].astype(int)
    values = arr[:, :, :, -1].astype(float)

    bond_idx = get_bond_idx(df["original_smiles"].values[0], begin_atom_map_idx, end_atom_map_idx)

    unique_bond_idxs = pd.unique(bond_idx)

    n_conformers = len(weights)
    n_bonds = len(unique_bond_idxs)
    n_features = len(bond_features)

    bond_matrix = values.reshape(n_conformers, n_bonds, n_features)
    averages = np.einsum("i,ijk->jk", weights, bond_matrix)  # (n_bonds, n_features)

    result = pd.DataFrame(averages, columns=bond_features)
    result.insert(0, "bond_idx", unique_bond_idxs)
    result.insert(0, "original_smiles", df["original_smiles"].iloc[0])

    return result.sort_values("bond_idx")


def extract_mol_features(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    rdkit_calc = RDKitFeatureCalculator("original_smiles", descriptor_names=["TPSA", "MolLogP", "MolWt"])
    df = rdkit_calc.add_to_dataframe(df)

    arr = df[mol_features].to_numpy()  # (n_conformers, n_features)

    averages = np.einsum("i,ij->j", weights, arr).reshape(1, -1)  # (, n_features)

    result = pd.DataFrame(averages, columns=mol_features)
    result.insert(0, "original_smiles", df["original_smiles"].iloc[0])

    return result


# ── file processing ───────────────────────────────────────────────────────────


def _process_single_file(
    file: Path,
    loader: QuantumFPFileLoader,
    *,
    use_atom_features: bool = True,
    use_bond_features: bool = False,
    use_mol_features: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    atoms: list[pd.DataFrame] = []
    bonds: list[pd.DataFrame] = []
    mols: list[pd.DataFrame] = []

    for df in stream_conformer_df(file, loader):
        G = df["gibbs_free_energy_300K"].unique()
        weights = boltzmann_weights(G)

        if use_atom_features:
            atoms.append(extract_atom_features(df, weights))
        if use_bond_features:
            bonds.append(extract_bond_features(df, weights))
        if use_mol_features:
            mols.append(extract_mol_features(df, weights))

    return atoms, bonds, mols


def process_files(
    files: Iterable[Path],
    loader: QuantumFPFileLoader,
    *,
    use_atom_features: bool = False,
    use_bond_features: bool = False,
    use_mol_features: bool = False,
    n_jobs: int = 5,
) -> dict[str, pd.DataFrame | None]:
    if (not use_atom_features) and (not use_bond_features) and (not use_mol_features):
        return {
            "atoms": None,
            "bonds": None,
            "mols": None,
        }

    all_atoms: list[pd.DataFrame] = []
    all_bonds: list[pd.DataFrame] = []
    all_mols: list[pd.DataFrame] = []

    p_process_single_file = partial(
        _process_single_file,
        loader=loader,
        use_atom_features=use_atom_features,
        use_bond_features=use_bond_features,
        use_mol_features=use_mol_features,
    )

    results = parallelize(p_process_single_file, files, n_jobs=n_jobs)

    for atoms, bonds, mols in results:
        all_atoms.extend(atoms)
        all_bonds.extend(bonds)
        all_mols.extend(mols)

    return {
        "atoms": pd.concat(all_atoms, ignore_index=True) if all_atoms else None,
        "bonds": pd.concat(all_bonds, ignore_index=True) if all_bonds else None,
        "mols": pd.concat(all_mols, ignore_index=True) if all_mols else None,
    }
