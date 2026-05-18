from dataclasses import dataclass

import numpy as np
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem
from rdkit.Chem.rdchem import Bond

from ml_enhance.nn.featurizer.atom_featurizer import CustomMultiHotAtomFeaturizer
from ml_enhance.nn.featurizer.bond_featurizer import CustomMultiHotBondFeaturizer


@dataclass
class CustomSimpleMoleculeMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    """A custom SimpleMoleculeMolGraphFeaturizer with additional feature control."""

    keep_atom_features: list[bool] | None = None
    keep_bond_features: list[bool] | None = None
    keep_atoms: list[bool] | None = None
    keep_bonds: list[bool] | None = None

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.atom_featurizer, CustomMultiHotAtomFeaturizer) and self.keep_atom_features is not None:
            self.atom_featurizer.keep_features = self.keep_atom_features
        if isinstance(self.bond_featurizer, CustomMultiHotBondFeaturizer) and self.keep_bond_features is not None:
            self.bond_featurizer.keep_features = self.keep_bond_features

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if self.keep_atoms is None:
            self.keep_atoms = [True] * n_atoms
        if self.keep_bonds is None:
            self.keep_bonds = [True] * n_bonds

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            msg = (
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
            raise ValueError(msg)

        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            msg = (
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )
            raise ValueError(msg)

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array(
                [
                    self.atom_featurizer(atom) if self.keep_atoms[atom.GetIdx()] else self.atom_featurizer.zero_mask()
                    for atom in mol.GetAtoms()
                ],
                dtype=np.single,
            )

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index: list[list[int]] = [[], []]

        i = 0
        for u in range(n_atoms):
            for v in range(u + 1, n_atoms):
                bond: Bond | None = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    continue

                x_e = self.bond_featurizer(bond) if self.keep_bonds[bond.GetIdx()] else self.bond_featurizer.zero_mask()

                if bond_features_extra is not None:
                    x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

                E[i : i + 2] = x_e
                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])
                i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)
        return MolGraph(V, E, edge_index, rev_edge_index)
