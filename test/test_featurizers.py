import numpy as np
from rdkit import Chem

from ml_enhance.nn import get_custom_atom_featurizer, get_featurizer


def test_custom_atom_featurizer() -> None:
    atom_featurizer = get_custom_atom_featurizer()

    atom_to_featurize = Chem.MolFromSmiles("CC").GetAtoms()[0]

    atom_features: np.ndarray = atom_featurizer(atom_to_featurize)

    result = np.array(
        [
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0.12011,
        ]
    )

    assert np.allclose(atom_features, result)


def test_mol_featuriezer() -> None:
    mol_to_featurize = Chem.MolFromSmiles("CC")

    featurizer = get_featurizer()

    mol_graph = featurizer(mol_to_featurize)

    print(mol_graph)
