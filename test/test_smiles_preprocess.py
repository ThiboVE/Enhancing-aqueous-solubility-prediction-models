from library import preprocess_smiles
from rdkit import Chem 
import pytest

import pytest

@pytest.mark.parametrize("bad_smiles", [
    "not_a_smiles",
    "C1CC",      # broken ring
])

def test_invalid_smiles_filtered(bad_smiles):
    assert preprocess_smiles(bad_smiles) is None


@pytest.mark.parametrize("single_atom", [
    "C",
    "[Na+]",
    "Cl",
])

def test_single_atom_filtered(single_atom):
    assert preprocess_smiles(single_atom) is None


@pytest.mark.parametrize("salted_smiles", [
    "CCO.[Na+]",
    "CCO.Cl",
    "c1ccccc1.[K+]",
])

def test_salt_molecules_filtered(salted_smiles):
    assert preprocess_smiles(salted_smiles) is None


@pytest.mark.parametrize("valid_smiles", [
    "CCO",
    "c1ccccc1",
    "O=C(O)C",
])

def test_valid_smiles_not_filtered(valid_smiles):
    result = preprocess_smiles(valid_smiles)
    assert result is not None


def test_explicit_hydrogens_added():
    result = preprocess_smiles("CCO")
    mol = Chem.MolFromSmiles(result)

    # No implicit hydrogens allowed
    for atom in mol.GetAtoms():
        assert atom.GetNumImplicitHs() == 0


def test_atom_mapping_added():
    result = preprocess_smiles("CCO")
    mol = Chem.MolFromSmiles(result)

    map_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

    # All atoms mapped
    assert all(num > 0 for num in map_nums)

    # Mapping unique
    assert len(map_nums) == len(set(map_nums))


