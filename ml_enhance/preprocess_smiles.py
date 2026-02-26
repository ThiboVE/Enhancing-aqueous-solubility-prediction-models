"""Module containing functions to preprocess SMILES strings for QFP input file preparation."""

from rdkit import Chem
from rdkit.Chem import rdmolops


def is_salt(mol: Chem.Mol | None) -> bool:
    """Determine whether the input RDKit Mol object is a salt based on the number of fragments.

    Args:
        mol (Chem.Mol | None): The input RDKit Mol object. If None, the function returns False.

    Returns:
        bool: Whether the RDKit Mol object corresponds to a salt.
    """
    if mol is None:
        return False  # Invalid SMILES, treat as non-salt

    fragments = rdmolops.GetMolFrags(mol, asMols=True)

    return len(fragments) > 1


def is_atom(mol: Chem.Mol) -> bool:
    """Check whether the input RDKit Mol object contains only a single heavy atom.

    Args:
        mol (Chem.Mol): The RDKit Mol object.

    Returns:
        bool: Whether the input Mol object only contains a single heavy atom.
    """
    return mol.GetNumHeavyAtoms() == 1


def set_atom_map_numbers(mol: Chem.Mol) -> Chem.Mol:
    """Assign 1-based atom map numbers based on the atom indices.

    Args:
        mol (Chem.Mol): Input RDKit Mol object.

    Returns:
        Chem.Mol: The RDKit Mol object with atom map numbers assigned.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    return mol


def get_preprocessed_smiles(smiles: str) -> str | None:
    """Preprocesses a SMILES for the QFP program by removing any salts, single atoms, or invalid SMILES. It also explicitly adds hydrogen atoms and sets map index numbers to each atom.

    Args:
        smiles (str): Smiles representation of the molecule.

    Returns:
        str: An explicit-hydrogen mapped SMILES representation of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None or is_salt(mol) or is_atom(mol):
        return None

    mol_with_H = Chem.AddHs(mol)

    mapped_mol = set_atom_map_numbers(mol_with_H)

    return Chem.MolToSmiles(mapped_mol, canonical=True)
