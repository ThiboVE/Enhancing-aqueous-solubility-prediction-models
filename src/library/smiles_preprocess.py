from rdkit.Chem import rdmolops
from rdkit import Chem

def is_salt(mol: Chem.Mol) -> bool:
    """
    Return True if the SMILES is likely a salt.
    """
    if mol is None:
        return False  # Invalid SMILES, treat as non-salt
    
    fragments = rdmolops.GetMolFrags(mol, asMols=True)
    
    if len(fragments) <= 1:
        return False  # Only one fragment, likely not a salt
    
    return True

def is_atom(mol: Chem.Mol) -> bool:
    return mol.GetNumHeavyAtoms() == 1

def atom_map_numbers(mol: Chem.Mol) -> str:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    return mol

def preprocess_smiles(smiles: str) -> str:
    """
    Preprocess SMILES for the QFP program:
    - removes any salts, single atoms or non-valid SMILES
    - explicitly add all the hydrogens
    - map index numbers to each atom

    params: unprocessed SMILES
    returns: processes SMILES
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None or is_salt(mol) or is_atom(mol):
        return None
    
    mol_with_H = Chem.AddHs(mol)

    mapped_mol = atom_map_numbers(mol_with_H)
    
    mapped_smiles = Chem.MolToSmiles(mapped_mol, canonical=True)
    return mapped_smiles