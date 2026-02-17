from rdkit.Chem import rdmolops
from rdkit import RDLogger
from rdkit import Chem
import pandas as pd
import json

DATA_SOURCE = "../data/AqSolDB/data_curated.csv"

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
        atom.SetAtomMapNum(atom.GetIdx())

    mapped_smiles = Chem.MolToSmiles(mol, canonical=False)
    return mapped_smiles

def preprocess_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None or is_salt(mol) or is_atom(mol):
        return None
    
    mol_with_H = Chem.AddHs(mol)

    mapped_smiles = atom_map_numbers(mol_with_H)
    
    return mapped_smiles

def molecule_repr(id: str, smiles: str) -> dict:
    return {
        "id": id,
        "smiles": smiles
    }

def generate_json(inputs: list[dict]) -> dict:
    input_file = {
        "options": {
            "max_conformers": 32,
            "max_microstates": 8,
            "metadynamics": False
        },
    }

    input_file.update(
        {
            "inputs": inputs
        }
    )

    return input_file


def main():
    df = pd.read_csv(DATA_SOURCE)
    preprocessed_smiles = [preprocess_smiles(smiles) for smiles in df["SMILES"]]

    cleaned_smiles = list(filter(lambda x: x is not None, preprocessed_smiles))
    
    inputs = [molecule_repr(idx, smiles) for idx, smiles in enumerate(cleaned_smiles)]

    json_obj = generate_json(inputs)

    with open("../data/QuantumFP/QFP_input.json", "w") as f:
        json.dump(json_obj, f, indent=4)

if __name__ == "__main__":
    main()