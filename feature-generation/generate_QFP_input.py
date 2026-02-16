from rdkit.Chem import rdmolops
from rdkit import Chem
import pandas as pd
import json

DATA_SOURCE = "../data/AqSolDB/data_curated.csv"

def is_salt(mol: Chem.Mol) -> bool:
    """
    Return True if the SMILES is likely a salt (contains a non-organic fragment).
    """
    if mol is None:
        return False  # Invalid SMILES, treat as non-salt
    
    fragments = rdmolops.GetMolFrags(mol, asMols=True)
    
    if len(fragments) <= 1:
        return False  # Only one fragment, likely not a salt
    
    # for frag in fragments:
    #     # Check if fragment contains carbon
    #     has_carbon = any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms())
    #     # Get SMILES for comparison
    #     frag_smiles = Chem.MolToSmiles(frag, canonical=True)
    #     frag_symbol = ''.join([atom.GetSymbol() for atom in frag.GetAtoms()])
    #     is_counterion = frag_symbol in COMMON_COUNTERIONS
    #     # If any fragment has no carbon and is not a recognized organic fragment -> salt
    #     if not has_carbon and not is_counterion:
    #         return True
    #     # Also treat recognized counterions as salt fragments
    #     if is_counterion:
    #         return True
    
    return True

def atom_map_numbers(mol: Chem.Mol) -> str:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    mapped_smiles = Chem.MolToSmiles(mol, canonical=False)
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

    inputs = []
    idx = 0
    for mol_data in df.iterrows():
        _, data = mol_data

        smiles = data["SMILES"]
        mol = Chem.MolFromSmiles(smiles)

        if (mol is not None) and (not is_salt(mol)):
            mol_with_H = Chem.AddHs(mol)

            mapped_smiles = atom_map_numbers(mol_with_H)

            mol_repr = molecule_repr(idx+1, mapped_smiles)
            inputs.append(mol_repr)

            idx += 1

    # assert len(inputs) == 9982, f"number of smiles in the input does not match number of smiles in the dataset: {len(inputs)}, but should be 9982."

    json_obj = generate_json(inputs)

    with open("../data/QuantumFP/QFP_input.json", "w") as f:
        json.dump(json_obj, f, indent=4)

if __name__ == "__main__":
    main()