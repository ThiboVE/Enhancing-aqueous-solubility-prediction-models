import pandas as pd
import json

DATA_SOURCE = "../data/AqSolDB/data_curated.csv"

def molecule_repr(id: str, mol_data: pd.Series) -> dict:
    smiles = mol_data["SMILES"]

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
    for mol_data in df.iterrows():
        idx, data = mol_data

        mol_repr = molecule_repr(idx+1, data)
        inputs.append(mol_repr)

    assert len(inputs) == 9982, f"number of smiles in the input does not match number of smiles in the dataset: {len(inputs)}, but should be 9982."

    json_obj = generate_json(inputs)

    with open("data/QuantumFP_input")

if __name__ == "__main__":
    main()