from library import preprocess_smiles
import pandas as pd
import json

DATA_SOURCE = "../data/AqSolDB/data_curated.csv"

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