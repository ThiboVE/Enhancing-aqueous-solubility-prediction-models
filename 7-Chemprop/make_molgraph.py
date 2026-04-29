import pickle

import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer


def preprocess_to_chemprop_format(
    raw_data: pd.DataFrame, featurizer: SimpleMoleculeMolGraphFeaturizer
) -> MoleculeDataset:
    all_data: list[MoleculeDatapoint] = [
        MoleculeDatapoint.from_smi(smi, [sol], keep_h=True)
        for _, (smi, sol) in raw_data[["smiles", "solubility"]].iterrows()
    ]

    return MoleculeDataset(all_data, featurizer)


def main() -> None:
    df = pd.read_csv("data/processed_dataset.csv")

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dataset = preprocess_to_chemprop_format(df, featurizer)

    with open("data/chemprop_dataset1.pkl", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
