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
    processed = preprocess_to_chemprop_format(df, featurizer)

    print(processed[0].x_d)


if __name__ == "__main__":
    main()
