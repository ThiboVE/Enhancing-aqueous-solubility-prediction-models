import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
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
    df = pd.read_csv("needed_data/target_df.csv")

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dataset = preprocess_to_chemprop_format(df, featurizer)
    loader = build_dataloader(dataset, shuffle=False, batch_size=1, num_workers=0)

    batch = next(iter(loader))
    bmg, V_d, X_d, *_ = batch

    print(bmg.V[0])


if __name__ == "__main__":
    main()
