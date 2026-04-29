import pickle

import pandas as pd
from sklearn.model_selection import RepeatedKFold


def main() -> None:
    df = pd.read_csv("../data/processed_dataset_wo_metals_w_even_more_qm2.csv")

    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    y = df["solubility"]

    repeated_kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=15)
    splits = list(repeated_kf.split(X, y))
    with open("splits.pkl", "wb") as f:
        pickle.dump(splits, f)


if __name__ == "__main__":
    main()
