import pickle

import pandas as pd
from sklearn.model_selection import RepeatedKFold


def main() -> None:
    df = pd.read_csv("../data/processed_dataset_wo_metals_w_even_more_qm2.csv")

    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    y = df["solubility"]

    X_filtered = X.drop(
        [
            "avg_atomic_quadrupole_principal_invariant_3",  # quadrupole principal invariant 3 features correlate highly with the invariant 2 features, so can drop them
            "max_atomic_quadrupole_principal_invariant_3",
            "molecular_quadrupole_principal_invariant_3",
            "avg_atomic_dipole_dipole_interaction",  # the dipole dipole interaction between atoms would physically not be that influential on the solubility, can drop it
        ],
        axis=1,
    )

    repeated_kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=15)
    splits = list(repeated_kf.split(X_filtered, y))
    with open("splits.pkl", "wb") as f:
        pickle.dump(splits, f)


if __name__ == "__main__":
    main()
