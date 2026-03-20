from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from ml_enhance import NumpyJSONCache


def save_fold_idxs(
    fold: tuple[np.ndarray, np.ndarray], fold_id: int, n_repeats: int, n_splits: int, file_name: str
) -> None:
    train_idxs, test_idxs = fold

    fold_info = {
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "repeat_num": fold_id // n_splits,
        "split_num": fold_id % n_splits,
        "train_idxs": train_idxs,
        "test_idxs": test_idxs,
    }

    name, suffix = file_name.split(".")

    file_path = Path(name + f"_fold_{fold_id}" + suffix)

    cache = NumpyJSONCache(file_path)
    cache.dump(fold_info, indent=4)

    print(f"wrote fold {fold_id} to {file_path!s}")


def main() -> None:
    df = pd.read_csv("../data/processed_dataset_wo_metals_w_even_more_qm2.csv")

    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    y = df["solubility"]

    X = X.drop(
        [
            "avg_atomic_quadrupole_principal_invariant_3",  # quadrupole principal invariant 3 features correlate highly with the invariant 2 features, so can drop them
            "max_atomic_quadrupole_principal_invariant_3",
            "molecular_quadrupole_principal_invariant_3",
            "avg_atomic_dipole_dipole_interaction",  # the dipole dipole interaction between atoms would physically not be that influential on the solubility, can drop it
        ],
        axis=1,
    )

    repeated_kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=15)
    splits = list(repeated_kf.split(X, y))
    # with open("splits_combo.pkl", "wb") as f:
    #     pickle.dump(splits, f)

    # folds = list(repeated_kf.split(X, y))

    # n_repeats = repeated_kf.n_repeats
    # n_splits = repeated_kf.get_n_splits() // n_repeats

    # for fold_id, fold in enumerate(folds):
    #     save_fold_idxs(fold, fold_id, n_repeats, n_splits, "test.json")


if __name__ == "__main__":
    main()
