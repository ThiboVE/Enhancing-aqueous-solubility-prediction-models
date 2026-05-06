from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold


def build_nested_cv_splits(
    target_df: pd.DataFrame,
    outer_cv: RepeatedKFold,
    inner_cv: KFold,
    id_col: str = "id",
) -> dict[str, Any]:
    ids = target_df[id_col].to_numpy()

    nested_splits = {}

    for outer_i, (outer_tr_idx, outer_te_idx) in enumerate(outer_cv.split(ids)):
        outer_train_ids = ids[outer_tr_idx]
        outer_test_ids = ids[outer_te_idx]

        inner_folds: list[dict[str, np.ndarray]] = []

        for inner_tr_idx, inner_val_idx in inner_cv.split(outer_train_ids):
            inner_train_ids = outer_train_ids[inner_tr_idx]
            inner_val_ids = outer_train_ids[inner_val_idx]

            inner_folds.append(
                {
                    "train_ids": inner_train_ids,
                    "val_ids": inner_val_ids,
                }
            )

        nested_splits[f"outer_fold_{outer_i}"] = {
            "train_ids": outer_train_ids,
            "test_ids": outer_test_ids,
            "inner_folds": inner_folds,
        }

    return nested_splits


def main() -> None:
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=15)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    target_df = pd.read_csv("../data/processed_dataset_rerun.csv")[["smiles", "id", "solubility"]]

    target_df["id"] = target_df["id"].apply(round)
    # target_df.to_csv("needed_data/target_df.csv")

    nested_splits = build_nested_cv_splits(target_df, outer_cv, inner_cv)

    # torch.save(nested_splits, "needed_data/chemprop_splits.pt")


if __name__ == "__main__":
    main()
