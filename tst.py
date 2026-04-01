import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from ml_enhance import load_hpc_result
from ml_enhance.hpc_utils import custom_train_test_split


def filter_X(X: pd.DataFrame, rdkit_features_file: Path) -> pd.DataFrame:
    X = X.drop(
        [
            "avg_atomic_quadrupole_principal_invariant_3",  # quadrupole principal invariant 3 features correlate highly with the invariant 2 features, so can drop them
            "max_atomic_quadrupole_principal_invariant_3",
            "molecular_quadrupole_principal_invariant_3",
            "avg_atomic_dipole_dipole_interaction",  # the dipole dipole interaction between atoms would physically not be that influential on the solubility, can drop it
        ],
        axis=1,
    )

    with rdkit_features_file.open("r") as f:
        rdkit_feature_list: list[str] = json.load(f)

    mask = X.columns.isin(rdkit_feature_list)

    return X.iloc[:, ~mask]  # only QM features remaining


def get_FI(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, n_cpus: int) -> pd.DataFrame:
    scoring = {"r2": "r2", "MSE": "neg_mean_squared_error"}

    PFI = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=20, random_state=9, n_jobs=n_cpus)

    return pd.DataFrame(
        {
            f"{metric}_{stat}": getattr(r, f"importances_{stat}")
            for metric, r in PFI.items()
            for stat in ("mean", "std")
        },
        index=X_test.columns,
    )


def main() -> None:
    fold_id = int(sys.argv[1])
    n_cpus = int(sys.argv[2])

    splits_file = Path("hpc_splits.pkl")
    data_file = Path(r"data\RF_results\RF_qm_results.pkl")
    save_path = Path("test.csv")
    rdkit_file = Path(r"data\rdkit_feature_names.json")

    df = pd.read_csv("data/processed_dataset_wo_metals_w_even_more_qm2.csv")
    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    X_filtered = filter_X(X, rdkit_file)
    y = df["solubility"]

    _, X_test, _, y_test = custom_train_test_split(splits_file, fold_id, X_filtered, y)

    combo_df: pd.DataFrame = load_hpc_result(data_file, "topo+QM")
    model: BaseEstimator = combo_df.loc[fold_id, "estimator"]

    fi_df = get_FI(model, X_test, y_test, n_cpus)

    fi_df.to_csv(save_path, index_label="fold_id")


if __name__ == "__main__":
    main()
