import sys
import time
from logging import Logger
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import load_hpc_result, permutation_importance
from utils import setup_logger


def filter_X(X: pd.DataFrame) -> pd.DataFrame:
    drop_qm_features = [
        "radius_of_gyration",
        "molecular_volume",
        "sterimol_L",
        "sterimol_Bmin",
        "sterimol_Bmax",
        "molecular_sasa",
        "solvation_energy_thf",
        "solvation_energy_cyclohexane",
        "solvation_energy_dmso",
        "avg_percentage_buried_volume",
        "min_percentage_buried_volume",
        "max_percentage_buried_volume",
        "std_percentage_buried_volume",
        "avg_atomic_sasa",
        "min_atomic_sasa",
        "max_atomic_sasa",
        "std_atomic_sasa",
        "min_partial_charge_thf",
        "max_partial_charge_thf",
        "std_partial_charge_thf",
        "min_partial_charge_cyclohexane",
        "max_partial_charge_cyclohexane",
        "std_partial_charge_cyclohexane",
        "min_partial_charge_dmso",
        "max_partial_charge_dmso",
        "std_partial_charge_dmso",
        "avg_bond_length",
        "min_bond_length",
        "max_bond_length",
        "std_bond_length",
        "avg_bond_stiffness",
        "min_bond_stiffness",
        "max_bond_stiffness",
        "std_bond_stiffness",
        "avg_effective_coordination_number",
        "min_effective_coordination_number",
        "max_effective_coordination_number",
        "std_effective_coordination_number",
    ]

    drop_topo_features = [
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
    ]

    return X.drop(
        [
            "avg_atomic_quadrupole_principal_invariant_3",  # quadrupole principal invariant 3 features correlate highly with the invariant 2 features, so can drop them
            "max_atomic_quadrupole_principal_invariant_3",
            "molecular_quadrupole_principal_invariant_3",
            "avg_atomic_dipole_dipole_interaction",  # the dipole dipole interaction between atoms would physically not be that influential on the solubility, can drop it
        ]
        + drop_qm_features
        + drop_topo_features,
        axis=1,
    )


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

    model_name = "_".join(Path(__file__).stem.split("_")[:-1])

    FILES_DIR = Path("/data/gent/489/vsc48953/ML_enhance") / model_name / "results"

    BASE: Path = Path(__file__).parent

    FILE_NAME: str = model_name + f"_{fold_id}"
    RESULTS_NAME: str = FILE_NAME + "_results.pkl"

    DF_FILE = BASE.parent / "processed_dataset_wo_metals_w_even_more_qm2.csv"
    SPLITS_FILE = BASE.parent / "splits.pkl"

    SAVE_FILE = BASE / Path(__file__).stem + ".csv"

    LOG_FILE: str = FILE_NAME + ".log"
    logger: Logger = setup_logger(BASE, LOG_FILE)

    df = pd.read_csv(DF_FILE)
    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    X_filtered = filter_X(X)
    y = df["solubility"]

    _, X_test, _, y_test = custom_train_test_split(SPLITS_FILE, fold_id, X_filtered, y)

    model_df: pd.DataFrame = load_hpc_result(FILES_DIR / RESULTS_NAME)
    model: BaseEstimator = model_df.loc[fold_id, "estimator"]

    logger.info("Begin model fit")
    start_fit = time.time()

    fi_df = get_FI(model, X_test, y_test, n_cpus)

    fit_time = time.time() - start_fit
    logger.info(f"Model fit took: {fit_time}s")

    fi_df.to_csv(SAVE_FILE, index_label="fold_id")


if __name__ == "__main__":
    main()
