import sys
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from utils import CorrelationFilter, Files, process_job, setup_logger


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


def get_rf_coef(estimator: BaseEstimator) -> np.ndarray:
    return estimator.named_steps["predict"].feature_importances_


def setup_model(n_cpus: int) -> BaseEstimator:
    """Because the combination of hyperparameter tuning and RFE takes too long, i try to approximate this process by using an optimized set of parameters to do RFE."""
    rf_params: dict[str, float | int] = {
        "max_depth": 33,
        "max_features": 0.3,
        "min_samples_leaf": 1,
        "min_samples_split": 13,
        "n_estimators": 776,
    }

    pl_rf = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("remove_corr", CorrelationFilter(threshold=0.95)),
            ("transform", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scale", StandardScaler()),
            ("predict", RandomForestRegressor(**rf_params, random_state=40)),
        ]
    )

    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)

    return RFECV(pl_rf, cv=inner_kf, scoring="r2", n_jobs=n_cpus, importance_getter=get_rf_coef, verbose=12)


def main() -> None:
    fold_id = int(sys.argv[1])
    n_cpus = int(sys.argv[2])

    OUTPUT_DIR = Path("/data/gent/489/vsc48953/ML_enhance") / Path(__file__).stem
    OUTPUT_DIR.mkdir(exist_ok=True)

    BASE: Path = Path(__file__).parent

    FILE_NAME: str = Path(__file__).stem + f"_{fold_id}"

    DF_FILE = BASE.parent / "processed_dataset_wo_metals_w_even_more_qm2.csv"
    SPLITS_FILE = BASE.parent / "splits.pkl"
    RESULTS_FILE = OUTPUT_DIR / "results" / (FILE_NAME + "_results.pkl")

    FILES = Files(SPLITS_FILE=SPLITS_FILE, RESULTS_FILE=RESULTS_FILE)

    LOG_FILE: str = FILE_NAME + ".log"
    logger: Logger = setup_logger(BASE, LOG_FILE)

    df = pd.read_csv(DF_FILE)
    X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
    X_filtered = filter_X(X)
    y = df["solubility"]

    rf_search = setup_model(n_cpus)

    process_job(FILES, rf_search, X_filtered, y, fold_id, logger)


if __name__ == "__main__":
    main()
