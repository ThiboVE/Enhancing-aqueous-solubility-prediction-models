import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def filter_X_combo(X: pd.DataFrame) -> pd.DataFrame:
    drop_qm_features = [
        "avg_atomic_quadrupole_principal_invariant_3",  # quadrupole principal invariant 3 features correlate highly with the invariant 2 features, so can drop them
        "max_atomic_quadrupole_principal_invariant_3",
        "molecular_quadrupole_principal_invariant_3",
        "avg_atomic_dipole_dipole_interaction",  # the dipole dipole interaction between atoms would physically not be that influential on the solubility, can drop it
    ]

    drop_topo_features = [
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
    ]

    return X.drop(drop_qm_features + drop_topo_features, axis=1)


def filter_X_topo(X: pd.DataFrame, rdkit_features_file: Path) -> pd.DataFrame:
    with rdkit_features_file.open("r") as f:
        rdkit_feature_list: list[str] = json.load(f)

    pattern = "|".join(rdkit_feature_list)
    mask = np.array([bool(re.search(pattern, feature)) for feature in X.columns])

    X = X.iloc[:, mask]  # only topological features remaining

    drop_topo_features = [
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
    ]

    return X.drop(drop_topo_features, axis=1, errors="ignore")
