"""Script used to process calculations downloaded from the HPC.

The calculations are in the following format:

directory 'calculation_name'/results contains files

'calculation_name'_id=..._results.pkl
'calculation_name'_id=..._PFI_results.csv

for objective 1 and

'calculation_name'_id=..._size=..._results.pkl
'calculation_name'_id=..._size=..._PFI_results.csv

for objective 2.

These files are pulled out of the results directory and combined into one two files: one for the model and performance (.pkl file) and one for the permutation feature importance (PFI) results (.csv file).
"""

import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import ml_enhance
from ml_enhance import CorrelationFilter, parse_filename  # noqa: F401

sys.modules["utils"] = ml_enhance


def read_file(file: Path) -> dict[str, Any]:
    with file.open("rb") as f:
        return pickle.load(f)


def read_pkl_summary(file: Path) -> dict[str, Any]:
    file_data = read_file(file)

    return {
        "estimator": file_data["model"],
        "train_r2": file_data["train_r2"],
        "test_r2": file_data["test_r2"],
        "train_MSE": file_data["train_MSE"],
        "test_MSE": file_data["test_MSE"],
    }


def read_pfi_csv(file: Path) -> dict[str, Any]:
    df = pd.read_csv(file, index_col="feature")

    return {"feature_importance": df["r2_mean"]}


def group_files(
    files: list[Path],
    reader_fn: Callable[[Path], dict[str, Any]],
) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}

    for file in files:
        info = parse_filename(file)

        data = reader_fn(file)

        combined = {**info, **data}

        for key, value in combined.items():
            grouped.setdefault(key, []).append(value)

    return grouped


def save_combined(combined_data: dict[str, np.ndarray[Any]], path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(combined_data, f)


def get_coef(estimator: BaseEstimator) -> np.ndarray:
    """Function not used in code but, required to properly read in the results."""
    return estimator.named_steps["predict"].coef_


def get_rf_coef(estimator: BaseEstimator) -> np.ndarray:
    """Function not used in code but, required to properly read in the results."""
    return estimator.named_steps["predict"].feature_importances_


def main() -> None:
    input_path: Path = Path(str(sys.argv[1]))

    dirname = input_path.name
    modelname = dirname.split("_")[1]

    output_name = f"data/{modelname}_results/{dirname}"
    output_file = Path(output_name + "_results.pkl")
    PFI_output_file = Path(output_name + "_PFI_results.csv")

    output_file.parent.mkdir(exist_ok=True)

    storage_folder: Path = Path(r"C:\Users\thibo\Downloads\hpc_results")

    files: list[Path] = [file for file in input_path.glob("**/*") if file.is_file()]

    if (input_path / "results").exists():
        for file in files:
            file.rename(input_path / file.name)

        (input_path / "results").rmdir()

    pkl_files: list[Path] = [file for file in input_path.glob("**/*.pkl") if file.is_file()]
    pkl_grouped = group_files(pkl_files, read_pkl_summary)
    pkl_grouped_np = {k: np.array(v) for k, v in pkl_grouped.items()}

    save_combined(pkl_grouped_np, output_file)

    csv_files: list[Path] = [file for file in input_path.glob("**/*.csv") if file.is_file()]
    if len(csv_files) != 0:
        csv_grouped = group_files(csv_files, read_pfi_csv)

        sizes = csv_grouped.get("size", [None] * len(csv_grouped["fold_id"]))
        keys = list(zip(csv_grouped["fold_id"], sizes, strict=True))

        FI_df = pd.concat(
            csv_grouped["feature_importance"], keys=keys, names=["fold_id", "size", "feature"]
        ).reset_index(name="r2_mean")

        FI_df.to_csv(PFI_output_file, index=False)

    input_path.rename(storage_folder / input_path)


if __name__ == "__main__":
    main()
