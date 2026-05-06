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

import json
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

import ml_enhance
from ml_enhance import CorrelationFilter, parse_filename  # noqa: F401

sys.modules["utils"] = ml_enhance
sys.modules["correlation_filter"] = ml_enhance


def read_pkl_summary(file: Path) -> dict[str, Any]:
    with file.open("rb") as f:
        file_data = pickle.load(f)

    return {
        "estimator": file_data["model"],
        "train_r2": file_data["train_r2"],
        "test_r2": file_data["test_r2"],
        "train_MSE": file_data["train_MSE"],
        "test_MSE": file_data["test_MSE"],
    }


def read_pfi_csv(file: Path) -> dict[str, Any]:
    df = pd.read_csv(file, index_col="feature")
    return {"pfi_df": df[["r2_mean", "r2_std", "MSE_mean", "MSE_std"]]}


def read_shap_csv(file: Path) -> dict[str, Any]:
    df = pd.read_csv(file, index_col="feature")
    return {"shap_df": df[["shap_mean_abs", "shap_std_abs", "shap_mean_signed", "shap_std_signed"]]}


def read_json(file: Path) -> dict[str, Any]:
    with file.open("r") as f:
        return json.load(f)


def read_pt(file: Path) -> dict[str, Any]:
    return torch.load(file, weights_only=False)


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


def save_pkl(combined_data: dict[str, np.ndarray[Any]], path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(combined_data, f)


def get_coef(estimator: BaseEstimator) -> np.ndarray:
    """Function not used in code but, required to properly read in the results."""
    return estimator.named_steps["predict"].coef_


def get_rf_coef(estimator: BaseEstimator) -> np.ndarray:
    """Function not used in code but, required to properly read in the results."""
    return estimator.named_steps["predict"].feature_importances_


def process_pkl_files(pkl_files: list[Path], output_file: Path) -> None:
    if len(pkl_files) == 0:
        return

    pkl_grouped = group_files(pkl_files, read_pkl_summary)
    pkl_grouped_np = {k: np.array(v) for k, v in pkl_grouped.items()}

    save_pkl(pkl_grouped_np, output_file)


def process_pfi_files(csv_files: list[Path], output_file: Path) -> None:
    if len(csv_files) == 0:
        return

    csv_grouped = group_files(csv_files, read_pfi_csv)

    sizes = csv_grouped.get("size", [None] * len(csv_grouped["fold_id"]))
    keys = list(zip(csv_grouped["fold_id"], sizes, strict=True))

    FI_df = pd.concat(csv_grouped["pfi_df"], keys=keys, names=["fold_id", "size", "feature"]).reset_index()
    FI_df.to_csv(output_file, index=False)


def process_shap_files(csv_files: list[Path], output_file: Path) -> None:
    if len(csv_files) == 0:
        return

    csv_grouped = group_files(csv_files, read_shap_csv)

    sizes = csv_grouped.get("size", [None] * len(csv_grouped["fold_id"]))
    keys = list(zip(csv_grouped["fold_id"], sizes, strict=True))

    shap_df = pd.concat(csv_grouped["shap_df"], keys=keys, names=["fold_id", "size", "feature"]).reset_index()
    shap_df.to_csv(output_file, index=False)


def process_json_files(json_files: list[Path], output_file: Path) -> None:
    if len(json_files) == 0:
        return

    json_grouped = group_files(json_files, read_json)
    json_df = pd.DataFrame(json_grouped)
    json_df.to_csv(output_file, index=False)


def process_pt_files(pt_files: list[Path], output_file: Path) -> None:
    if len(pt_files) == 0:
        return

    pt_dict = {}
    for file in pt_files:
        info = parse_filename(file)
        data = read_pt(file)
        combined = {**info, **data}

        pt_dict[f"fold_{info['fold_id']}"] = combined

    torch.save(pt_dict, output_file)


def main() -> None:
    input_path: Path = Path(str(sys.argv[1]))

    dirname = input_path.name
    modelname = dirname.split("_")[1]

    output_dir = Path(f"data/{modelname}_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{dirname}_results.pkl"
    PFI_output_file = output_dir / f"{dirname}_PFI_results.csv"
    SHAP_output_file = output_dir / f"{dirname}_SHAP_results.csv"
    json_output_file = output_dir / f"{dirname}_results.json"
    pt_output_file = output_dir / f"{dirname}_results.pt"

    storage_folder: Path = Path(r"C:\Users\thibo\Downloads\hpc_results")  # local path, update per machine

    files: list[Path] = [file for file in input_path.glob("**/*") if file.is_file()]

    if (input_path / "results").exists():
        for file in files:
            file.rename(input_path / file.name)
        (input_path / "results").rmdir()

    pkl_files: list[Path] = [file for file in input_path.glob("**/*.pkl") if file.is_file()]
    process_pkl_files(pkl_files, output_file)

    pfi_files: list[Path] = [file for file in input_path.glob("**/*PFI*.csv") if file.is_file()]
    process_pfi_files(pfi_files, PFI_output_file)

    shap_files: list[Path] = [file for file in input_path.glob("**/*SHAP*.csv") if file.is_file()]
    process_shap_files(shap_files, SHAP_output_file)

    json_files: list[Path] = [file for file in input_path.glob("**/*.json") if file.is_file()]
    process_json_files(json_files, json_output_file)

    pt_files: list[Path] = [file for file in input_path.glob("**/*.pt") if file.is_file()]
    process_pt_files(pt_files, pt_output_file)

    input_path.rename(storage_folder / input_path)


if __name__ == "__main__":
    main()
