import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

import ml_enhance
from ml_enhance import CorrelationFilter  # noqa: F401

sys.modules["utils"] = ml_enhance


def read_file(file: Path) -> dict[str, Any]:
    with file.open("rb") as f:
        return pickle.load(f)


def group_files_data(files: list[Path]) -> dict[str, np.ndarray[Any]]:
    models: list[BaseEstimator] = []
    train_r2s: list[float] = []
    test_r2s: list[float] = []
    train_MSEs: list[float] = []
    test_MSEs: list[float] = []
    for file in files:
        file_data = read_file(file)
        models.append(file_data["model"])

        train_r2s.append(file_data["train_r2"])
        train_MSEs.append(file_data["train_MSE"])

        test_r2s.append(file_data["test_r2"])
        test_MSEs.append(file_data["test_MSE"])

    return {
        "estimator": np.array(models),
        "train_r2": np.array(train_r2s),
        "test_r2": np.array(test_r2s),
        "train_MSE": np.array(train_MSEs),
        "test_MSE": np.array(test_MSEs),
    }


def save_combined(combined_data: dict[str, np.ndarray[Any]], path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(combined_data, f)


def get_coef(estimator: BaseEstimator) -> np.ndarray:
    return estimator.named_steps["predict"].coef_


def main() -> None:
    input_path: Path = Path(str(sys.argv[1]))
    output_file: Path = Path(str(sys.argv[2]))

    output_file.parent.mkdir(exist_ok=True)

    files: list[Path] = [file for file in input_path.glob("**/*") if file.is_file()]

    combined_dict = group_files_data(files)

    save_combined(combined_dict, output_file)


if __name__ == "__main__":
    main()
