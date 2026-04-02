import pickle
import re
import sys
from pathlib import Path

import pandas as pd


def save_combined(combined_data: dict[int, pd.Series], path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(combined_data, f)


def main() -> None:
    input_path: Path = Path(str(sys.argv[1]))
    output_file: Path = Path(str(sys.argv[2]))

    output_file.parent.mkdir(exist_ok=True)

    files: list[Path] = [file for file in input_path.glob("**/*") if file.is_file()]

    FI_dict: dict[int, pd.Series] = {}
    for file in files:
        pattern_match = re.search(r"\d+", file.stem)
        fold_id = int(pattern_match.group())

        fold_df = pd.read_csv(file, index_col="feature")

        FI_dict[fold_id] = fold_df["r2_mean"]

    save_combined(FI_dict, output_file)


if __name__ == "__main__":
    main()
