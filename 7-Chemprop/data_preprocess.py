import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ml_enhance import QuantumFPFileLoader
from ml_enhance.chemprop import Config, process_files


def filter_files(files: Iterable[Path], used_ids: list[int]) -> list[Path]:
    used_files: list[Path] = []

    for file in files:
        matches = re.findall(r"\d+", file.name)
        file_id = int(matches[0])

        if file_id in used_ids:
            used_files.append(file)

    return used_files


# ── fold building ─────────────────────────────────────────────────────────────


def get_smiles_from_files(
    files: Iterable[Path],
    target_df: pd.DataFrame,
) -> list[str]:
    ids = [int(re.findall(r"\d+", file.name)[0]) for file in files]
    return target_df[target_df["id"].isin(ids)]["smiles"].tolist()


# ── main ──────────────────────────────────────────────────────────────────────


def save_df(df: pd.DataFrame | None, path: str) -> None:
    if df is None:
        pd.DataFrame().to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    target_df = pd.read_csv("../data/processed_dataset_rerun.csv")
    used_ids: list[int] = target_df["id"].apply(round).to_list()

    qfp_loader = QuantumFPFileLoader(Path("../data/QFP_rerun/output"))
    filelist: list[Path] = qfp_loader.list_output_files()
    used_files = np.array(filter_files(filelist, used_ids))

    config = Config(
        use_atom_features=False,
        use_bond_features=False,
        use_mol_features=False,
        use_custom_atom_featurizer=True,
        use_custom_bond_featurizer=False,
    )

    print("Begin processing all files")

    all_features = process_files(
        used_files,
        qfp_loader,
        use_atom_features=config.use_atom_features,
        use_bond_features=config.use_bond_features,
        use_mol_features=config.use_mol_features,
        n_jobs=5,
    )

    atom_feature_df = all_features["atom"]
    bond_feature_df = all_features["bond"]
    mol_feature_df = all_features["mol"]

    save_df(atom_feature_df, "needed_data/atom_features.csv")
    save_df(bond_feature_df, "needed_data/bond_features.csv")
    save_df(mol_feature_df, "needed_data/mol_features.csv")


if __name__ == "__main__":
    main()
