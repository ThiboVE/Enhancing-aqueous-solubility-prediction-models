import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from library import QuantumFPFileLoader


def test_list_output_files(tmp_path: Path):

    # Create fake files
    (tmp_path / "file1.json.gz").touch()
    (tmp_path / "file2.json.gz").touch()
    (tmp_path / "ignore.txt").touch()

    loader = QuantumFPFileLoader(tmp_path, property_dict={})

    files = loader.list_output_files()

    assert len(files) == 2
    assert all(f.suffix == ".gz" for f in files)


def test_build_conformer_dataframe():

    property_dict = {
        1: "energy",
        2: "dipole"
    }

    loader = QuantumFPFileLoader("dummy_path", property_dict)

    fake_data = [
        {
            "prop_id_1": -10.5,
            "prop_id_2": 3.2,
            "original_smiles": "CCO",
            "output_smiles": "CCO"
        },
        {
            "prop_id_1": -9.8,
            "prop_id_2": 2.9,
            "original_smiles": "CCO",
            "output_smiles": "CCO"
        }
    ]

    df = loader._build_conformer_dataframe(fake_data)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert "energy" in df.columns
    assert "dipole" in df.columns
    assert "original_smiles" in df.columns


def test_stream_conformer_dataframe(tmp_path: Path):

    property_dict = {1: "energy"}

    fake_data = [
        {
            "prop_id_1": -10.0,
            "original_smiles": "CCO",
            "output_smiles": "CCO"
        }
    ]

    file_path = tmp_path / "test.gz"

    with gzip.open(file_path, "wt") as f:
        json.dump(fake_data, f)

    loader = QuantumFPFileLoader(tmp_path, property_dict)

    generator = loader.stream_conformer_dataframe(file_path)

    df = next(generator)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert "energy" in df.columns