import pandas as pd
import pytest

from library import QFPFeatureEngineer


def make_mock_dataframe():
    """
    Create a minimal DataFrame with just enough columns
    for testing the current engineer methods.
    """
    return pd.DataFrame(
        {
            "molecular_dipole": [1.0, 2.0],
            "molecular_quadrupole": [0.5, 1.0],
            "molecular_polarizability": [1.2, 3.6],
            "atomic_dipole": [[0.5], [0.7]],
            "atomic_quadrupole": [[1.5], [1.7]],
            "atomic_polarizability": [[1.5], [1.7]],
            "gibbs_free_energy": [[(200, -10), (300, -12)], [(200, -9), (300, -11)]],
            "entropy": [[(200, 50), (300, 55)], [(200, 52), (300, 57)]],
            "heat_capacity": [[(200, 10), (300, 11)], [(200, 10.5), (300, 11.5)]],
        }
    )


def test_remove_tensor_features():
    df = make_mock_dataframe()
    engineer = QFPFeatureEngineer(temperature=300)

    df_clean = engineer._remove_tensor_features(df)

    for col in [
        "molecular_dipole",
        "molecular_quadrupole",
        "molecular_polarizability",
        "atomic_dipole",
        "atomic_quadrupole",
        "atomic_polarizability",
    ]:
        assert col not in df_clean.columns


def test_select_thermodynamic_features():
    df = make_mock_dataframe()
    engineer = QFPFeatureEngineer(temperature=300)

    df_out = engineer._select_thermodynamic_features(df)

    # New columns should exist
    for feature in ["gibbs_free_energy", "entropy", "heat_capacity"]:
        col_name = f"{feature}_300K"
        assert col_name in df_out.columns

    # Values at 300K should match
    assert df_out.loc[0, "gibbs_free_energy_300K"] == -12
    assert df_out.loc[1, "entropy_300K"] == 57
    assert df_out.loc[0, "heat_capacity_300K"] == 11


def test_aggregate_ir_regions_basic():
    engineer = QFPFeatureEngineer(temperature=300)

    df = pd.DataFrame(
        {
            "normal_mode_frequencies": [
                [500, 1000, 2000, 3000],  # covers all bins
            ],
            "infrared_intensity": [
                [1.0, 2.0, 3.0, 4.0],
            ],
            "normal_modes": [[]],  # dummy column to be dropped
        }
    )

    result = engineer._aggregate_ir_regions(df)

    # ---- Expected means ----
    # <1500: 500, 1000
    assert result.loc[0, "avg_ir_freq_1500"] == pytest.approx(750.0)
    assert result.loc[0, "avg_ir_intensity_1500"] == pytest.approx(1.5)

    # 1500-2750: 2000
    assert result.loc[0, "avg_ir_freq_1500_2750"] == pytest.approx(2000.0)
    assert result.loc[0, "avg_ir_intensity_1500_2750"] == pytest.approx(3.0)

    # 2750-4000: 3000
    assert result.loc[0, "avg_ir_freq_2750_4000"] == pytest.approx(3000.0)
    assert result.loc[0, "avg_ir_intensity_2750_4000"] == pytest.approx(4.0)

    # ---- Raw columns should be removed ----
    assert "normal_mode_frequencies" not in result.columns
    assert "infrared_intensity" not in result.columns
    assert "normal_modes" not in result.columns
