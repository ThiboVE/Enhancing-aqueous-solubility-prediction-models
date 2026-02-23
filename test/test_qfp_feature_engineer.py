import pandas as pd
import pytest

from library import QFPFeatureEngineer


@pytest.fixture
def mock_dataframe():
    """Mock DataFrame with all features needed to run QFPFeatureEngineer."""
    return pd.DataFrame(
        {
            # Tensor features
            "molecular_dipole": [1.0, 2.0],
            "molecular_quadrupole": [0.5, 1.0],
            "molecular_polarizability": [1.2, 3.6],
            "atomic_dipole": [[0.5], [0.7]],
            "atomic_quadrupole": [[1.5], [1.7]],
            "atomic_polarizability": [[1.5], [1.7]],
            # Thermodynamic features
            "gibbs_free_energy": [[(200, -10), (300, -12)], [(200, -9), (300, -11)]],
            "entropy": [[(200, 50), (300, 55)], [(200, 52), (300, 57)]],
            "heat_capacity": [[(200, 10), (300, 11)], [(200, 10.5), (300, 11.5)]],
            # IR features
            "normal_mode_frequencies": [
                [500, 1000, 2000, 3000],
                [600, 1200, 2100, 3100],
            ],
            "infrared_intensity": [[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]],
            "normal_modes": [[], []],
            # Atomic features
            "effective_coordination_number": [
                [(1, 2.0), (2, 1.0)],
                [(1, 1.5), (2, 2.5)],
            ],
            "partial_charge": [[(1, -0.2), (2, 0.1)], [(1, 0.0), (2, -0.1)]],
            "atomic_fukui_minus": [[(1, 0.3), (2, 0.4)], [(1, 0.35), (2, 0.45)]],
            "atomic_fukui_plus": [[(1, 0.5), (2, 0.6)], [(1, 0.55), (2, 0.65)]],
            "atomic_dipole_norm": [[(1, 1.0), (2, 2.0)], [(1, 1.1), (2, 2.1)]],
            "atomic_quadrupole_principal_invariant_2": [
                [(1, 3.0), (2, 5.0)],
                [(1, 3.5), (2, 5.5)],
            ],
            "atomic_quadrupole_principal_invariant_3": [
                [(1, 4.0), (2, 6.0)],
                [(1, 4.5), (2, 6.5)],
            ],
            "atomic_polarizability_mean": [[(1, 7.0), (2, 9.0)], [(1, 7.5), (2, 9.5)]],
            "atomic_polarizability_anisotropy": [
                [(1, 8.0), (2, 10.0)],
                [(1, 8.5), (2, 10.5)],
            ],
            "percentage_buried_volume": [
                [(1, 11.0), (2, 13.0)],
                [(1, 11.5), (2, 13.5)],
            ],
            "atomic_sasa": [[(1, 12.0), (2, 14.0)], [(1, 12.5), (2, 14.5)]],
            "partial_charge_water": [[(1, -0.3), (2, 0.2)], [(1, -0.25), (2, 0.25)]],
            "partial_charge_thf": [[(1, -0.4), (2, 0.3)], [(1, -0.35), (2, 0.35)]],
            "partial_charge_cyclohexane": [
                [(1, -0.5), (2, 0.4)],
                [(1, -0.45), (2, 0.45)],
            ],
            "partial_charge_dmso": [[(1, -0.6), (2, 0.5)], [(1, -0.55), (2, 0.55)]],
            # Interaction features
            "bond_energy": [[(0, 1, 10.0), (1, 2, 20.0)], [(0, 1, 15.0), (1, 2, 25.0)]],
            "bond_length": [[(0, 1, 1.0), (1, 2, 2.0)], [(0, 1, 1.5), (1, 2, 2.5)]],
            "bond_stiffness": [
                [(0, 1, 100.0), (1, 2, 200.0)],
                [(0, 1, 150.0), (1, 2, 250.0)],
            ],
            "overlap_integral": [
                [(0, 1, 0.1), (1, 2, 0.2)],
                [(0, 1, 0.15), (1, 2, 0.25)],
            ],
            "nuclear_repulsion": [
                [(0, 1, 5.0), (1, 2, 15.0)],
                [(0, 1, 7.5), (1, 2, 17.5)],
            ],
            "atomic_charge_dipole_interaction": [
                [(0, 1, 0.5), (1, 2, 1.5)],
                [(0, 1, 0.75), (1, 2, 1.75)],
            ],
            "atomic_charge_quadrupole_interaction": [
                [(0, 1, 0.6), (1, 2, 1.6)],
                [(0, 1, 0.8), (1, 2, 1.8)],
            ],
            "atomic_dipole_dipole_interaction": [
                [(0, 1, 0.7), (1, 2, 1.7)],
                [(0, 1, 0.85), (1, 2, 1.85)],
            ],
        }
    )


def test_remove_tensor_features(mock_dataframe):
    engineer = QFPFeatureEngineer(temperature=300)
    df_clean = engineer._remove_tensor_features(mock_dataframe)
    for col in [
        "molecular_dipole",
        "molecular_quadrupole",
        "molecular_polarizability",
        "atomic_dipole",
        "atomic_quadrupole",
        "atomic_polarizability",
    ]:
        assert col not in df_clean.columns


def test_select_thermodynamic_features(mock_dataframe):
    engineer = QFPFeatureEngineer(temperature=300)
    df_out = engineer._select_thermodynamic_features(mock_dataframe)
    for feature in ["gibbs_free_energy", "entropy", "heat_capacity"]:
        col_name = f"{feature}_300K"
        assert col_name in df_out.columns
    assert df_out.loc[0, "gibbs_free_energy_300K"] == -12


def test_aggregate_ir_regions(mock_dataframe):
    engineer = QFPFeatureEngineer(temperature=300)
    result = engineer._aggregate_ir_regions(mock_dataframe)
    for col in [
        "avg_ir_freq_1500",
        "avg_ir_intensity_1500",
        "avg_ir_freq_1500_2750",
        "avg_ir_intensity_1500_2750",
        "avg_ir_freq_2750_4000",
        "avg_ir_intensity_2750_4000",
    ]:
        assert col in result.columns
    assert "normal_mode_frequencies" not in result.columns


def test_aggregate_atomic_features(mock_dataframe):
    engineer = QFPFeatureEngineer(temperature=300)
    result = engineer._aggregate_atomic_features(mock_dataframe)
    for col in ["avg_effective_coordination_number", "avg_partial_charge"]:
        assert col in result.columns
    assert "partial_charge" not in result.columns


def test_aggregate_interaction_features(mock_dataframe):
    engineer = QFPFeatureEngineer(temperature=300)
    result = engineer._aggregate_interaction_features(mock_dataframe)
    for col in ["avg_bond_energy", "avg_bond_length"]:
        assert col in result.columns
    assert "bond_energy" not in result.columns
