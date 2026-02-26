import numpy as np
import pandas as pd

from ml_enhance import ConformerAggregator  # replace with actual import


def test_thermal_average_basic() -> None:
    """Test thermal_average with a small, known dataset."""
    # Small test dataframe: 2 conformers
    df = pd.DataFrame(
        {
            "original_smiles": ["C", "C"],
            "gibbs_free_energy_300K": [0.0, 1.0],  # kcal/mol
            "dipole": [1.0, 3.0],
            "homo_lumo_gap": [5.0, 7.0],
            "entropy_300K": [10.0, 20.0],
        }
    )

    aggregator = ConformerAggregator(temperature=300.0)
    result = aggregator.thermal_average(df)

    # Compute expected weights manually
    k_B = aggregator.k_B
    G = np.array([0.0, 1.0])
    w = np.exp(-G / (k_B * aggregator.temperature))
    w /= w.sum()

    # Expected weighted averages
    expected_dipole = np.dot(w, [1.0, 3.0])
    expected_gap = np.dot(w, [5.0, 7.0])
    expected_entropy = np.dot(w, [10.0, 20.0])

    # Assertions
    assert result["smiles"] == "C"
    assert np.isclose(result["dipole"], expected_dipole)
    assert np.isclose(result["homo_lumo_gap"], expected_gap)
    assert np.isclose(result["entropy_300K"], expected_entropy)


def test_only_float_columns_aggregated() -> None:
    """Ensure only float columns are aggregated."""
    df = pd.DataFrame(
        {
            "original_smiles": ["C", "C"],
            "gibbs_free_energy_300K": [0.0, 0.0],
            "dipole": [1.0, 2.0],
            "label": ["A", "B"],  # non-float
        }
    )
    aggregator = ConformerAggregator()
    result = aggregator.thermal_average(df)

    # Non-float column should not appear
    assert "label" not in result
    # Float column should be averaged
    assert np.isclose(result["dipole"], 1.5)
