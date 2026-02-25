import pandas as pd
import pytest

from library import RDKitFeatureCalculator

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


@pytest.fixture
def calculator():
    return RDKitFeatureCalculator()


@pytest.fixture
def simple_df():
    return pd.DataFrame({"smiles": ["CC", "O"]})


@pytest.fixture
def invalid_df():
    return pd.DataFrame({"smiles": ["INVALID"]})


# ---------------------------------------------------------
# Descriptor Computation Tests
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "smiles, expected_molwt, expected_hdonors",
    [
        ("CC", 30.07, 0),  # ethane
        ("O", 18.01, 0),  # water
    ],
)
def test_compute_descriptors_values(
    calculator, smiles, expected_molwt, expected_hdonors
):
    df = pd.DataFrame({"smiles": [smiles]})
    result = calculator.compute_descriptors(df)

    assert "MolWt" in result.columns
    assert "NumHDonors" in result.columns

    assert result.loc[0, "MolWt"] == pytest.approx(expected_molwt, rel=1e-2)
    assert result.loc[0, "NumHDonors"] == expected_hdonors


def test_compute_descriptors_invalid_smiles(calculator, invalid_df):
    result = calculator.compute_descriptors(invalid_df)

    assert "MolWt" in result.columns
    assert result.loc[0, "MolWt"] is None


# ---------------------------------------------------------
# Integration Tests (Merging)
# ---------------------------------------------------------


def test_add_to_dataframe_preserves_original_columns(calculator, simple_df):
    result = calculator.add_to_dataframe(simple_df)

    assert "smiles" in result.columns
    assert "MolWt" in result.columns
    assert len(result) == len(simple_df)


def test_add_to_dataframe_no_row_change(calculator, simple_df):
    result = calculator.add_to_dataframe(simple_df)
    assert result.shape[0] == simple_df.shape[0]
