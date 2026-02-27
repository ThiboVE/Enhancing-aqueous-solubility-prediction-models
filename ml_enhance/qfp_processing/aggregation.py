import numpy as np
import pandas as pd
from scipy.constants import (
    Avogadro,  # 1/mol
    Boltzmann,  # in J/K
)


class ConformerAggregator:
    """Aggregates conformer-level features to molecule-level features."""

    def __init__(self, temperature: float = 300.0) -> None:
        self.temperature = temperature
        self.k_B: float = Boltzmann * Avogadro * 0.000239005736  # kcal/mol K

    def thermal_average(self, df: pd.DataFrame) -> pd.Series:
        """Performs Boltzmann (thermal) averaging of the numeric features of a set of conformers.

        Args:
            df (pd.DataFrame): dataframe, where each row is a conformer.

        Returns:
            pd.Series: Features for a molecule as Boltzmann average over its conformers.

        """
        G = df["gibbs_free_energy_300K"].to_numpy()
        delta_G = G - G.min()
        boltzmann_factors = np.exp(-delta_G / (self.k_B * self.temperature))
        weights = boltzmann_factors / boltzmann_factors.sum()

        result = {"smiles": df["original_smiles"].iloc[0]}

        float_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in float_cols:
            result[col] = np.dot(weights, df[col].to_numpy()).astype(pd.Float64Dtype)

        return pd.Series(result)
