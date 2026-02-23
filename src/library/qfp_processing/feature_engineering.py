import pandas as pd


class QFPFeatureEngineer:
    """
    Handles conformer-level feature processing.
    """
    def __init__(self, temperature: float):
        self.temperature = temperature

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public entry point for conformer-level feature processing.
        """

        df = self._remove_tensor_features(df)
        df = self._select_thermodynamic_features(df)
        df = self._aggregate_ir_regions(df)
        df = self._aggregate_atomic_features(df)
        df = self._aggregate_bond_features(df)

        return df

    def _remove_tensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_to_remove = ["molecular_dipole", "molecular_quadrupole", "molecular_polarizability", "atomic_dipole", "atomic_quadrupole", "atomic_polarizability"]

        df = df.drop(features_to_remove, axis='columns')

        return df

    def _select_thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        thermodynamic_features = ["gibbs_free_energy", "entropy", "heat_capacity"]

        def get_val_at_T(conformer_list, temperature=self.temperature):
            for T, val in conformer_list:
                if T == temperature:
                    return val
            return None

        for feature in thermodynamic_features:
            df[f'{feature}_{int(self.temperature)}K'] = df[feature].apply(get_val_at_T).astype("Float64")

        df = df.drop(thermodynamic_features, axis='columns')

        return df

    def _aggregate_ir_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _aggregate_atomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Naive approach of just taking the average, later we can look into deviding per atom type
        """
        return df

    def _aggregate_bond_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df