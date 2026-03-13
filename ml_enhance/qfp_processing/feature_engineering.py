"""Feature engineering module for QFP (Quantum Functional Property) processing.

This module provides conformer-level feature processing and aggregation,
including IR region aggregation, atomic feature aggregation, and interaction
feature aggregation for molecular property prediction.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


class QFPFeatureEngineer:
    """Handles conformer-level feature processing."""

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public entry point for conformer-level feature processing."""
        df = self._remove_tensor_features(df)
        df = self._select_thermodynamic_features(df)
        df = self._process_energy_features(df)
        df = self._aggregate_ir_regions(df)
        df = self._aggregate_atomic_features(df)
        return self._aggregate_interaction_features(df)

    def _remove_tensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_to_remove = [
            "molecular_dipole",
            "molecular_quadrupole",
            "molecular_polarizability",
            "atomic_dipole",
            "atomic_quadrupole",
            "atomic_polarizability",
        ]

        return df.drop(features_to_remove, axis=1, errors="ignore")

    def _select_thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        thermodynamic_features = ["gibbs_free_energy", "entropy", "heat_capacity"]

        def get_val_at_T(value_list: list[list[float, float]], temperature: float = self.temperature) -> float | None:
            """Extract the value of a thermodynamic property at a specific temperature.

            Args:
                value_list (list[list[float, float]]): list of values for a thermodynamic property at predifined temperaturs [[200, ...], [300, ...], [400, ...]]
                temperature (float): desired temperature value

            Returns:
                float | None: The value corresponding to the desired temperature, or None if the temperature is not present in the value_list.
            """
            for T, val in value_list:
                if temperature == T:
                    return val
            return None

        for feature in thermodynamic_features:
            df[f"{feature}_{int(self.temperature)}K"] = df[feature].apply(get_val_at_T).astype("float64")

        return df.drop(thermodynamic_features, axis=1, errors="ignore")

    def _process_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        energy_features = ["energy", "gibbs_free_energy_300K"]

        for feature in energy_features:
            values = df[feature].to_numpy()

            min_val = values.min()
            max_val = values.max()
            std_val = values.std()

            df[f"delta_{feature}"] = values - min_val
            df[f"{feature}_range"] = max_val - min_val
            df[f"{feature}_std"] = std_val

        df["rigid_flag"] = 1 if df["energy"].max() == df["energy"].min() else 0

        return df.drop("energy", axis=1)

    def _aggregate_ir_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate IR frequencies and intensities into defined regions.

        - <1500
        - 1500-2750
        - 2750-4000
        """
        new_features_list = []

        bins = [0, 1500, 2750, 4000]
        freq_labels = ["1500", "1500_2750", "2750_4000"]

        for freq_list, intensity_list in zip(df["normal_mode_frequencies"], df["infrared_intensity"], strict=True):
            freqs = np.array(freq_list)
            intensities = np.array(intensity_list)

            # Keep only physical frequencies (MARK: TODO: maybe take imaginary freqs into account one way or another)
            mask: np.ndarray[bool] = (freqs >= 0) & (freqs <= 4000)
            freqs = freqs[mask]
            intensities = intensities[mask]

            # Assign each frequency to a bin
            bin_indices = np.digitize(freqs, bins)

            feature_dict = {}

            for i, label in enumerate(freq_labels, start=1):
                idxs = np.where(bin_indices == i)[0]

                feature_dict[f"ir_mode_count_{label}"] = len(idxs)

                feature_dict[f"ir_centroid_freq_{label}"] = (
                    centroid_freq(freqs, intensities, idxs) if len(idxs) > 0 else 0.0
                )
                feature_dict[f"ir_norm_intensity_{label}"] = norm_intensity(intensities, idxs) if len(idxs) > 0 else 0.0

            new_features_list.append(feature_dict)

        new_features = pd.DataFrame(new_features_list).astype("Float64")

        # Concatenate with original dataframe
        df = pd.concat([df.reset_index(drop=True), new_features], axis=1)

        # Drop raw IR columns
        return df.drop(
            ["infrared_intensity", "normal_mode_frequencies", "normal_modes"],
            axis=1,
            errors="ignore",
        )

    def _aggregate_list_features(
        self, df: pd.DataFrame, feature_set: set[str], value_extract_function: Callable
    ) -> pd.DataFrame:
        for feature in feature_set:
            floats_only = df[feature].apply(value_extract_function)
            exploded = floats_only.explode()
            stats = exploded.groupby(level=0).agg(["mean", "min", "max"])
            stats = stats.fillna(0)
            stats.columns = [f"avg_{feature}", f"min_{feature}", f"max_{feature}"]
            df = df.join(stats.astype("float64"))

        return df

    def _aggregate_atomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Naive approach of just taking the average, later we can look into dividing per atom type."""
        # MARK: TODO: split into atom types and average for each type

        atomic_features = {
            "effective_coordination_number",
            "partial_charge",
            "atomic_fukui_minus",
            "atomic_fukui_plus",
            "atomic_dipole_norm",
            "atomic_quadrupole_principal_invariant_2",
            "atomic_quadrupole_principal_invariant_3",
            "atomic_polarizability_mean",
            "atomic_polarizability_anisotropy",
            "percentage_buried_volume",
            "atomic_sasa",
            "partial_charge_water",
            "partial_charge_thf",
            "partial_charge_cyclohexane",
            "partial_charge_dmso",
        }

        def atomic_value_extract_fn(lst: list[list[int, float]]) -> list[float]:
            return [t[1] for t in lst]

        new_df = self._aggregate_list_features(df, atomic_features, atomic_value_extract_fn)

        return new_df.drop(
            atomic_features,
            axis=1,
            errors="ignore",
        )

    def _aggregate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        interaction_features = {
            "bond_length",
            "bond_stiffness",
            "overlap_integral",
            "nuclear_repulsion",
            "atomic_charge_dipole_interaction",
            "atomic_charge_quadrupole_interaction",
            "atomic_dipole_dipole_interaction",
            "bond_energy",
        }

        def interaction_value_extract_fn(lst: list[list[int, int, float]]) -> list[float]:
            return [inner[-1] for inner in lst] if lst else [0.0]

        new_df = self._aggregate_list_features(df, interaction_features, interaction_value_extract_fn)

        new_df["num_heavy_H_bonds"] = (
            new_df["bond_energy"].apply(len).astype("float64")
        )  # deal with the case that there are no bond energies (absence of heavy atom-H bonds)

        return new_df.drop(
            interaction_features,
            axis=1,
            errors="ignore",
        )


def centroid_freq(freqs: np.ndarray[float], intensities: np.ndarray[float], mask: np.ndarray[bool]) -> float:
    r"""Region centroid frequency for a region: v_k,R = \frac{\\sum_{i \\in R} v_k,i I_k,i}{\\sum_{i \\in R} v_k,i}."""
    return np.dot(freqs[mask], intensities[mask]) / intensities[mask].sum()


def norm_intensity(intensities: np.ndarray[float], mask: np.ndarray[bool]) -> float:
    r"""Normalized intensity for a region: I_k,R = \frac{\\sum_{i \\in R} I_k,i}{\\sum_{i \\in all} I_k,i}."""
    return intensities[mask].sum() / intensities.sum()
