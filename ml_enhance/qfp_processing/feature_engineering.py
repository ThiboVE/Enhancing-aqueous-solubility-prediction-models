"""Feature engineering module for QFP (Quantum Functional Property) processing.

This module provides conformer-level feature processing and aggregation,
including IR region aggregation, atomic feature aggregation, and interaction
feature aggregation for molecular property prediction.
"""

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
        df = self._aggregate_ir_regions(df)
        df = self._aggregate_atomic_features(df)
        df = self._aggregate_interaction_features(df)
        return self._aggregate_bond_energy(df)

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
            df[f"{feature}_{int(self.temperature)}K"] = df[feature].apply(get_val_at_T).astype("Float64")

        return df.drop(thermodynamic_features, axis=1, errors="ignore")

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

    def _aggregate_atomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Naive approach of just taking the average, later we can look into deviding per atom type."""
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

        for feature in atomic_features:
            df[f"avg_{feature}"] = df[feature].apply(self.get_atomic_mean).astype("Float64")

        return df.drop(
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
        }

        for feature in interaction_features:
            df[f"avg_{feature}"] = df[feature].apply(self.get_interaction_mean).astype("Float64")

        return df.drop(
            interaction_features,
            axis=1,
            errors="ignore",
        )

    def _aggregate_bond_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specifically deal with the heavy atom-H bond energy.

        This function also explicitly deals with the case that there are no bond energies (absence of heavy atom-H bonds).
        """
        df["avg_bond_energy"] = df["bond_energy"].apply(self.get_interaction_mean).astype("Float64")

        df["num_heavy_H_bonds"] = df["bond_energy"].apply(len).astype("Float64")

        return df.drop(["bond_energy"], axis=1, errors="ignore")

    def get_atomic_mean(self, atomic_feature: list[list[int, float]]) -> float:
        """Helper function te get the mean of atomic features."""
        return np.array([x[1] for x in atomic_feature]).mean()

    def get_interaction_mean(self, interaction_feature: list[list[int, int, float]]) -> float:
        """Helper function to get the mean of interaction features."""
        return np.array([x[2] for x in interaction_feature]).mean() if len(interaction_feature) != 0 else 0


def centroid_freq(freqs: np.ndarray[float], intensities: np.ndarray[float], mask: np.ndarray[bool]) -> float:
    r"""Region centroid frequency for a region: v_k,R = \frac{\\sum_{i \\in R} v_k,i I_k,i}{\\sum_{i \\in R} v_k,i}."""
    return np.dot(freqs[mask], intensities[mask]) / intensities[mask].sum()


def norm_intensity(intensities: np.ndarray[float], mask: np.ndarray[bool]) -> float:
    r"""Normalized intensity for a region: I_k,R = \frac{\\sum_{i \\in R} I_k,i}{\\sum_{i \\in all} I_k,i}."""
    return intensities[mask].sum() / intensities.sum()
