import numpy as np
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
        df = self._aggregate_interaction_features(df)

        return df

    def _remove_tensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_to_remove = [
            "molecular_dipole",
            "molecular_quadrupole",
            "molecular_polarizability",
            "atomic_dipole",
            "atomic_quadrupole",
            "atomic_polarizability",
        ]

        df = df.drop(features_to_remove, axis=1, errors="ignore")

        return df

    def _select_thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        thermodynamic_features = ["gibbs_free_energy", "entropy", "heat_capacity"]

        def get_val_at_T(conformer_list, temperature=self.temperature):
            for T, val in conformer_list:
                if T == temperature:
                    return val
            return None

        for feature in thermodynamic_features:
            df[f"{feature}_{int(self.temperature)}K"] = (
                df[feature].apply(get_val_at_T).astype("Float64")
            )

        df = df.drop(thermodynamic_features, axis=1, errors="ignore")

        return df

    def _aggregate_ir_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate IR frequencies and intensities into defined regions:
        - <1500
        - 1500-2750
        - 2750-4000
        """

        new_features_list = []

        bins = [0, 1500, 2750, 4000]
        freq_labels = ["1500", "1500_2750", "2750_4000"]

        for freqs, intensities in zip(
            df["normal_mode_frequencies"], df["infrared_intensity"]
        ):
            freqs = np.array(freqs)
            intensities = np.array(intensities)

            # Keep only physical frequencies (MARK: TODO: maybe take imaginary freqs into account one way or another)
            mask = (freqs >= 0) & (freqs <= 4000)
            freqs = freqs[mask]
            intensities = intensities[mask]

            # Assign each frequency to a bin
            bin_indices = np.digitize(freqs, bins)

            feature_dict = {}

            for i, label in enumerate(freq_labels, start=1):
                idxs = np.where(bin_indices == i)[0]

                feature_dict[f"ir_centroid_freq_{label}"] = (
                    centroid_freq(freqs, intensities, idxs) if len(idxs) > 0 else 0
                )
                feature_dict[f"ir_norm_intensity_{label}"] = (
                    norm_intensity(intensities, idxs) if len(idxs) > 0 else 0
                )

            new_features_list.append(feature_dict)

        new_features = pd.DataFrame(new_features_list).astype("Float64")

        # Concatenate with original dataframe
        df = pd.concat([df.reset_index(drop=True), new_features], axis=1)

        # Drop raw IR columns
        df = df.drop(
            ["infrared_intensity", "normal_mode_frequencies", "normal_modes"],
            axis=1,
            errors="ignore",
        )

        return df

    def _aggregate_atomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Naive approach of just taking the average, later we can look into deviding per atom type
        """
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

        def get_mean(atomic_feature):
            return np.array([x[1] for x in atomic_feature]).mean()

        for feature in atomic_features:
            df[f"avg_{feature}"] = df[feature].apply(get_mean).astype("Float64")

        df = df.drop(
            atomic_features,
            axis=1,
            errors="ignore",
        )

        return df

    def _aggregate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:

        interaction_features = {
            "bond_energy",
            "bond_length",
            "bond_stiffness",
            "overlap_integral",
            "nuclear_repulsion",
            "atomic_charge_dipole_interaction",
            "atomic_charge_quadrupole_interaction",
            "atomic_dipole_dipole_interaction",
        }

        def get_mean(interaction_feature):
            return np.array([x[2] for x in interaction_feature]).mean()

        for feature in interaction_features:
            df[f"avg_{feature}"] = df[feature].apply(get_mean).astype("Float64")

        df = df.drop(
            interaction_features,
            axis=1,
            errors="ignore",
        )

        return df


def centroid_freq(freqs, intensities, mask):
    """
    region centroid frequency for a region: v_k,R = \frac{\sum_{i \in R} v_k,i I_k,i}{\sum_{i \in R} v_k,i}
    """
    return np.dot(freqs[mask], intensities[mask]) / intensities[mask].sum()


def norm_intensity(intensities, mask):
    """
    normalized intensity for a region: I_k,R = \frac{\sum_{i \in R} I_k,i}{\sum_{i \in all} I_k,i}
    """
    return intensities[mask].sum() / intensities.sum()
