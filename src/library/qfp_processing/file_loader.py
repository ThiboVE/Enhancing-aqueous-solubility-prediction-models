from typing import Generator
from pathlib import Path
import pandas as pd
import gzip
import json
import re

PROPERTY_DICT = {
        0: 'energy',
        1: 'atomization_energy',
        2: 'homo_lumo_gap',
        3: 'ionization_energy',
        4: 'electron_affinity',
        5: 'chemical_potential',
        6: 'molecular_dipole',
        7: 'molecular_dipole_norm',
        8: 'molecular_quadrupole',
        9: 'molecular_quadrupole_principal_invariant_2',
        10: 'molecular_quadrupole_principal_invariant_3',
        11: 'molecular_polarizability',
        12: 'molecular_polarizability_mean',
        13: 'molecular_polarizability_anisotropy',
        14: 'normal_modes',
        15: 'normal_mode_frequencies',
        16: 'infrared_intensity',
        17: 'enthalpy',
        18: 'gibbs_free_energy',
        19: 'heat_capacity',
        20: 'entropy',
        21: 'zero_point_energy',
        22: 'radius_of_gyration',
        23: 'molecular_volume',
        24: 'molecular_sasa',
        25: 'atomic_sasa',
        26: 'effective_coordination_number',
        27: 'partial_charge',
        28: 'atomic_fukui_minus',
        29: 'atomic_fukui_plus',
        30: 'atomic_dipole',
        31: 'atomic_dipole_norm',
        32: 'atomic_quadrupole',
        33: 'atomic_quadrupole_principal_invariant_2',
        34: 'atomic_quadrupole_principal_invariant_3',
        35: 'atomic_polarizability',
        36: 'atomic_polarizability_mean',
        37: 'atomic_polarizability_anisotropy',
        38: 'nuclear_repulsion',
        39: 'bond_energy',
        40: 'bond_length',
        41: 'bond_stiffness',
        42: 'overlap_integral',
        43: 'atomic_dipole_dipole_interaction',
        44: 'atomic_charge_dipole_interaction',
        45: 'atomic_charge_quadrupole_interaction',
        46: 'sterimol_L',
        47: 'sterimol_Bmin',
        48: 'sterimol_Bmax',
        49: 'percentage_buried_volume',
        50: 'solvation_energy_water',
        51: 'solvation_energy_thf',
        52: 'solvation_energy_cyclohexane',
        53: 'solvation_energy_dmso',
        54: 'partial_charge_water',
        55: 'partial_charge_thf',
        56: 'partial_charge_cyclohexane',
        57: 'partial_charge_dmso'
    }

class QuantumFPFileLoader:
    """
    Responsible for discovering and loading QuantumFP output files.
    """

    _PROP_PATTERN = re.compile(r"prop_id")

    def __init__(
        self,
        output_path: Path,
        property_dict: dict[int, str] = PROPERTY_DICT,
    ) -> None:
        self.output_path = Path(output_path)
        self.property_dict = property_dict

    def list_output_files(self, extension: str = ".gz") -> list[Path]:
        """
        Return all output files with the given extension.
        """
        return [
            path for path in self.output_path.iterdir()
            if path.suffix == extension
        ]

    def stream_conformer_dataframe(
        self,
        file: Path,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load a single file, convert it to a conformer-level DataFrame,
        yield it, and release memory afterward.
        """
        with gzip.open(file, "rt") as f:
            data: list[dict] = json.load(f)

        df = self._build_conformer_dataframe(data)

        yield df


    def _build_conformer_dataframe(
        self,
        data: list[dict],
    ) -> pd.DataFrame:
        """
        Convert raw QuantumFP conformer JSON data
        into a conformer-level DataFrame.
        """

        rows: list[dict] = []

        for conformer in data:

            qm_features: dict[str, float] = {}

            for key, value in conformer.items():
                if self._PROP_PATTERN.search(key):

                    prop_id = int(key.rsplit("_", 1)[-1])

                    feature_name = self.property_dict.get(prop_id)

                    if feature_name is not None:
                        qm_features[feature_name] = value

            # Attach metadata
            qm_features["original_smiles"] = conformer["original_smiles"]
            qm_features["output_smiles"] = conformer["output_smiles"]

            rows.append(qm_features)

        return pd.DataFrame(rows).convert_dtypes()