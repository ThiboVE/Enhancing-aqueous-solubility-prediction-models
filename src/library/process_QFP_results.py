from contextlib import contextmanager
from typing import Generator
from pathlib import Path
import pandas as pd
import json
import gzip
import os
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

class ProcessData:
    """
    The objective of this class is to take the output data from the QuantumFP program and transform it into a dataset usable for ML model training.

    Steps to be implemented:
    - extract data from the individual files and put it in individual dataframes
    - process the QM features
    - add the RDKit features
    - combine all conformers into one data series (thermal averaging, most stable conformer or something else)
    - combine all molecules into one dataset
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_files = self.get_output_files()

    def get_output_files(self, extension: str = '.gz'):
        """
        Generate a list of all files present in the output_path passed to the class. 
        """

        return [self.output_path / Path(path) for path in os.listdir(self.output_path) if path.endswith(extension)]
    
    def create_df(self, property_dict: dict[int, str] = PROPERTY_DICT) -> pd.DataFrame:
        pattern = re.compile(r"prop_id")

        molecule_dict = {}
        for idx, conformer in enumerate(molecule_data):
            # Get all QM properties of a conformer and assign the proper name of the property
            conformer_dict = {property_dict[int(k.split("_")[-1])]: v for k, v in conformer.items() if pattern.search(k)}

            # Add the SMILES representations to the dataframe
            conformer_dict.update({"original_smiles": conformer["original_smiles"], "output_smiles": conformer["output_smiles"]})

            # Add the dict of the conformer to the dict of the molecule
            molecule_dict[f"conformer {idx}"] = conformer_dict
        
        return pd.DataFrame(molecule_dict).T.convert_dtypes()
    
    @contextmanager
    def load_json_as_dataframe(self, file: Path) -> Generator[pd.DataFrame, None, None]:
        """
        Context manager to iterate over the individual data files.
        """
        with gzip.open(file, "rt") as f:
            data: list[dict] = json.load(f)

        df: pd.DataFrame = self.create_df(data)

        yield df
    




def create_molecule_df(molecule_data: list[dict], property_dict: dict[int, str] = PROPERTY_DICT) -> pd.DataFrame:
    """
    Transform the data from QuantumFP into a pandas dataframe. The dataframe consists of all QM properties extracted from QFP for each conformer of the molecule.

    params: 
        molecule_data: A list containing a dict for each conformer of the molecule with its corresponding QM properties.

    return:
        a pd.DataFrame containing the QM properties of each conformer of the molecule.
    """
    pattern = re.compile(r"prop_id")

    molecule_dict = {}
    for idx, conformer in enumerate(molecule_data):
        # Get all QM properties of a conformer and assign the proper name of the property
        conformer_dict = {property_dict[int(k.split("_")[-1])]: v for k, v in conformer.items() if pattern.search(k)}

        # Add the SMILES representations to the dataframe
        conformer_dict.update({"original_smiles": conformer["original_smiles"], "output_smiles": conformer["output_smiles"]})

        # Add the dict of the conformer to the dict of the molecule
        molecule_dict[f"conformer {idx}"] = conformer_dict
    
    return pd.DataFrame(molecule_dict).T.convert_dtypes()


    