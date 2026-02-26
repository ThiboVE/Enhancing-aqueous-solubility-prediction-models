"""RDKit molecular descriptor calculator for solubility prediction.

This module provides functionality to compute RDKit molecular descriptors
from SMILES strings and integrate them with the QFP pipeline output.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


class RDKitFeatureCalculator:
    """Compute molecular descriptors from SMILES using RDKit.

    Designed to integrate with the QFP pipeline output.
    """

    def __init__(self, smiles_column: str = "smiles") -> None:
        self.smiles_column = smiles_column

        self.descriptor_names = [name for name, _ in Descriptors._descList]  # noqa: SLF001
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)

    def compute_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RDKit descriptors for all molecules in the DataFrame.

        Returns a new DataFrame with the computed features.
        """
        smiles_list = df[self.smiles_column]

        feature_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                print(f"SMILES: {smiles} is invalid, features are assigned 'None'.")  # noqa: T201
                feature_list.append([None] * 217)
            else:
                feature_list.append(self.calculator.CalcDescriptors(mol))

        return pd.DataFrame(feature_list, columns=self.descriptor_names)

    def add_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RDKit descriptors and merge them into the original DataFrame."""
        rdkit_features = self.compute_descriptors(df)
        return pd.concat([df.reset_index(drop=True), rdkit_features.reset_index(drop=True)], axis=1)
