"""RDKit molecular descriptor calculator for solubility prediction.

This module provides functionality to compute RDKit molecular descriptors
from SMILES strings and integrate them with the QFP pipeline output.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from ml_enhance import parallelize


class RDKitFeatureCalculator:
    """Compute molecular descriptors from SMILES using RDKit.

    Designed to integrate with the QFP pipeline output.
    """

    def __init__(self, smiles_column: str = "smiles", descriptor_names: list[str] | None = None) -> None:
        self.smiles_column = smiles_column

        if descriptor_names is None:
            descriptor_names = [name for name, _ in Descriptors._descList]  # noqa: SLF001

        self.descriptor_names = descriptor_names
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)

    def _compute_descriptor_per_mol(self, smiles: str) -> tuple:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"SMILES: {smiles} is invalid, features are assigned 'None'.")  # noqa: T201
            return [None] * 217

        return self.calculator.CalcDescriptors(mol)

    def compute_descriptors(
        self, df: pd.DataFrame, *, multiprocess: bool = False, n_jobs: int = 4, backend: str = "loky"
    ) -> pd.DataFrame:
        """Compute RDKit descriptors for all molecules in the DataFrame.

        Returns a new DataFrame with the computed features.
        """
        smiles_list = df[self.smiles_column]

        feature_list = (
            parallelize(self._compute_descriptor_per_mol, smiles_list, n_jobs=n_jobs, backend=backend)
            if multiprocess
            else [self._compute_descriptor_per_mol(smiles) for smiles in smiles_list]
        )

        return pd.DataFrame(feature_list, columns=self.descriptor_names)

    def add_to_dataframe(
        self, df: pd.DataFrame, *, multiprocess: bool = False, n_jobs: int = 4, backend: str = "loky"
    ) -> pd.DataFrame:
        """Compute RDKit descriptors and merge them into the original DataFrame."""
        rdkit_features = self.compute_descriptors(df, multiprocess=multiprocess, n_jobs=n_jobs, backend=backend)
        return pd.concat([df.reset_index(drop=True), rdkit_features.reset_index(drop=True)], axis=1)
