from pathlib import Path
from typing import Generator

import pandas as pd

from library.qfp_processing.aggregation import ConformerAggregator
from library.qfp_processing.feature_engineering import QFPFeatureEngineer
from library.qfp_processing.file_loader import QuantumFPFileLoader

# MARK: TODO: add a function to save the final dataset to disk (add intermediate save point every 1000 molecules or something)


class QuantumFPDatasetBuilder:
    """
    Coordinates streaming, feature processing,
    conformer aggregation, and dataset assembly.
    """

    def __init__(
        self, output_path: Path, temperature: float = 300.0, cap: int | None = None
    ) -> None:
        self.loader = QuantumFPFileLoader(output_path)
        self.engineer = QFPFeatureEngineer(temperature)
        self.aggregator = ConformerAggregator(temperature)

        self.cap = cap

    def build_dataset(self) -> pd.DataFrame:
        """
        Stream-process all molecules and assemble final dataset.
        """

        molecule_rows: list[pd.Series] = []

        output_files = self.loader.list_output_files()
        output_files = (
            output_files[: self.cap] if self.cap is not None else output_files
        )

        for file in output_files:
            try:
                for df in self.loader.stream_conformer_dataframe(file):
                    df = self.engineer.clean_features(df)

                    mol_features: pd.Series = self.aggregator.thermal_average(df)

                    molecule_rows.append(mol_features)

            except Exception as e:
                print(f"Error '{e}' occured for {file}")

        return pd.DataFrame(molecule_rows)

    def stream_dataset(self, clean=True) -> Generator[pd.Series, None, None]:
        """
        Fully streaming version.
        Yields one molecule-level row at a time.
        """

        for file in self.loader.list_output_files():
            for df in self.loader.stream_conformer_dataframe(file):
                if clean:
                    df = self.engineer.clean_features(df)

                yield self.aggregator.thermal_average(df)
