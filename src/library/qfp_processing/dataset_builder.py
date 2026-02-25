from pathlib import Path
from typing import Generator

import pandas as pd

from library.general_functions import parallelize
from library.qfp_processing.aggregation import ConformerAggregator
from library.qfp_processing.feature_engineering import QFPFeatureEngineer
from library.qfp_processing.file_loader import QuantumFPFileLoader

# MARK: TODO: add a function to save the final dataset to disk (add intermediate save point every 1000 molecules or something)


class QuantumFPDatasetBuilder:
    """
    Coordinates streaming, feature processing,
    conformer aggregation, and dataset assembly.
    """

    def __init__(self, data_directory: Path, temperature: float = 300.0) -> None:
        self.loader = QuantumFPFileLoader(data_directory)
        self.engineer = QFPFeatureEngineer(temperature)
        self.aggregator = ConformerAggregator(temperature)

    def build_single_molecule(self, file: Path) -> pd.Series | None:
        try:
            for df in self.loader.stream_conformer_dataframe(file):
                df = self.engineer.clean_features(df)

                mol_features: pd.Series = self.aggregator.thermal_average(df)

            return mol_features

        except Exception as e:
            print(f"Error '{e}' occured for {file}")
            return None

    def build_dataset(
        self, cap: int | None = None, multiprocess: bool = False, n_jobs: int = 4
    ) -> tuple[pd.DataFrame, list[Path]]:
        """
        Stream-process all molecules and assemble final dataset.
        """

        molecule_rows: list[pd.Series] = []
        error_files: list[Path] = []

        output_files = self.loader.list_output_files()
        output_files = output_files[:cap] if cap is not None else output_files

        results = (
            parallelize(self.build_single_molecule, output_files, n_jobs=n_jobs)
            if multiprocess
            else [self.build_single_molecule(file) for file in output_files]
        )

        for idx, result in enumerate(results):
            if result is None:
                error_files.append(output_files[idx])
            else:
                molecule_rows.append(result)

        return pd.DataFrame(molecule_rows), error_files

    def _stream_dataset(self, clean=True) -> Generator[pd.Series, None, None]:
        """
        Fully streaming version.
        Yields one molecule-level row at a time.
        """

        for file in self.loader.list_output_files():
            for df in self.loader.stream_conformer_dataframe(file):
                if clean:
                    df = self.engineer.clean_features(df)

                yield self.aggregator.thermal_average(df)
