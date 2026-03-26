from collections.abc import Generator
from pathlib import Path

import pandas as pd

from ml_enhance.general_functions import parallelize
from ml_enhance.qfp_processing.aggregation import ConformerAggregator
from ml_enhance.qfp_processing.feature_engineering import QFPFeatureEngineer
from ml_enhance.qfp_processing.file_loader import QuantumFPFileLoader


class QuantumFPDatasetBuilder:
    """Coordinates streaming, feature processing, and dataset assembly.

    Handles conformer aggregation and dataset assembly.
    """

    def __init__(self, data_directory: Path, temperature: float = 300.0) -> None:
        self.loader = QuantumFPFileLoader(data_directory)
        self.engineer = QFPFeatureEngineer(temperature)
        self.aggregator = ConformerAggregator(temperature)

    def build_single_molecule(self, file: Path) -> pd.Series | None:
        try:
            for df in self.loader.stream_conformer_dataframe(file):
                cleaned_df = self.engineer.clean_features(df)

                mol_features: pd.Series = self.aggregator.thermal_average(cleaned_df)

        except Exception as e:
            print(f"Error '{e}' occured for {file}")
            return None
        else:
            return mol_features

    def build_batch(self, files_batch: list[Path]) -> list[pd.Series | None]:
        """Process a batch of molecules and return a list of feature Series."""
        return [self.build_single_molecule(file) for file in files_batch]

    def build_dataset(
        self,
        *,
        files_list: list[Path] | None = None,
        multiprocess: bool = True,
        n_jobs: int = 4,
    ) -> tuple[pd.DataFrame, list[Path]]:
        """Stream-process all molecules and assemble final dataset."""
        if files_list is None:
            files_list = self.loader.list_output_files()

        results: list[pd.Series | None] = (
            parallelize(self.build_single_molecule, files_list, n_jobs=n_jobs)
            if multiprocess
            else [self.build_single_molecule(file) for file in files_list]
        )

        molecule_rows: list[pd.Series] = []
        error_files: list[Path] = []
        for idx, result in enumerate(results):
            if result is None:
                error_files.append(files_list[idx])
            else:
                molecule_rows.append(result)

        return pd.DataFrame(molecule_rows), error_files

    def build_dataset_batched(
        self, *, files_list: list[Path] | None = None, n_jobs: int = 4, batch_size: int | None = None
    ) -> tuple[pd.DataFrame, list[Path]]:
        if files_list is None:
            files_list = self.loader.list_output_files()
        n_files = len(files_list)

        if batch_size is None:
            batch_size = max(1, n_files // (n_jobs * 5))

        batches: list[list[Path]] = [files_list[i : i + batch_size] for i in range(0, n_files, batch_size)]

        batch_results: list[list[pd.Series | None]] = parallelize(self.build_batch, batches, n_jobs=n_jobs)

        molecule_rows: list[pd.Series] = []
        error_files: list[Path] = []
        for idx, results in enumerate(batch_results):
            for jdx, result in enumerate(results):
                if result is None:
                    error_files.append(files_list[idx * batch_size + jdx])
                else:
                    molecule_rows.append(result)

        return pd.DataFrame(molecule_rows), error_files

    def _stream_dataset(self, *, clean: bool = True) -> Generator[pd.Series]:
        """Fully streaming version.

        Yields one molecule-level row at a time.
        """
        for file in self.loader.list_output_files():
            for df in self.loader.stream_conformer_dataframe(file):
                if clean:
                    df = self.engineer.clean_features(df)

                yield self.aggregator.thermal_average(df)
