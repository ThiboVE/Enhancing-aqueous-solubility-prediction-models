"""Library for training ML models using topological and QFP descriptors."""

from ml_enhance.correlation_filter import CorrelationFilter
from ml_enhance.general_functions import (
    canonicalize_smiles,
    get_topology_features,
    load_hpc_result,
    parallelize,
    parse_filename,
)
from ml_enhance.hpc_utils import (
    Files,
    LoggerWriter,
    Score,
    custom_train_test_split,
    process_fold,
    process_job,
    save_results,
    setup_logger,
)
from ml_enhance.JSONparser import NumpyJSONCache
from ml_enhance.model_analysis import (
    FeatureImportance,
    StatisticalComparison,
    compare,
    plot_FI,
    plot_scaled_linreg_result,
)
from ml_enhance.preprocess_smiles import get_preprocessed_smiles, is_atom, is_salt, set_atom_map_numbers
from ml_enhance.qfp_processing import (
    ConformerAggregator,
    QFPFeatureEngineer,
    QuantumFPDatasetBuilder,
    QuantumFPFileLoader,
    RDKitFeatureCalculator,
)

__all__ = [
    "ConformerAggregator",
    "CorrelationFilter",
    "FeatureImportance",
    "Files",
    "LoggerWriter",
    "NumpyJSONCache",
    "QFPFeatureEngineer",
    "QuantumFPDatasetBuilder",
    "QuantumFPFileLoader",
    "RDKitFeatureCalculator",
    "Score",
    "StatisticalComparison",
    "canonicalize_smiles",
    "compare",
    "custom_train_test_split",
    "get_preprocessed_smiles",
    "get_topology_features",
    "is_atom",
    "is_salt",
    "load_hpc_result",
    "parallelize",
    "parse_filename",
    "plot_FI",
    "plot_scaled_linreg_result",
    "process_fold",
    "process_job",
    "save_results",
    "set_atom_map_numbers",
    "setup_logger",
]
