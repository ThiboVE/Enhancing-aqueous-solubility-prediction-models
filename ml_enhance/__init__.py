"""Library for training ML models using topological and QFP descriptors."""

from ml_enhance.correlation_filter import CorrelationFilter
from ml_enhance.general_functions import canonicalize_smiles, get_topology_features, parallelize
from ml_enhance.JSONparser import NumpyJSONCache
from ml_enhance.model_analysis import StatisticalComparison, compare, plot_FI, plot_scaled_linreg_result
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
    "NumpyJSONCache",
    "QFPFeatureEngineer",
    "QuantumFPDatasetBuilder",
    "QuantumFPFileLoader",
    "RDKitFeatureCalculator",
    "StatisticalComparison",
    "canonicalize_smiles",
    "compare",
    "get_preprocessed_smiles",
    "get_topology_features",
    "is_atom",
    "is_salt",
    "parallelize",
    "plot_FI",
    "plot_scaled_linreg_result",
    "set_atom_map_numbers",
]
