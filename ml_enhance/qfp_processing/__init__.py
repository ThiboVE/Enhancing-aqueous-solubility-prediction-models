"""Quantum Fingerprint processing package.

This package provides tools for processing, aggregating, and engineering features
from quantum fingerprint data, including dataset building and RDKit feature calculation.
"""

from ml_enhance.qfp_processing.aggregation import ConformerAggregator
from ml_enhance.qfp_processing.dataset_builder import QuantumFPDatasetBuilder
from ml_enhance.qfp_processing.feature_engineering import (
    QFPFeatureEngineer,
    centroid_freq,
    norm_intensity,
)
from ml_enhance.qfp_processing.file_loader import QuantumFPFileLoader
from ml_enhance.qfp_processing.rdkit_feature_calculator import RDKitFeatureCalculator

__all__ = [
    "ConformerAggregator",
    "QFPFeatureEngineer",
    "QuantumFPDatasetBuilder",
    "QuantumFPFileLoader",
    "RDKitFeatureCalculator",
    "centroid_freq",
    "norm_intensity",
]
