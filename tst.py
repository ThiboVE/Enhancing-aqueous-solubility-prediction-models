import io
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from chemprop import models
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from lightning import pytorch as pl
from nn_utils import (
    Config,
    atom_features,
    bond_features,
    mol_features,
)
from utils_featurizers import (
    CustomMultiHotAtomFeaturizer,
    CustomMultiHotBondFeaturizer,
    CustomSimpleMoleculeMolGraphFeaturizer,
)

# ── model wrapper for SHAP ────────────────────────────────────────────────────

type PredictionFn = Callable[[np.ndarray, MoleculeDatapoint, ShapMaskConfig, MPNN, pl.Trainer], float]


class ShapMaskConfig:
    """Defines the structure of the SHAP binary mask vector.

    The mask is a flat binary vector with one bit per feature group:
        [default_atom_groups... | default_bond_groups... | extra_atom_groups... | extra_bond_groups... | mol_features...]

    Parameters
    ----------
    n_default_atom_groups:
        Number of default atom feature groups (= len(_subfeats) + 2).
    n_default_bond_groups:
        Number of default bond feature groups (4 for the default featurizer).
    extra_atom_feature_names:
        Names of the extra atom feature groups (before RBF expansion).
    extra_bond_feature_names:
        Names of the extra bond feature groups (before RBF expansion).
    mol_feature_names:
        Names of the molecule-level features.
    n_rbf:
        Number of RBF basis functions per extra atom/bond feature group.
    """

    def __init__(
        self,
        n_default_atom_groups: int,
        n_default_bond_groups: int,
        extra_atom_feature_names: list[str],
        extra_bond_feature_names: list[str],
        mol_feature_names: list[str],
        n_rbf: int = 10,
    ) -> None:
        self.n_default_atom_groups = n_default_atom_groups
        self.n_default_bond_groups = n_default_bond_groups
        self.extra_atom_feature_names = extra_atom_feature_names
        self.extra_bond_feature_names = extra_bond_feature_names
        self.mol_feature_names = mol_feature_names
        self.n_rbf = n_rbf

        # index ranges within the flat mask
        self.default_atom_start = 0
        self.default_atom_end = n_default_atom_groups

        self.default_bond_start = self.default_atom_end
        self.default_bond_end = self.default_bond_start + n_default_bond_groups

        self.extra_atom_start = self.default_bond_end
        self.extra_atom_end = self.extra_atom_start + len(extra_atom_feature_names)

        self.extra_bond_start = self.extra_atom_end
        self.extra_bond_end = self.extra_bond_start + len(extra_bond_feature_names)

        self.mol_start = self.extra_bond_end
        self.mol_end = self.mol_start + len(mol_feature_names)

        self.total = self.mol_end

    @property
    def feature_names(self) -> list[str]:
        """Human-readable name for each mask bit, for SHAP plots."""
        # default_atom_names = [f"atom_default_{i}" for i in range(self.n_default_atom_groups)]
        # default_bond_names = [f"bond_default_{i}" for i in range(self.n_default_bond_groups)]
        default_atom_names = [
            "atomic_number",
            "degree",
            "formal_charge",
            "chiral_tag",
            "number_of_hydrogens",
            "hybridization",
            "aromaticity",
            "mass",
        ]
        default_bond_names = ["bond_type", "conjugated?", "in_ring?", "stereochemistry"]
        return [
            *default_atom_names,
            *default_bond_names,
            *self.extra_atom_feature_names,
            *self.extra_bond_feature_names,
            *self.mol_feature_names,
        ]

    def unpack(self, mask: np.ndarray) -> dict[str, list[bool]]:
        """Unpack a flat mask vector into its components."""
        return {
            "keep_atom_features": mask[self.default_atom_start : self.default_atom_end].astype(bool).tolist(),
            "keep_bond_features": mask[self.default_bond_start : self.default_bond_end].astype(bool).tolist(),
            "keep_extra_atom_groups": mask[self.extra_atom_start : self.extra_atom_end].astype(bool).tolist(),
            "keep_extra_bond_groups": mask[self.extra_bond_start : self.extra_bond_end].astype(bool).tolist(),
            "keep_mol_features": mask[self.mol_start : self.mol_end].astype(bool).tolist(),
        }


class SHAPModelWrapper:
    def __init__(
        self,
        datapoint: MoleculeDatapoint,
        mask_config: ShapMaskConfig,
        mpnn: MPNN,
        trainer: pl.Trainer,
        get_predictions: PredictionFn,
    ) -> None:
        self.datapoint = datapoint
        self.mask_config = mask_config
        self.mpnn = mpnn
        self.trainer = trainer
        self.get_predictions = get_predictions

    def __call__(self, mask_batch: np.ndarray) -> np.ndarray:
        # mask_batch has shape (n_mask_samples, n_features)
        return np.array(
            [
                self.get_predictions(
                    mask=mask_batch[i],
                    datapoint=self.datapoint,
                    mask_config=self.mask_config,
                    mpnn=self.mpnn,
                    trainer=self.trainer,
                )
                for i in range(mask_batch.shape[0])
            ]
        )


def shap_feature_transform(
    mask: np.ndarray,
    mask_config: ShapMaskConfig,
    V_f: np.ndarray | None,
    E_f: np.ndarray | None,
    x_d: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    unpacked = mask_config.unpack(mask)

    return (
        mask_extra_features(V_f, unpacked["keep_extra_atom_groups"]),
        mask_extra_features(E_f, unpacked["keep_extra_bond_groups"]),
        mask_mol_features(x_d, unpacked["keep_mol_features"]),
    )


def mask_extra_features(
    arr: np.ndarray | None,
    keep_features: list[bool],
    n_rbf: int = 10,
) -> np.ndarray | None:
    """Zero out RBF features in V_f or E_f based on keep_features mask.

    Parameters
    ----------
    arr:
        Feature array of shape (n_atoms_or_bonds, n_features * n_rbf).
    keep_features:
        One bool per feature. False means zero out that feature's n_rbf columns.
    n_rbf:
        Number of columns per feature.
    """
    if arr is None:
        return None

    masked = arr.copy()
    for group_idx, keep in enumerate(keep_features):
        if not keep:
            start = group_idx * n_rbf
            end = start + n_rbf
            masked[:, start:end] = 0.0
    return masked


def mask_mol_features(
    x_d: np.ndarray | None,
    keep_features: list[bool],
) -> np.ndarray | None:
    """Zero out individual molecule-level features based on keep_features mask."""
    if x_d is None:
        return None

    masked = x_d.copy()
    for i, keep in enumerate(keep_features):
        if not keep:
            masked[i] = 0
    return masked


def load_model_from_fold(ckpt: dict) -> models.MPNN:
    """Load a single fold's MPNN by writing its checkpoint to a BytesIO buffer and passing that to MPNN.load_from_file."""
    buffer = io.BytesIO()
    torch.save(ckpt, buffer)
    buffer.seek(0)
    return models.MPNN.load_from_file(buffer)


def get_all_features() -> dict[str, pd.DataFrame | None]:
    atom_features = pd.read_csv("needed_data/atom_features.csv")
    atom_features = atom_features if atom_features.size > 0 else None

    bond_features = pd.read_csv("needed_data/bond_features.csv")
    bond_features = bond_features if bond_features.size > 0 else None

    mol_features = pd.read_csv("needed_data/mol_features.csv")
    mol_features = mol_features if mol_features.size > 0 else None

    return {
        "atoms": atom_features,
        "bonds": bond_features,
        "mols": mol_features,
    }


def get_predictions(
    mask: np.ndarray,
    datapoint: MoleculeDatapoint,
    mask_config: ShapMaskConfig,
    mpnn: models.MPNN,
    trainer: pl.Trainer,
) -> float:
    """Get a model prediction for one molecule with a feature group mask applied.

    Parameters
    ----------
    mask:
        Flat binary mask vector of length mask_config.total.
    datapoint:
        The MoleculeDatapoint for the molecule (with precomputed V_f, E_f, x_d).
    mask_config:
        Defines the structure of the mask vector.
    base_atom_featurizer:
        The atom featurizer used during training (to read its config).
    base_bond_featurizer:
        The bond featurizer used during training.
    mpnn:
        The trained Chemprop MPNN model.
    trainer:
        Lightning Trainer configured for inference.
    """
    smiles = datapoint.name
    y = datapoint.y
    V_f = datapoint.V_f
    E_f = datapoint.E_f
    x_d = datapoint.x_d

    V_f_transformed, E_f_transformed, x_d_transformed = shap_feature_transform(mask, mask_config, V_f, E_f, x_d)

    masked_datapoint = MoleculeDatapoint.from_smi(
        smi=smiles,
        y=y,
        V_f=V_f_transformed,
        E_f=E_f_transformed,
        x_d=x_d_transformed,
        keep_h=True,
    )

    unpacked = mask_config.unpack(mask)

    # build custom featurizers with the default feature groups masked
    atom_featurizer = CustomMultiHotAtomFeaturizer.from_custom(
        keep_features=unpacked["keep_atom_features"],
    )
    bond_featurizer = CustomMultiHotBondFeaturizer.from_base(
        keep_features=unpacked["keep_bond_features"],
    )

    extra_atom_fdim = datapoint.V_f.shape[1] if datapoint.V_f is not None else 0
    extra_bond_fdim = datapoint.E_f.shape[1] if datapoint.E_f is not None else 0

    featurizer = CustomSimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=CustomMultiHotAtomFeaturizer.from_custom(
            keep_features=unpacked["keep_atom_features"],
        ),
        bond_featurizer=CustomMultiHotBondFeaturizer.from_base(
            keep_features=unpacked["keep_bond_features"],
        ),
        extra_atom_fdim=extra_atom_fdim,
        extra_bond_fdim=extra_bond_fdim,
    )

    dataset = MoleculeDataset([masked_datapoint], featurizer=featurizer)
    loader = build_dataloader(dataset, shuffle=False, batch_size=1, num_workers=0)

    mpnn.eval()
    device = next(mpnn.parameters()).device
    with torch.no_grad():
        batch = next(iter(loader))
        # MoleculeDataLoader yields (bmg, V_d, X_d, y, w, lt_mask, gt_mask)
        bmg, V_d, X_d, *_ = batch
        pred = mpnn(bmg.to(device), V_d, X_d).squeeze()

    return pred.item()


# ── main SHAP loop ────────────────────────────────────────────────────────────


def determine_shap(
    datapoint: MoleculeDatapoint,
    mask_config: ShapMaskConfig,
    mpnn: models.MPNN,
    trainer: pl.Trainer,
    binary_masker: Any,
    feature_choice: np.ndarray,
    max_evals: int,
) -> np.ndarray:
    model_wrapper = SHAPModelWrapper(
        datapoint=datapoint, mask_config=mask_config, mpnn=mpnn, trainer=trainer, get_predictions=get_predictions
    )

    explainer = shap.PermutationExplainer(model_wrapper, masker=binary_masker, seed=42)
    explanation = explainer(feature_choice, max_evals=max_evals)

    return explanation.values[0]


def run_shap_analysis(
    test_dataset: MoleculeDataset,
    mpnn: models.MPNN,
    extra_atom_feature_names: list[str],
    extra_bond_feature_names: list[str],
    mol_feature_names: list[str],
    n_rbf: int = 10,
    max_evals: int = 100,
) -> dict[str, Any]:
    """Run SHAP analysis over an entire test dataset.

    Parameters
    ----------
    test_dataset:
        Precomputed MoleculeDataset for the test set.
    mpnn:
        Trained Chemprop MPNN model.
    base_atom_featurizer:
        The atom featurizer used during training.
    base_bond_featurizer:
        The bond featurizer used during training.
    extra_atom_feature_names:
        Names of the extra atom feature groups (one per RBF group).
    extra_bond_feature_names:
        Names of the extra bond feature groups (one per RBF group).
    mol_feature_names:
        Names of the molecule-level features.
    n_rbf:
        Number of RBF basis functions per feature group.
    max_evals:
        Number of permutation evaluations per molecule for SHAP.

    Returns:
    -------
    dict with keys:
        "shap_values":   np.ndarray of shape (n_molecules, n_feature_groups)
        "mean_shap":     np.ndarray of shape (n_feature_groups,) — mean absolute SHAP
        "feature_names": list[str]
    """
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )

    n_default_atom_groups = 8
    n_default_bond_groups = 4  # bond_type, is_conjugated, is_in_ring, stereo

    mask_config = ShapMaskConfig(
        n_default_atom_groups=n_default_atom_groups,
        n_default_bond_groups=n_default_bond_groups,
        extra_atom_feature_names=extra_atom_feature_names,
        extra_bond_feature_names=extra_bond_feature_names,
        mol_feature_names=mol_feature_names,
        n_rbf=n_rbf,
    )

    # binary masker: each feature group is independently toggled 0 or 1
    # background is all-ones (all features present)
    background = np.zeros((1, mask_config.total))
    binary_masker = shap.maskers.Independent(background, max_samples=100)

    # the feature choice vector: all features on (1 = keep, 0 = zero out)
    feature_choice = np.ones((1, mask_config.total))

    p_determine_shap = partial(
        determine_shap,
        mask_config=mask_config,
        mpnn=mpnn,
        trainer=trainer,
        binary_masker=binary_masker,
        feature_choice=feature_choice,
        max_evals=max_evals,
    )

    shap_values = np.array(
        [p_determine_shap(datapoint) for datapoint in test_dataset.data]
    )  # shape (n_molecules, n_feature_groups)
    mean_shap = np.mean(np.abs(shap_values), axis=0)

    return {
        "shap_values": shap_values.tolist(),
        "mean_shap": mean_shap.tolist(),
        "feature_names": mask_config.feature_names,
    }


# ── example usage ─────────────────────────────────────────────────────────────


def run_fold(
    fold_id: int, data: dict[str, Any], mpnn: models.MPNN, p_build_datasets: Callable, cfg: Config
) -> dict[str, Any]:
    split_data = data[f"outer_fold_{fold_id}"]

    outer_train_ids = split_data["train_ids"]
    outer_test_ids = split_data["test_ids"]

    _, outer_test_dset, _ = p_build_datasets(outer_train_ids, outer_test_ids)

    results = run_shap_analysis(
        test_dataset=outer_test_dset,
        mpnn=mpnn,
        extra_atom_feature_names=atom_features if cfg.use_atom_features else [],
        extra_bond_feature_names=bond_features if cfg.use_bond_features else [],
        mol_feature_names=mol_features if cfg.use_mol_features else [],
        max_evals=25,
    )

    # print("Mean absolute SHAP values per feature group:")
    # for name, val in zip(results["feature_names"], results["mean_shap"], strict=True):
    #     print(f"  {name}: {val:.4f}")

    return results
