import io
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from chemprop import models
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from lightning import pytorch as pl

from ml_enhance.nn import (
    Config,
    CustomMultiHotAtomFeaturizer,
    CustomMultiHotBondFeaturizer,
    CustomSimpleMoleculeMolGraphFeaturizer,
    ShapMaskConfig,
    SHAPModelWrapper,
    atom_features,
    bond_features,
    build_datasets,
    mol_features,
    shap_feature_transform,
)


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
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        extra_atom_fdim=extra_atom_fdim,
        extra_bond_fdim=extra_bond_fdim,
    )

    dataset = MoleculeDataset([masked_datapoint], featurizer=featurizer)
    loader = build_dataloader(dataset, shuffle=False, batch_size=1)

    preds = trainer.predict(mpnn, loader)
    # preds is a list of tensors, one per batch
    return preds[0][0].item()


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

    explainer = shap.PermutationExplainer(model_wrapper, masker=binary_masker)
    explanation = explainer(feature_choice, max_evals=max_evals)

    return explanation.values[0]


def run_shap_analysis(
    test_dataset: MoleculeDataset,
    mpnn: models.MPNN,
    extra_atom_feature_names: list[str],
    extra_bond_feature_names: list[str],
    mol_feature_names: list[str],
    n_rbf: int = 10,
    max_evals: int = 25,
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
        "shap_values": shap_values,
        "mean_shap": mean_shap,
        "feature_names": mask_config.feature_names,
    }


# ── example usage ─────────────────────────────────────────────────────────────


def run_fold(
    fold_id: int, data: dict[str, Any], model_folds_data: dict[str, Any], p_build_datasets: Callable, cfg: Config
):
    split_data = data[f"outer_fold_{fold_id}"]

    outer_train_ids = split_data["train_ids"]
    outer_test_ids = split_data["test_ids"]

    model_data = model_folds_data[f"fold_{fold_id}"]

    _, outer_test_dset, _ = p_build_datasets(outer_train_ids, outer_test_ids)

    # load your trained model checkpoint
    mpnn = load_model_from_fold(model_data)

    results = run_shap_analysis(
        test_dataset=outer_test_dset,
        mpnn=mpnn,
        extra_atom_feature_names=atom_features if cfg.use_atom_features else [],
        extra_bond_feature_names=bond_features if cfg.use_bond_features else [],
        mol_feature_names=mol_features if cfg.use_mol_features else [],
    )

    print("Mean absolute SHAP values per feature group:")
    for name, val in zip(results["feature_names"], results["mean_shap"], strict=True):
        print(f"  {name}: {val:.4f}")


def main() -> None:
    cfg = Config(
        use_atom_features=False,
        use_bond_features=False,
        use_mol_features=False,
        use_custom_atom_featurizer=True,
        use_custom_bond_featurizer=False,
    )

    fold_id = int(sys.argv[1])

    target_df = pd.read_csv("needed_data/target_df.csv")

    all_features: dict[str, pd.DataFrame | None] = get_all_features()

    data = torch.load("needed_data/chemprop_splits.pt", weights_only=False)

    model_folds_data = torch.load("../data/chemprop_results/3_chemprop_no_added_rerun_results.pt", weights_only=False)

    p_build_datasets = partial(build_datasets, target_df=target_df, all_features=all_features, config=cfg)

    run_fold(fold_id, data, model_folds_data, p_build_datasets, cfg)

    # results = parallelize(run_fold, fold_ids, data=data, p_build_datasets=p_build_datasets)

    # save results
    # np.save("shap_values.npy", results["shap_values"])


if __name__ == "__main__":
    main()
