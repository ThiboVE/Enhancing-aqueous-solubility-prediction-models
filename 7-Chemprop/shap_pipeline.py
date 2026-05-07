from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from chemprop import data, models
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from lightning import pytorch as pl

from ml_enhance.nn import (
    Config,
    CustomMultiHotAtomFeaturizer,
    CustomMultiHotBondFeaturizer,
    CustomSimpleMoleculeMolGraphFeaturizer,
    ShapMaskConfig,
    atom_features,
    build_datasets,
    shap_feature_transform,
)


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

    V_f_transformed, E_f_transformed, x_d_transformed = shap_feature_transform(V_f, E_f, x_d)

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
    loader = data.build_dataloader(dataset, shuffle=False, batch_size=1)

    preds = trainer.predict(mpnn, loader)
    # preds is a list of tensors, one per batch
    return preds[0][0].item()


# ── model wrapper for SHAP ────────────────────────────────────────────────────


class MoleculeModelWrapper:
    def __init__(
        self,
        datapoint: MoleculeDatapoint,
        mask_config: ShapMaskConfig,
        mpnn: models.MPNN,
        trainer: pl.Trainer,
    ) -> None:
        self.datapoint = datapoint
        self.mask_config = mask_config
        self.mpnn = mpnn
        self.trainer = trainer

    def __call__(self, mask_batch: np.ndarray) -> np.ndarray:
        # mask_batch has shape (n_mask_samples, n_features)
        return np.array(
            [
                get_predictions(
                    mask=mask_batch[i],
                    datapoint=self.datapoint,
                    mask_config=self.mask_config,
                    mpnn=self.mpnn,
                    trainer=self.trainer,
                )
                for i in range(mask_batch.shape[0])
            ]
        )


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
    model_wrapper = MoleculeModelWrapper(
        datapoint=datapoint,
        mask_config=mask_config,
        mpnn=mpnn,
        trainer=trainer,
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
    max_evals: int = 100,
    n_jobs: int = 5,
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
    background = np.ones((1, mask_config.total))
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


def main() -> None:
    cfg = Config(
        use_atom_features=True,
        use_bond_features=False,
        use_mol_features=False,
        use_custom_atom_featurizer=True,
        use_custom_bond_featurizer=False,
    )

    target_df = pd.read_csv("../target_df.csv")

    all_features: dict[str, pd.DataFrame | None] = get_all_features()

    data = torch.load("../chemprop_splits.pt", weights_only=False)
    split_data = data[f"outer_fold_{fold_id}"]

    outer_train_ids: Iterable[int] = split_data["train_ids"]
    outer_test_ids: Iterable[int] = split_data["test_ids"]

    _, outer_test_dset, _ = build_datasets(outer_train_ids, outer_test_ids, target_df, all_features, config=cfg)

    # load your trained model checkpoint
    mpnn = models.MPNN.load_from_file("")

    results = run_shap_analysis(
        test_dataset=outer_test_dset,
        mpnn=mpnn,
        extra_atom_feature_names=atom_features,
        extra_bond_feature_names=[],
        mol_feature_names=[],
    )

    print("Mean absolute SHAP values per feature group:")
    for name, val in zip(results["feature_names"], results["mean_shap"], strict=True):
        print(f"  {name}: {val:.4f}")

    # save results
    # np.save("shap_values.npy", results["shap_values"])


if __name__ == "__main__":
    main()
