import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from chemprop import data, nn
from chemprop.models import MPNN
from chemprop.models.utils import save_model
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import KFold, train_test_split
from utils import Files


@dataclass
class FixedParams:
    num_workers: int
    use_gpu: bool = torch.cuda.is_available()
    batch_size: int = 100
    dropout: float = 0.1
    n_epochs: int = 40
    metrics: list = field(default_factory=lambda: [nn.metrics.MSE(), nn.metrics.R2Score()])
    opt_n_trials: int = 30
    opt_direction: str = "minimize"


class MPNNFactory:
    def __init__(self, fxdprms: FixedParams) -> None:
        self.fixed_params = asdict(fxdprms)

    def build(self, params: dict[str, Any], train_dset: data.MoleculeDataset) -> MPNN:
        p = {**params, **self.fixed_params}

        sample = train_dset[0]
        d_v = sample.mg.V.shape[1]
        d_e = sample.mg.E.shape[1]

        mp = BondMessagePassing(
            d_h=p["message_hidden_dim"],
            depth=p["depth"],
            d_v=d_v,
            d_e=d_e,
        )

        agg = MeanAggregation()

        d_xd = train_dset[0].x_d.shape[0] if train_dset[0].x_d is not None else 0

        ffn = RegressionFFN(
            input_dim=p["message_hidden_dim"] + d_xd,
            hidden_dim=p["ffn_hidden_dim"],
            n_layers=p["ffn_n_layers"],
            dropout=p.get("dropout", 0.1),
            # output_transform=output_transform,
        )

        return MPNN(mp, agg, ffn, batch_norm=True, metrics=p["metrics"])


def make_dataloaders(
    train_dset: data.MoleculeDataset,
    val_dset: data.MoleculeDataset,
    fxdprms: FixedParams,
):
    train_loader = data.build_dataloader(
        train_dset,
        batch_size=fxdprms.batch_size,
        shuffle=True,
        num_workers=fxdprms.num_workers,
        persistent_workers=True,
        pin_memory=fxdprms.use_gpu,
    )
    val_loader = data.build_dataloader(
        val_dset,
        batch_size=fxdprms.batch_size,
        shuffle=False,
        num_workers=fxdprms.num_workers,
        persistent_workers=True,
        pin_memory=fxdprms.use_gpu,
    )

    return train_loader, val_loader


# ----------------------------
# TRAINER
# ----------------------------


def make_trainer(
    logger: CSVLogger | bool | None, fxdprms: FixedParams, trial: optuna.Trial | None = None
) -> pl.Trainer:
    # callbacks = []

    if logger is None:
        logger = False

    # if trial is not None:
    #     callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

    return pl.Trainer(
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="gpu" if fxdprms.use_gpu else "cpu",
        devices=1,
        max_epochs=fxdprms.n_epochs,
        deterministic=True,
        # callbacks=callbacks,
        # precision="16-mixed" if use_GPU else 32,
    )


def inner_cv_objective(
    trial: optuna.Trial,
    inner_fold_data: list[dict[str, data.MoleculeDataset]],
    logfile: Path,
    logger: Logger,
    fxdprms: FixedParams,
) -> float:
    losses: list[float] = []

    params: dict[str, int] = {
        "message_hidden_dim": trial.suggest_int("message_hidden_dim", 100, 500),
        "depth": trial.suggest_int("depth", 1, 5),
        "ffn_hidden_dim": trial.suggest_int("ffn_hidden_dim", 100, 500),
        "ffn_n_layers": trial.suggest_int("ffn_n_layers", 1, 5),
    }

    n_inner_folds = len(inner_fold_data)
    for fold_id, fold_dict in enumerate(inner_fold_data, 1):
        logger.info(f"Inner Fold {fold_id}/{n_inner_folds} for trial {trial.number}...")

        pl_logger = CSVLogger(logfile) if fold_id == 1 else False

        train_dset: data.MoleculeDataset = fold_dict["train"]
        val_dset: data.MoleculeDataset = fold_dict["val"]

        train_loader, val_loader = make_dataloaders(train_dset, val_dset, fxdprms)

        trainer = make_trainer(pl_logger, fxdprms, trial)
        model = MPNNFactory(fxdprms).build(params, train_dset)

        logger.info("Start training")

        trainer.fit(model, train_loader, val_loader)

        logger.info("Start validation")

        val_metrics = trainer.validate(model, val_loader, verbose=False)

        val_loss: float = val_metrics[0]["val_loss"]
        losses.append(val_loss)

        logger.info(f"Validation Loss: [{val_loss:.2f}]")

        # Per fold pruning
        trial.report(np.mean(losses), step=fold_id)

        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at fold {fold_id}")
            raise optuna.TrialPruned

        # Clear GPU mem post-trial
        if fxdprms.use_gpu and fold_id % 2 == 0:
            torch.cuda.empty_cache()

    return float(np.mean(losses))


def run_tuning_per_fold(
    inner_fold_data: list[dict[str, data.MoleculeDataset]], logger: Logger, logfile: Path, fxdprms: FixedParams
) -> dict[str, Any]:
    """Single-phase tuning on outer_train; final train/eval."""
    logger.info("Starting Single-Phase Tuning...")

    def obj(trial: optuna.Trial):
        return inner_cv_objective(trial, inner_fold_data, logfile, logger, fxdprms)

    study = optuna.create_study(
        direction=fxdprms.opt_direction,
        sampler=optuna.samplers.TPESampler(seed=112),
        pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource=len(inner_fold_data)),
    )

    study.optimize(obj, n_trials=fxdprms.opt_n_trials, n_jobs=1)

    best_params = study.best_params
    logger.info(f"Best Params: {best_params}")
    logger.info(f"Best Inner CV Loss: {study.best_value:.4f}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


def save_fold_result(
    results_file: Path,
    r2_test: float,
    mse_test: float,
    best_params: dict[str, int],
    logger: Logger,
) -> None:
    """Save the results of an outer fold"""
    fold_data: dict[str, int | float] = {
        "r2_test": r2_test,
        "mse_test": mse_test,
        **best_params,
    }

    with open(results_file, "w") as f:
        json.dump(fold_data, f)

    logger.info(f"Saved results to '{results_file}'")


def test_model(
    train_dset: data.MoleculeDataset,
    test_dset: data.MoleculeDataset,
    best_params: dict[str, Any],
    logger: Logger,
    fxdprms: FixedParams,
) -> tuple[MPNN, float, float]:
    # Final train on full outer_train
    logger.info("  Final Training on Outer Train...")

    train_loader, test_loader = make_dataloaders(train_dset, test_dset, fxdprms)

    trainer = make_trainer(logger=False, fxdprms=fxdprms)
    mpnn = MPNNFactory(fxdprms).build(best_params, train_dset)

    trainer.fit(mpnn, train_loader)

    # Outer test
    logger.info("  Evaluating on Outer Test...")
    trainer.test(mpnn, test_loader, weights_only=False)

    test_metrics = trainer.test(mpnn, test_loader)[0]
    r2_score = test_metrics["test/r2"]
    mse_score = test_metrics["test/mse"]

    logger.info(f"Test accuracy: {r2_score:.4f}")

    return mpnn, r2_score, mse_score


def get_all_features() -> dict[str, pd.DataFrame | None]:
    storage_path = Path("/data/gent/489/vsc48953/ML_enhance")

    atom_features = pd.read_csv(storage_path / "atom_features.csv")
    atom_features = atom_features if atom_features.size > 0 else None

    bond_features = pd.read_csv(storage_path / "bond_features.csv")
    bond_features = bond_features if bond_features.size > 0 else None

    mol_features = pd.read_csv(storage_path / "mol_features.csv")
    mol_features = mol_features if mol_features.size > 0 else None

    return {
        "atoms": atom_features,
        "bonds": bond_features,
        "mols": mol_features,
    }


def create_inner_folds_dsets(
    inner_folds: list[dict[str, Sequence[int]]],
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    cfg: Config,
) -> list[dict[str, data.MoleculeDataset]]:
    inner_fold_data: list[dict[str, data.MoleculeDataset]] = []
    for inner_fold in inner_folds:
        inner_train_dset, inner_val_dset, _ = build_datasets(
            inner_fold["train_ids"], inner_fold["val_ids"], target_df, all_features, config=cfg
        )

        inner_fold_data.append(
            {
                "train": inner_train_dset,
                "val": inner_val_dset,
            }
        )

    return inner_fold_data


def subsample_train_ids(train_ids: Sequence[int], size: float, target_df: pd.DataFrame) -> Sequence[int]:
    y_train = target_df.loc[target_df["id"].isin(train_ids), "solubility"]

    if size < 1.0:
        y_train_binned = pd.qcut(y_train, q=10)

        train_ids_sampled, _ = train_test_split(train_ids, train_size=size, stratify=y_train_binned, random_state=100)
        return train_ids_sampled

    return train_ids


def split_sampled_fold(train_ids: Sequence[int]) -> list[dict[str, Sequence[int]]]:
    inner_cv = KFold(5, shuffle=True, random_state=42)

    inner_folds: list[dict[str, Sequence[int]]] = []
    for train_idxs, val_idxs in inner_cv.split(train_ids):
        inner_train_ids_sampled = train_ids[train_idxs]
        inner_val_ids_sampled = train_ids[val_idxs]

        inner_folds.append({"train_ids": inner_train_ids_sampled, "val_ids": inner_val_ids_sampled})

    return inner_folds


def calculation(
    outer_train_ids: Sequence[int],
    outer_test_ids: Sequence[int],
    inner_fold_data: list[dict[str, data.MoleculeDataset]],
    target_df: pd.DataFrame,
    all_features: dict[str, pd.DataFrame | None],
    FILES: Files,
    logger: Logger,
    fxdprms: FixedParams,
    cfg: Config,
) -> None:
    logger.info("Start tuning")
    study_results = run_tuning_per_fold(
        inner_fold_data=inner_fold_data, logger=logger, logfile=FILES.LIGHTNING_LOG_DIR, fxdprms=fxdprms
    )
    best_params = study_results["best_params"]

    # Already save the best parameters of the outer fold
    save_fold_result(FILES.RESULTS_FILE_JSON, 0, 0, best_params, logger)

    outer_train_dset, outer_test_dset, _ = build_datasets(
        outer_train_ids, outer_test_ids, target_df, all_features, config=cfg
    )

    logger.info("Start testing")
    mpnn, r2_test, mse_test = test_model(outer_train_dset, outer_test_dset, best_params, logger, fxdprms)

    save_fold_result(FILES.RESULTS_FILE_JSON, r2_test, mse_test, best_params, logger)

    save_model(FILES.RESULTS_FILE_MODEL, mpnn)
