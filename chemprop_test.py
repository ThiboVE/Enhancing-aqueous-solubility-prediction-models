import json
import sys
from collections.abc import Iterable
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
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset

from ml_enhance.hpc_utils import setup_logger

pl.seed_everything(152, workers=True)


@dataclass
class FixedParams:
    use_gpu: bool = torch.cuda.is_available()
    batch_size: int = 100
    num_workers: int = 1
    dropout: float = 0.1
    n_epochs: int = 2  # 30
    metrics: list = field(default_factory=lambda: [nn.metrics.MSE(), nn.metrics.R2Score()])
    opt_n_trials: int = 2  # 30
    opt_direction: str = "minimize"


class Files:
    def __init__(self, running_file: str, filename: str) -> None:
        self.running_file: Path = Path(running_file)
        self.filename: str = filename

        self.base = self.running_file.parent

        self.output_dir = Path(self.running_file.stem) / "results"
        self.log_dir = Path("logs")

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_df_file(self, df_file_name: str) -> Path:
        return self.base.parent / df_file_name

    @property
    def SPLITS_FILE(self) -> Path:
        return Path("hpc_splits.pkl")

    @property
    def RDKIT_FILE(self) -> Path:
        return self.base.parent / "rdkit_feature_names.json"

    @property
    def LOG_FILE(self) -> Path:
        return self.log_dir / f"{self.filename}.log"

    @property
    def RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_results.pkl"

    @property
    def RESULTS_FILE_JSON(self) -> Path:
        return self.output_dir / f"{self.filename}_results.json"

    @property
    def RESULTS_FILE_MODEL(self) -> Path:
        return self.output_dir / f"{self.filename}_model.json"

    @property
    def PFI_RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_PFI_results.csv"


class MPNNFactory:
    def __init__(self, cfg: FixedParams) -> None:
        self.fixed_params = asdict(cfg)

    def build(self, params: dict[str, Any], scaler: StandardScaler | None) -> MPNN:
        p = {**params, **self.fixed_params}

        mp = BondMessagePassing(
            d_h=p["message_hidden_dim"],
            depth=p["depth"],
        )

        agg = MeanAggregation()

        # output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        ffn = RegressionFFN(
            input_dim=p["message_hidden_dim"],
            hidden_dim=p["ffn_hidden_dim"],
            n_layers=p["ffn_n_layers"],
            dropout=p.get("dropout", 0.1),
            # output_transform=output_transform,
        )

        return MPNN(mp, agg, ffn, batch_norm=True, metrics=p["metrics"])


def make_dataloaders(
    train_dset: Dataset,
    val_dset: Dataset,
    cfg: FixedParams,
    scaler: StandardScaler | None = None,
):
    if scaler is not None:
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(
        train_dset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = data.build_dataloader(val_dset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, scaler


# ----------------------------
# TRAINER
# ----------------------------


def make_trainer(logger: Logger, cfg: FixedParams, trial: optuna.Trial | None = None) -> pl.Trainer:
    callbacks = []

    if trial is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

    return pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="gpu" if cfg.use_gpu else "cpu",
        devices=1,
        max_epochs=cfg.n_epochs,
        deterministic=True,
        callbacks=callbacks,
        # precision="16-mixed" if use_GPU else 32,
    )


def inner_cv_objective(
    trial: optuna.Trial, dataset: Dataset, inner_cv: KFold, model_factory: MPNNFactory, logger: Logger, cfg: FixedParams
) -> float:
    losses: list[float] = []

    params = {
        "message_hidden_dim": trial.suggest_int("message_hidden_dim", 100, 500),
        "depth": trial.suggest_int("depth", 1, 5),
        "ffn_hidden_dim": trial.suggest_int("ffn_hidden_dim", 100, 500),
        "ffn_n_layers": trial.suggest_int("ffn_n_layers", 1, 5),
    }

    n_inner_folds = inner_cv.get_n_splits()
    for fold_id, (tr_idxs, va_idxs) in enumerate(inner_cv.split(range(len(dataset))), 1):
        logger.info(f"Inner Fold {fold_id}/{n_inner_folds} for trial {trial.number}...")

        train_dset = Subset(dataset, tr_idxs)
        val_dset = Subset(dataset, va_idxs)

        train_loader, val_loader, scaler = make_dataloaders(train_dset, val_dset, cfg)

        trainer = make_trainer(logger, cfg, trial)
        model = MPNNFactory(cfg).build(params, scaler)

        logger.info("Start training")

        trainer.fit(model, train_loader, val_loader)

        logger.info("Start validation")

        val_metrics = trainer.validate(model, val_loader, verbose=False)

        val_loss: float = val_metrics[0]["val_loss"]
        losses.append(val_loss)

        logger.info(f"Validation Loss: [{val_loss:.2f}]")

        # Per fold pruning
        # trial.report(np.mean(losses), step=fold_id)

        # if trial.should_prune():
        #     raise optuna.TrialPruned()

        # Clear GPU mem post-trial
        if cfg.use_gpu and fold_id % 2 == 0:
            torch.cuda.empty_cache()

    return float(np.mean(losses))


def run_tuning_per_fold(
    outer_train_dataset: Dataset, inner_cv: KFold, logger: Logger, cfg: FixedParams
) -> dict[str, Any]:
    """Single-phase tuning on outer_train; final train/eval."""
    logger.info("Starting Single-Phase Tuning...")

    model_factory = MPNNFactory(cfg)

    def obj(trial: optuna.Trial):
        return inner_cv_objective(trial, outer_train_dataset, inner_cv, model_factory, logger, cfg)

    study = optuna.create_study(
        direction=cfg.opt_direction,
        sampler=optuna.samplers.TPESampler(seed=112),
        pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=cfg.n_epochs),  # Prune early
    )

    # study.enqueue_trial(INIT_PARAMS)  # Define an inital trial I want the study to run
    study.optimize(obj, n_trials=cfg.opt_n_trials, n_jobs=1)

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
    train_losses: Iterable[float] | float,
    train_accs: Iterable[float] | float,
    test_loss: Iterable[float] | float,
    test_acc: Iterable[float] | float,
    best_params: dict,
    logger: Logger,
) -> None:
    """Save the results of an outer fold"""
    fold_data = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_loss": test_loss,
        "test_acc": test_acc,
        **best_params,
    }

    with open(results_file, "w") as f:
        json.dump(fold_data, f)

    logger.info(f"Saved results to {results_file}")


def test_model(
    train_dset: Dataset, test_dset: Dataset, best_params: dict[str, Any], logger: Logger, cfg: FixedParams
) -> tuple[MPNN, float]:
    # Final train on full outer_train
    logger.info("  Final Training on Outer Train...")

    train_loader, test_loader, scaler = make_dataloaders(train_dset, test_dset, cfg)

    trainer = make_trainer(logger, cfg)
    mpnn = MPNNFactory(cfg).build(best_params, scaler)

    train_losses, train_accs = [], []

    trainer.fit(mpnn, train_loader)

    # train_losses.append(train_loss)
    # train_accs.append(train_acc)

    # logger.info(
    #     f"Outer Epoch: [{epoch}]\tTraining Loss: [{train_loss:.2f}]\tReconstruction accuracy: [{train_acc}]"
    # )

    # Outer test
    logger.info("  Evaluating on Outer Test...")
    trainer.test(mpnn, test_loader, weights_only=False)

    test_metrics = trainer.test(mpnn, test_loader)[0]
    test_loss = test_metrics["test_loss"]

    logger.info(f"Test loss: {test_loss:.4f}")

    return mpnn, test_loss

    # test_loss = ...
    # test_acc = ...
    # logger.info(f"  Outer Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Best params: {best_params}")

    # return mpnn, train_losses, train_accs, test_loss, test_acc


def main():
    fold_id = int(sys.argv[1])

    FILE_NAME: str = Path(__file__).stem + f"_id={fold_id}"
    FILES = Files(__file__, FILE_NAME)
    FILES.ensure_dirs()

    cfg = FixedParams()

    logger: Logger = setup_logger(FILES.LOG_FILE)

    full_dataset: data.MoleculeDataset = pd.read_pickle("data/chemprop_dataset1.pkl")
    splits = pd.read_pickle(FILES.SPLITS_FILE)

    train_idxs, test_idxs = splits[fold_id]

    outer_train_dset = Subset(full_dataset, train_idxs)
    outer_test_dset = Subset(full_dataset, test_idxs)

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    study_results = run_tuning_per_fold(outer_train_dset, inner_cv=inner_cv, logger=logger, cfg=cfg)
    best_params = study_results["best_params"]

    # Already save the best parameters of the outer fold
    save_fold_result(FILES.RESULTS_FILE_JSON, 0, 0, 0, 0, best_params, logger)

    mpnn, test_loss = test_model(outer_train_dset, outer_test_dset, best_params, logger, cfg)

    # Save the results of the outer fold
    save_fold_result(FILES.RESULTS_FILE_JSON, 0, 0, test_loss, 0, best_params, logger)

    # Save the weights of the model
    save_model(FILES.RESULTS_FILE_MODEL, mpnn)


if __name__ == "__main__":
    main()
