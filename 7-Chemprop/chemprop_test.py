import json
import sys
from dataclasses import asdict, dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from chemprop import data, nn
from chemprop.models import MPNN
from chemprop.models.utils import save_model
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler

from ml_enhance.hpc_utils import setup_logger

type InnerFoldData = list[dict[str, data.MoleculeDataset | StandardScaler]]


@dataclass
class FixedParams:
    use_gpu: bool = torch.cuda.is_available()
    batch_size: int = 100
    num_workers: int = 2
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
    def LIGHTNING_LOG_DIR(self) -> Path:
        return self.log_dir / f"{self.filename}_log"

    @property
    def RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_results.pkl"

    @property
    def RESULTS_FILE_JSON(self) -> Path:
        return self.output_dir / f"{self.filename}_results.json"

    @property
    def RESULTS_FILE_MODEL(self) -> Path:
        return self.output_dir / f"{self.filename}_model.pt"

    @property
    def PFI_RESULTS_FILE(self) -> Path:
        return self.output_dir / f"{self.filename}_PFI_results.csv"


class MPNNFactory:
    def __init__(self, cfg: FixedParams) -> None:
        self.fixed_params = asdict(cfg)

    def build(self, params: dict[str, Any]) -> MPNN:
        p = {**params, **self.fixed_params}

        mp = BondMessagePassing(
            d_h=p["message_hidden_dim"],
            depth=p["depth"],
        )

        agg = MeanAggregation()

        ffn = RegressionFFN(
            input_dim=p["message_hidden_dim"],
            hidden_dim=p["ffn_hidden_dim"],
            n_layers=p["ffn_n_layers"],
            dropout=p.get("dropout", 0.1),
            # output_transform=output_transform,
        )

        return MPNN(mp, agg, ffn, batch_norm=True, metrics=p["metrics"])


def make_dataloaders(
    train_dset: data.MoleculeDataset,
    val_dset: data.MoleculeDataset,
    cfg: FixedParams,
):
    train_loader = data.build_dataloader(
        train_dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=cfg.use_gpu,
    )
    val_loader = data.build_dataloader(
        val_dset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=cfg.use_gpu,
    )

    return train_loader, val_loader


# ----------------------------
# TRAINER
# ----------------------------


def make_trainer(logger: CSVLogger | bool | None, cfg: FixedParams, trial: optuna.Trial | None = None) -> pl.Trainer:
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
        accelerator="gpu" if cfg.use_gpu else "cpu",
        devices=1,
        max_epochs=cfg.n_epochs,
        deterministic=True,
        # callbacks=callbacks,
        # precision="16-mixed" if use_GPU else 32,
    )


def inner_cv_objective(
    trial: optuna.Trial,
    inner_fold_data: InnerFoldData,
    logfile: Path,
    logger: Logger,
    cfg: FixedParams,
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

        train_loader, val_loader = make_dataloaders(train_dset, val_dset, cfg)

        trainer = make_trainer(pl_logger, cfg, trial)
        model = MPNNFactory(cfg).build(params)

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
        if cfg.use_gpu and fold_id % 2 == 0:
            torch.cuda.empty_cache()

    return float(np.mean(losses))


def run_tuning_per_fold(
    inner_fold_data: InnerFoldData, logger: Logger, logfile: Path, cfg: FixedParams
) -> dict[str, Any]:
    """Single-phase tuning on outer_train; final train/eval."""
    logger.info("Starting Single-Phase Tuning...")

    def obj(trial: optuna.Trial):
        return inner_cv_objective(trial, inner_fold_data, logfile, logger, cfg)

    study = optuna.create_study(
        direction=cfg.opt_direction,
        sampler=optuna.samplers.TPESampler(seed=112),
        pruner=optuna.pruners.HyperbandPruner(min_resource=2, max_resource=len(inner_fold_data)),
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
    cfg: FixedParams,
) -> tuple[MPNN, float, float]:
    # Final train on full outer_train
    logger.info("  Final Training on Outer Train...")

    train_loader, test_loader = make_dataloaders(train_dset, test_dset, cfg)

    trainer = make_trainer(logger=False, cfg=cfg)
    mpnn = MPNNFactory(cfg).build(best_params)

    trainer.fit(mpnn, train_loader)

    # Outer test
    logger.info("  Evaluating on Outer Test...")
    trainer.test(mpnn, test_loader, weights_only=False)

    test_metrics = trainer.test(mpnn, test_loader)[0]
    r2_score = test_metrics["test/r2"]
    mse_score = test_metrics["test/mse"]

    logger.info(f"Test accuracy: {r2_score:.4f}")

    return mpnn, r2_score, mse_score


def main() -> None:
    pl.seed_everything(152, workers=True)

    fold_id = int(sys.argv[1])

    FILE_NAME: str = Path(__file__).stem + f"_id={fold_id}"
    FILES = Files(__file__, FILE_NAME)
    FILES.ensure_dirs()

    cfg = FixedParams()

    logger: Logger = setup_logger(FILES.LOG_FILE)

    logger.info(f"use GPU: {cfg.use_gpu}")

    outer_fold_data = torch.load(f"../data/folds_no_added/outer_fold_{fold_id}.pt", weights_only=False)
    outer_train_dset = outer_fold_data["outer_train"]
    outer_test_dset = outer_fold_data["outer_test"]

    inner_fold_data: InnerFoldData = outer_fold_data["inner_folds"]

    study_results = run_tuning_per_fold(
        inner_fold_data=inner_fold_data, logger=logger, logfile=FILES.LIGHTNING_LOG_DIR, cfg=cfg
    )
    best_params = study_results["best_params"]

    # Already save the best parameters of the outer fold
    save_fold_result(FILES.RESULTS_FILE_JSON, 0, 0, best_params, logger)

    mpnn, r2_test, mse_test = test_model(outer_train_dset, outer_test_dset, best_params, logger, cfg)

    save_fold_result(FILES.RESULTS_FILE_JSON, r2_test, mse_test, best_params, logger)

    save_model(FILES.RESULTS_FILE_MODEL, mpnn)


if __name__ == "__main__":
    main()
