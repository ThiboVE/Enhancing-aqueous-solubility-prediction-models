import json
import pickle
import sys
from collections.abc import Iterable
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from chemprop import data, nn
from chemprop.models import MPNN
from chemprop.models.utils import save_model
from chemprop.nn import Aggregation, BondMessagePassing, RegressionFFN
from lightning import pytorch as pl
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset

pl.seed_everything(152, workers=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed params
FIXED_PARAMS: dict[str, Any] = {
    "use_GPU": DEVICE == "cuda",
    "batch_size": 100,
    "p_dropout": 0.1,
    "n_epochs": 30,
    "metrics": [nn.metrics.RMSE(), nn.metrics.MAE()],
    "dropout": 0.1,
}

PARAMS: dict[str, Any] = {
    "params": {
        "message_hidden_dim": {"type": "int", "low": 100, "high": 500},
        "depth": {"type": "int", "low": 1, "high": 5},
        "ffn_hidden_dim": {"type": "int", "low": 100, "high": 500},
        "ffn_n_layers": {"type": "int", "low": 1, "high": 5},
    },
    "n_trials": 30,
    "direction": "minimize",
}


class ParamSampler:
    def __init__(self, param_space: dict) -> None:
        self.param_space = param_space

    def sample(self, trial: optuna.Trial) -> dict[str, int | float]:
        params: dict[str, int | float] = {}
        for name, spec in self.param_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        return params


class MPNNFactory:
    def __init__(self, fixed_params: dict[str, Any] | None = None) -> None:
        self.fixed_params = fixed_params or {}

    def build(self, params: dict[str, Any], scaler: StandardScaler | None) -> MPNN:
        p = {**params, **self.fixed_params}

        mp = BondMessagePassing(
            d_h=p["message_hidden_dim"],
            depth=p["depth"],
        )

        agg = Aggregation(method=p.get("aggregation", "mean"))

        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        ffn = RegressionFFN(
            input_dim=p["message_hidden_dim"],
            hidden_dim=p["ffn_hidden_dim"],
            n_layers=p["ffn_n_layers"],
            output_dim=1,
            dropout=p.get("dropout", 0.1),
            output_transform=output_transform,
        )

        return MPNN(mp, agg, ffn, batch_norm=True, metrics=p["metrics"])


def make_dataloaders(
    train_dset: Dataset,
    val_dset: Dataset,
    scaler: StandardScaler | None = None,
    batch_size: int = FIXED_PARAMS["batch_size"],
    num_workers: int = FIXED_PARAMS["num_workers"],
):
    if scaler is not None:
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, scaler


# ---------------------------------------------------------------


def make_trainer(logger: Logger, trial: optuna.Trial | None = None) -> pl.Trainer:
    callbacks = []

    if trial is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

    return pl.Trainer(
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="gpu" if FIXED_PARAMS["use_GPU"] else "cpu",
        devices=1,
        max_epochs=FIXED_PARAMS["n_epochs"],
        deterministic=True,
        callbacks=callbacks,
        # precision="16-mixed" if use_GPU else 32,
    )


def inner_cv_objective(
    trial: optuna.Trial,
    dataset: Dataset,
    inner_cv: KFold,
    sampler: ParamSampler,
    model_factory: MPNNFactory,
    logger: Logger,
) -> float:
    losses: list[float] = []

    n_inner_folds = inner_cv.get_n_splits()
    for fold_id, (tr_idxs, va_idxs) in enumerate(inner_cv.split(range(len(dataset))), 1):
        logger.info(f"Inner Fold {fold_id}/{n_inner_folds} for trial {trial.number}...")

        train_dset = Subset(dataset, tr_idxs)
        val_dset = Subset(dataset, va_idxs)

        train_loader, val_loader, scaler = make_dataloaders(train_dset, val_dset)

        params = sampler.sample(trial)
        model = model_factory.build(params, scaler)

        trainer = make_trainer(logger, trial)

        trainer.fit(model, train_loader, val_loader)

        val_metrics = trainer.validate(model, val_loader, verbose=False)

        val_loss: float = val_metrics[0]["val_loss"]
        losses.append(val_loss)

        logger.info(f"Validation Loss: [{val_loss:.2f}]")

        # Clear GPU mem post-trial
        if FIXED_PARAMS["use_GPU"] and fold_id % 2 == 0:
            torch.cuda.empty_cache()

    return float(np.mean(losses))


def run_tuning_per_fold(
    outer_train_dataset: Dataset,
    inner_cv: KFold,
    logger: Logger,
) -> dict[str, Any]:
    """Single-phase tuning on outer_train; final train/eval."""
    logger.info("Starting Single-Phase Tuning...")

    model_factory = MPNNFactory()
    sampler = ParamSampler(PARAMS["params"])

    def obj(trial: optuna.Trial):
        return inner_cv_objective(trial, outer_train_dataset, inner_cv, sampler, model_factory, logger)

    study = optuna.create_study(
        direction=PARAMS["direction"],
        sampler=optuna.samplers.TPESampler(seed=112),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=20),  # Prune early
    )

    # study.enqueue_trial(INIT_PARAMS)  # Define an inital trial I want the study to run
    study.optimize(obj, n_trials=PARAMS["n_trials"], n_jobs=1)

    best_params = study.best_params
    logger.info(f"Best Params: {best_params}")
    logger.info(f"Best Inner CV Loss: {study.best_value:.4f}")

    return best_params


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
    train_dset: Dataset, test_dset: Dataset, best_params: dict[str, Any], logger: Logger
) -> tuple[MPNN, list[float], list[float], float, float]:
    # Final train on full outer_train
    logger.info("  Final Training on Outer Train...")

    train_loader, test_loader, scaler = make_dataloaders(train_dset, test_dset)

    trainer = make_trainer(logger)
    mpnn = MPNNFactory().build(best_params, scaler)

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
    test_loss = ...
    test_acc = ...
    logger.info(f"  Outer Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Best params: {best_params}")

    return mpnn, train_losses, train_accs, test_loss, test_acc


def main():
    fold_id = int(sys.argv[1])

    BASE: Path = Path(__file__).parent

    FILE_NAME: str = Path(__file__).stem + f"_id={fold_id}"

    FILES = Files(__file__, FILE_NAME)
    FILES.ensure_dirs()

    logger: Logger = setup_logger(FILES.LOG_FILE)

    full_dataset: data.MoleculeDataset = ...

    with FILES.SPLITS_FILE.open("rb") as f:
        splits = pickle.load(f)

    train_idxs, test_idxs = splits[fold_id]

    outer_train_dset = Subset(full_dataset, train_idxs)
    outer_test_dset = Subset(full_dataset, test_idxs)

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    best_params = run_tuning_per_fold(outer_train_dset, inner_cv=inner_cv, logger=logger)

    # Already save the best parameters of the outer fold
    save_fold_result(FILES.RESULTS_FILE_JSON, 0, 0, 0, 0, best_params, logger)

    mpnn, train_losses, train_accs, test_loss, test_acc = test_model(
        outer_train_dset, outer_test_dset, best_params, logger
    )

    # Save the results of the outer fold
    save_fold_result(FILES.RESULTS_FILE_JSON, train_losses, train_accs, test_loss, test_acc, best_params, logger)

    # Save the weights of the model
    save_model(FILES.RESULTS_FILE_MODEL, mpnn)


if __name__ == "__main__":
    main()
