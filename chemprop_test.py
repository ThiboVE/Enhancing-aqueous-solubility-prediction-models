import json
from collections.abc import Iterable
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from chemprop import data
from chemprop.models import MPNN
from chemprop.nn import Aggregation, BondMessagePassing, RegressionFFN
from lightning import pytorch as pl
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.model_selection import KFold

# Fixed params
FIXED_PARAMS = {
    "max_atoms": 30,
    "node_vec_len": 16,
    "use_GPU": False,  # Set to True if CUDA available
    "vocab_size": 24,
    "batch_size": 1000,
    "p_dropout": 0.1,
    "n_epochs": 30,
}

PARAMS = {
    "params": {
        # High importance
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        "latent_dim": {"type": "int", "low": 8, "high": 64},
        "n_hidden": {"type": "int", "low": 16, "high": 96},
        "gru_dim": {"type": "int", "low": 8, "high": 64},
        # Medium
        "n_conv_layers": {"type": "int", "low": 1, "high": 4},
        "n_hidden_layers": {"type": "int", "low": 1, "high": 3},
        "n_gru_layers": {"type": "int", "low": 1, "high": 3},
        "n_fc_layers": {"type": "int", "low": 2, "high": 3},
        "embedding_dim": {"type": "int", "low": 8, "high": 24},
        # Low
        "teacher_forcing_ratio": {"type": "float", "low": 0.1, "high": 0.9},
        "beta": {"type": "float", "low": 0.1, "high": 10, "log": True},
    },
    "n_trials": 15,
    "direction": "minimize",
}

# Parameters of the first point I want to explore in the Bayesian optimization algorithm.
INIT_PARAMS = {
    "lr": 0.0013292918943162175,
    "latent_dim": 62,
    "n_hidden": 79,
    "gru_dim": 42,
    "n_conv_layers": 1,
    "n_hidden_layers": 1,
    "n_gru_layers": 1,
    "n_fc_layers": 3,
    "embedding_dim": 18,
    "teacher_forcing_ratio": 0.8759278817295955,
    "beta": 4.622589001020832,
}


# Auto-detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_GPU = DEVICE == "cuda"
FIXED_PARAMS["use_GPU"] = use_GPU

OUTER_SEED = 42
INNER_SEED = 156

DATA_PATH = Path.cwd().parents[0] / "data"


def create_model(
    trial: optuna.Trial | None = None,
    init_params: dict[str, float] | None = None,
    fixed_params: dict[str, float] | None = None,
) -> MPNN:
    """Create cVAE model based on Optuna trial suggestions for the current phase."""
    if init_params is not None:
        params = init_params

    else:
        params = {}
        # Suggest params for current phase
        for param_name, spec in PARAMS["params"].items():
            if spec["type"] == "float":
                if spec.get("log", False):
                    params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], log=True)
                else:
                    params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"])
            elif spec["type"] == "int":
                params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"])

    # Merge with fixed params from prior phases
    if fixed_params:
        params.update(fixed_params)

    # Extract for model construction
    lr = params["learning_rate"]
    latent_dim = params["latent_dim"]
    n_hidden = params["n_hidden"]
    gru_dim = params["gru_dim"]
    embedding_dim = params.get("embedding_dim", 8)  # Default if not tuned
    n_conv_layers = params.get("n_conv_layers", 2)
    n_hidden_layers = params.get("n_hidden_layers", 2)
    n_gru_layers = params.get("n_gru_layers", 2)
    n_fc_layers = params.get("n_fc_layers", 3)
    teacher_forcing_ratio = params.get("teacher_forcing_ratio", 0.5)
    gcn_hidden_nodes = n_hidden + 1  # Derived

    # Build components
    mp = BondMessagePassing(d_h=300, depth=3)

    agg = Aggregation(
        method="mean"  # should be norm or something else as solubility is extensive (dependent on size)
    )

    ffn = RegressionFFN(input_dim=300, hidden_dim=300, n_layers=1, output_dim=1, dropout=0.1)

    return MPNN(mp, agg, ffn)


# ---------------------------------------------------------------


def inner_cv_objective(
    trial: optuna.Trial, outer_train_dataset: Iterable[data.MoleculeDatapoint], inner_cv: KFold
) -> float:
    """Inner folds on outer_train_indices for one trial."""
    inner_losses: list[float] = []

    n_inner_folds = inner_cv.get_n_splits()

    for inner_fold, (inner_train_idxs, inner_val_idxs) in enumerate(
        inner_cv.split(range(len(outer_train_dataset))), start=1
    ):
        logger.info(f"Inner Fold {inner_fold}/{n_inner_folds} for trial {trial.number}...")

        train_data = [outer_train_dataset[i] for i in inner_train_idxs]
        val_data = [outer_train_dataset[i] for i in inner_val_idxs]

        train_dset = data.MoleculeDataset(train_data, featurizer)
        val_dset = data.MoleculeDataset(val_data, featurizer)

        batch_size: int = FIXED_PARAMS["batch_size"]
        num_workers: int = FIXED_PARAMS["num_workers"]

        train_loader = data.build_dataloader(
            train_dset, batch_size=batch_size, num_workers=num_workers, shuffle=True, seed=96
        )
        val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        model = create_model(trial=trial)

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="gpu" if use_GPU else "cpu",
            devices=1,
            max_epochs=20,
            # precision="16-mixed" if use_GPU else 32,
            deterministic=True,
            callbacks=[pruning_callback],
        )

        trainer.fit(model, train_loader, val_loader)
        val_metrics = trainer.validate(model, val_loader, verbose=False)
        val_loss: float = val_metrics[0]["val_loss"]
        inner_losses.append(val_loss)

        logger.info(f"Validation Loss: [{val_loss:.2f}]")

        trial.report(val_loss, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # Clear GPU mem post-trial
        if use_GPU and inner_fold % 2 == 0:
            torch.cuda.empty_cache()

    return float(np.mean(inner_losses))


def run_tuning_per_fold(
    outer_train_dataset: list[data.MoleculeDatapoint], inner_cv: KFold, logger: Logger
) -> dict[str, Any]:
    """Single-phase tuning on outer_train; final train/eval."""
    logger.info("Starting Single-Phase Tuning...")

    def obj(trial: optuna.Trial):
        return inner_cv_objective(trial, outer_train_dataset, inner_cv)

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


def save_fold(fold, train_idx, test_idx, vocab_size, fold_type):
    """Save the data of an outer fold"""
    output_dir = DATA_PATH / "cvae_folds"
    output_dir.mkdir(exist_ok=True)

    fold_data = {
        "fold_id": fold,
        "train_indices": train_idx.tolist(),  # Global indices for Subset
        "test_indices": test_idx.tolist(),
        "n_samples_train": len(train_idx),
        "n_samples_test": len(test_idx),
        "vocab_size": vocab_size,  # For consistency
    }

    fold_file = output_dir / f"fold_{fold}_{fold_type}_data.json"

    with open(fold_file, "w") as f:
        json.dump(fold_data, f)
    logger.info(f"Saved fold {fold} to {fold_file}")


def save_fold_result(fold, train_losses, train_accs, test_loss, test_acc, best_params, fold_type="stratified"):
    """Save the results of an outer fold"""
    output_dir = DATA_PATH / "results"
    output_dir.mkdir(exist_ok=True)

    fold_data = {
        "fold_id": fold,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_loss": test_loss,
        "test_acc": test_acc,
        **best_params,
    }

    fold_file = output_dir / f"fold_{fold}_{fold_type}_results.json"

    with open(fold_file, "w") as f:
        json.dump(fold_data, f)
    logger.info(f"Saved results of fold {fold} to {fold_file}")


def save_model(state_dict, fold, fold_type="stratified"):
    """Save the weights of a model after its final training in an outer fold"""
    output_dir = DATA_PATH / "models"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"fold_{fold}_{fold_type}_model.pt"

    torch.save(state_dict, output_file)
    logger.info(f"Saved model of fold {fold} to {output_dir}")


def main():
    # Load in the dataset

    # Load outer fold data

    # train/test split
    outer_train_data, _, outer_test_data = data.split_data_by_indices(
        data=all_data, train_indices=train_idxs, test_indices=test_idxs
    )

    outer_train_data: list[data.MoleculeDatapoint] = outer_train_data[0]
    outer_test_data: list[data.MoleculeDatapoint] = outer_test_data[0]

    best_params = run_tuning_per_fold(outer_train_data)

    test_dset = data.MoleculeDataset(test_data[0], featurizer)

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()

    # Already save the best parameters of the outer fold
    save_fold_result(fold, 0, 0, 0, 0, best_params)

    # Final train on full outer_train
    logger.info("  Final Training on Outer Train...")
    batch_size = FIXED_PARAMS["batch_size"]
    n_epochs = FIXED_PARAMS["n_epochs"]
    train_loader = get_dataloader(dataset, outer_train_indices, batch_size)
    test_loader = get_dataloader(dataset, outer_test_indices, batch_size)

    model, _, lr = create_model(init_params=best_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    for epoch in range(n_epochs):
        train_loss, train_acc = cVAE_train_model(
            epoch,
            model,
            train_loader,
            optimizer,
            lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params["beta"]),
            FIXED_PARAMS["use_GPU"],
            DEVICE,
            FIXED_PARAMS["max_atoms"],
            FIXED_PARAMS["node_vec_len"],
            token2idx,
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        logger.info(
            f"Outer Epoch: [{epoch}]\tTraining Loss: [{train_loss:.2f}]\tReconstruction accuracy: [{train_acc}]"
        )

    # Outer test
    logger.info("  Evaluating on Outer Test...")
    test_loss, test_acc = cVAE_test_model(
        model,
        test_loader,
        lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params["beta"]),
        FIXED_PARAMS["use_GPU"],
        DEVICE,
        FIXED_PARAMS["max_atoms"],
        FIXED_PARAMS["node_vec_len"],
        token2idx,
    )

    # Save the results of the outer fold
    save_fold_result(fold, train_losses, train_accs, test_loss, test_acc, best_params)
    # Save the weights of the model
    save_model(model.state_dict(), fold)

    logger.info(f"  Outer Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    logger.info(test_loss, test_acc, best_params)


if __name__ == "__main__":
    logger = setup_logging(DATA_PATH)
    logger.info("Logging system initialized")

    main()
