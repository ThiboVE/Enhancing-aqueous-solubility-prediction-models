from collections.abc import Callable

import numpy as np
from chemprop.data import MoleculeDatapoint
from chemprop.models import MPNN
from lightning import pytorch as pl

from ml_enhance.nn.custom_shap.shap_config import ShapMaskConfig

# ── model wrapper for SHAP ────────────────────────────────────────────────────

type PredictionFn = Callable[[np.ndarray, MoleculeDatapoint, ShapMaskConfig, MPNN, pl.Trainer], float]


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
