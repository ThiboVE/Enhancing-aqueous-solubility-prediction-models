from collections.abc import Sequence

import numpy as np
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from rdkit.Chem.rdchem import Bond, BondType


class CustomMultiHotBondFeaturizer(MultiHotBondFeaturizer):
    """A custom MultiHotBondFeaturizer that allows for selective feature ablation.

    Parameters
    ----------
    keep_features : List[bool], optional
        a list of booleans to indicate which bond features to keep except for nullity. If None, all features are kept. For any element that is False, the corresponding feature's encoding is set to all zeros. Useful for ablation and SHAP analysis.
    """

    def __init__(
        self,
        bond_types: Sequence[BondType] | None = None,
        stereos: Sequence[int] | None = None,
        keep_features: list[bool] | None = None,
    ) -> None:
        super().__init__(bond_types, stereos)

        self._MultiHotBondFeaturizer__size = 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

        if keep_features is None:
            keep_features = [True] * 4
        self.keep_features = keep_features

    def __len__(self) -> int:
        return self._MultiHotBondFeaturizer__size

    def __call__(self, bond: Bond | None) -> np.ndarray:
        masked_features = np.zeros(len(self), int)

        if bond is None:
            masked_features[0] = 1
            return masked_features
        i = 1
        bond_type = bond.GetBondType()

        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if self.keep_features[0] and bt_bit != size:
            masked_features[i + bt_bit] = 1
        i += size - 1

        if self.keep_features[1]:
            masked_features[i] = int(bond.GetIsConjugated())

        if self.keep_features[2]:
            masked_features[i + 1] = int(bond.IsInRing())
        i += 2

        if self.keep_features[3]:
            stereo_bit, _ = self.one_hot_index(int(bond.GetStereo()), self.stereo)
            masked_features[i + stereo_bit] = 1

        return masked_features

    def zero_mask(self) -> np.ndarray:
        """Featurize the bond by setting all bits to zero."""
        return np.zeros(len(self), int)

    @classmethod
    def one_hot_index[T](cls, x: T, xs: Sequence[T]) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.

        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``.
        """
        n = len(xs)
        return xs.index(x) if x in xs else n, n + 1
