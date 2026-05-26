from collections.abc import Sequence
from typing import Self

import numpy as np
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from rdkit.Chem.rdchem import Atom, HybridizationType


class CustomMultiHotAtomFeaturizer(MultiHotAtomFeaturizer):
    """A custom MultiHotAtomFeaturizer that allows for selective feature ablation.

    Parameters
    ----------
    keep_features : List[bool], optional
        a list of booleans to indicate which atom features to keep. If None, all features are kept. For any element that is False, the corresponding feature's encoding is set to all zeros. Useful for ablation and SHAP analysis.
    """

    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int | HybridizationType],
        keep_features: list[bool] | None = None,
    ) -> None:
        super().__init__(atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations)

        if keep_features is None:
            keep_features = [True] * (len(self._subfeats) + 2)
        self.keep_features = keep_features

    def __call__(self, atom: Atom | None) -> np.ndarray:
        masked_features = np.zeros(self._MultiHotAtomFeaturizer__size)
        if atom is None:
            return masked_features

        feats = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetChiralTag()),
            int(atom.GetTotalNumHs()),
            atom.GetHybridization(),
        ]

        i = 0
        for feat, choices, keep in zip(feats, self._subfeats, self.keep_features[: len(feats)], strict=True):
            j = choices.get(feat, len(choices))
            if keep:
                masked_features[i + j] = 1
            i += len(choices) + 1

        if self.keep_features[len(feats)]:
            masked_features[i] = int(atom.GetIsAromatic())
        if self.keep_features[len(feats) + 1]:
            masked_features[i + 1] = 0.01 * atom.GetMass()

        return masked_features

    def zero_mask(self) -> np.ndarray:
        """Featurize the atom by setting all bits to zero."""
        return np.zeros(len(self))

    @classmethod
    def from_custom(
        cls,
        keep_features: list[bool] | None = None,
    ) -> Self:
        atomic_nums = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 35, 53]
        degrees = [0, 1, 2, 3, 4, 5]
        formal_charges = []
        chiral_tags = [0, 1, 2, 3]
        num_Hs = [0, 1, 2, 3, 4]
        hybridizations = [
            HybridizationType.S,
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP2D,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ]

        return cls(atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations, keep_features)

    @classmethod
    def from_base(cls, keep_features: list[bool] | None = None) -> Self:
        base = MultiHotAtomFeaturizer.v2()
        return cls(
            base.atomic_nums,
            base.degrees,
            base.formal_charges,
            base.chiral_tags,
            base.num_Hs,
            base.hybridizations,
            keep_features,
        )
