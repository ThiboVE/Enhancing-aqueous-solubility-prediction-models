from collections.abc import Sequence

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
