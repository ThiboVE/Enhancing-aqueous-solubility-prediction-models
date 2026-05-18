import numpy as np

from ml_enhance.nn.custom_shap.shap_masks import mask_extra_features, mask_mol_features


class ShapMaskConfig:
    """Defines the structure of the SHAP binary mask vector.

    The mask is a flat binary vector with one bit per feature group:
        [default_atom_groups... | default_bond_groups... | extra_atom_groups... | extra_bond_groups... | mol_features...]

    Parameters
    ----------
    n_default_atom_groups:
        Number of default atom feature groups (= len(_subfeats) + 2).
    n_default_bond_groups:
        Number of default bond feature groups (4 for the default featurizer).
    extra_atom_feature_names:
        Names of the extra atom feature groups (before RBF expansion).
    extra_bond_feature_names:
        Names of the extra bond feature groups (before RBF expansion).
    mol_feature_names:
        Names of the molecule-level features.
    n_rbf:
        Number of RBF basis functions per extra atom/bond feature group.
    """

    def __init__(
        self,
        n_default_atom_groups: int,
        n_default_bond_groups: int,
        extra_atom_feature_names: list[str],
        extra_bond_feature_names: list[str],
        mol_feature_names: list[str],
        n_rbf: int = 10,
    ) -> None:
        self.n_default_atom_groups = n_default_atom_groups
        self.n_default_bond_groups = n_default_bond_groups
        self.extra_atom_feature_names = extra_atom_feature_names
        self.extra_bond_feature_names = extra_bond_feature_names
        self.mol_feature_names = mol_feature_names
        self.n_rbf = n_rbf

        # index ranges within the flat mask
        self.default_atom_start = 0
        self.default_atom_end = n_default_atom_groups

        self.default_bond_start = self.default_atom_end
        self.default_bond_end = self.default_bond_start + n_default_bond_groups

        self.extra_atom_start = self.default_bond_end
        self.extra_atom_end = self.extra_atom_start + len(extra_atom_feature_names)

        self.extra_bond_start = self.extra_atom_end
        self.extra_bond_end = self.extra_bond_start + len(extra_bond_feature_names)

        self.mol_start = self.extra_bond_end
        self.mol_end = self.mol_start + len(mol_feature_names)

        self.total = self.mol_end

    @property
    def feature_names(self) -> list[str]:
        """Human-readable name for each mask bit, for SHAP plots."""
        # default_atom_names = [f"atom_default_{i}" for i in range(self.n_default_atom_groups)]
        # default_bond_names = [f"bond_default_{i}" for i in range(self.n_default_bond_groups)]
        default_atom_names = [
            "atomic_number",
            "degree",
            "formal_charge",
            "chiral_tag",
            "number_of_hydrogens",
            "hybridization",
            "aromaticity",
            "mass",
        ]
        default_bond_names = ["bond_type", "conjugated?", "in_ring?", "stereochemistry"]
        return [
            *default_atom_names,
            *default_bond_names,
            *self.extra_atom_feature_names,
            *self.extra_bond_feature_names,
            *self.mol_feature_names,
        ]

    def unpack(self, mask: np.ndarray) -> dict[str, list[bool]]:
        """Unpack a flat mask vector into its components."""
        return {
            "keep_atom_features": mask[self.default_atom_start : self.default_atom_end].astype(bool).tolist(),
            "keep_bond_features": mask[self.default_bond_start : self.default_bond_end].astype(bool).tolist(),
            "keep_extra_atom_groups": mask[self.extra_atom_start : self.extra_atom_end].astype(bool).tolist(),
            "keep_extra_bond_groups": mask[self.extra_bond_start : self.extra_bond_end].astype(bool).tolist(),
            "keep_mol_features": mask[self.mol_start : self.mol_end].astype(bool).tolist(),
        }


def shap_feature_transform(
    mask: np.ndarray,
    mask_config: ShapMaskConfig,
    V_f: np.ndarray | None,
    E_f: np.ndarray | None,
    x_d: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    unpacked = mask_config.unpack(mask)

    return (
        mask_extra_features(V_f, unpacked["keep_extra_atom_groups"]),
        mask_extra_features(E_f, unpacked["keep_extra_bond_groups"]),
        mask_mol_features(x_d, unpacked["keep_mol_features"]),
    )
