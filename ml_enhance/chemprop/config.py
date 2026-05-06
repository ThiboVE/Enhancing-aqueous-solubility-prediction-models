from dataclasses import dataclass


@dataclass
class Config:
    use_atom_features: bool
    use_bond_features: bool
    use_mol_features: bool
    use_custom_atom_featurizer: bool
    use_custom_bond_featurizer: bool
    n_rbf_centers: int = 10
    target_col: str = "solubility"


atom_features: list[str] = [
    "atomic_fukui_minus",
    "atomic_fukui_plus",
    "atomic_dipole_norm",
    "atomic_quadrupole_principal_invariant_2",
    "atomic_quadrupole_principal_invariant_3",
    "atomic_polarizability_mean",
    "atomic_polarizability_anisotropy",
    "atomic_sasa",
    "partial_charge",
]

bond_features: list[str] = [
    "bond_length",
    "bond_stiffness",
    "bond_energy",
    "nuclear_repulsion",
    "atomic_charge_dipole_interaction",
    "atomic_charge_quadrupole_interaction",
    "atomic_dipole_dipole_interaction",
]

mol_features: list[str] = [
    "molecular_dipole_norm",
]  # TODO: fill in molecular feature column names
