"""Goal of this file is to generate a .csv for the HPC. In this file, a drastic filtering of the feature set is done such that only the relevant (chemically/physically reasonable) features remain."""

import pandas as pd
from rdkit import Chem


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    irrelevant_qm_features = [
        "molecular_quadrupole_principal_invariant_3",
        "avg_effective_coordination_number",
        "avg_atomic_dipole_norm",
        "min_atomic_dipole_norm",
        "avg_atomic_quadrupole_principal_invariant_2",
        "max_atomic_quadrupole_principal_invariant_2",
        "avg_atomic_quadrupole_principal_invariant_3",
        "min_atomic_quadrupole_principal_invariant_3",
        "max_atomic_quadrupole_principal_invariant_3",
        "std_atomic_quadrupole_principal_invariant_3",
        "avg_atomic_polarizability_mean",
        "max_atomic_polarizability_mean",
        "avg_atomic_polarizability_anisotropy",
        "max_atomic_polarizability_anisotropy",
        "atomization_energy",
        "zero_point_energy",
        "gibbs_free_energy_300K",
        "solvation_energy_thf",
        "solvation_energy_cyclohexane",
        "solvation_energy_dmso",
        "delta_energy",
        "std_energy",
        "delta_gibbs_free_energy_300K",
        "std_gibbs_free_energy_300K",
        "delta_enthalpy",
        "std_enthalpy",
        "delta_entropy_300K",
        "std_entropy_300K",
        "delta_heat_capacity_300K",
        "std_heat_capacity_300K",
        "min_partial_charge_water",
        "max_partial_charge_water",
        "std_partial_charge_water",
        "min_partial_charge_thf",
        "max_partial_charge_thf",
        "std_partial_charge_thf",
        "min_partial_charge_cyclohexane",
        "max_partial_charge_cyclohexane",
        "std_partial_charge_cyclohexane",
        "min_partial_charge_dmso",
        "max_partial_charge_dmso",
        "std_partial_charge_dmso",
        "avg_atomic_charge_dipole_interaction",
        "min_atomic_charge_dipole_interaction",
        "max_atomic_charge_dipole_interaction",
        "std_atomic_charge_dipole_interaction",
        "avg_atomic_charge_quadrupole_interaction",
        "min_atomic_charge_quadrupole_interaction",
        "max_atomic_charge_quadrupole_interaction",
        "std_atomic_charge_quadrupole_interaction",
        "avg_atomic_dipole_dipole_interaction",
        "min_atomic_dipole_dipole_interaction",
        "max_atomic_dipole_dipole_interaction",
        "std_atomic_dipole_dipole_interaction",
    ]
    irrelevant_qm_topo_features = [
        "avg_atomic_sasa",
        "min_atomic_sasa",
        "max_atomic_sasa",
        "std_atomic_sasa",
        "avg_overlap_integral",
        "min_overlap_integral",
        "max_overlap_integral",
        "std_overlap_integral",
        "avg_bond_length",
        "min_bond_length",
        "max_bond_length",
        "std_bond_length",
        "avg_bond_stiffness",
        "min_bond_stiffness",
        "max_bond_stiffness",
        "std_bond_stiffness",
        "avg_bond_energy",
        "min_bond_energy",
        "max_bond_energy",
        "std_bond_energy",
        "num_heavy_H_bonds",
        "rigid_flag",
        "molecular_volume",
        "avg_nuclear_repulsion",
        "min_nuclear_repulsion",
        "max_nuclear_repulsion",
        "std_nuclear_repulsion",
    ]

    irrelevant_topo_features = [
        "MaxAbsEStateIndex",
        "MinAbsEStateIndex",
        "NumRadicalElectrons",
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
        "BertzCT",
        "Chi0",
        "Chi0n",
        "Chi1",
        "Chi1n",
        "Chi2n",
        "Chi3n",
        "Chi4n",
        "Ipc",
        "MolWt",
        "HeavyAtomMolWt",
        "ExactMolWt",
        "NumValenceElectrons",
        "HeavyAtomCount",
        "LabuteASA",
    ]

    return df.drop(irrelevant_qm_features + irrelevant_qm_topo_features + irrelevant_topo_features, axis=1)


def get_num_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    to_normalize = [
        "radius_of_gyration",
        "molecular_sasa",
        "sterimol_L",
        "sterimol_Bmin",
        "sterimol_Bmax",
        "molecular_polarizability_mean",
        "molecular_polarizability_anisotropy",
        "Chi0v",
        "Chi1v",
        "Chi2v",
        "Chi3v",
        "Chi4v",
        "Kappa1",
        "Kappa2",
        "Kappa3",
        # "TPSA",
        "Phi",
        "NumHAcceptors",
        "RingCount",
        "HallKierAlpha",
        "NumHeteroatoms",
        "NumRotatableBonds",
    ]

    new_cols = {f"relative_{feature}": df[feature] / df["n_atoms"] for feature in to_normalize}

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    return df.drop(to_normalize, axis=1)


def main() -> None:
    df = pd.read_csv(r"data\processed_dataset_rerun.csv")
    df = df.copy()

    df = filter_df(df)
    df["n_atoms"] = df["smiles"].map(get_num_atoms)
    df = normalize(df)

    df.to_csv(r"./data/processed_dataset_rerun_relevant.csv", index=False)
    # df.to_csv(r"./data/processed_dataset_rerun_relevant_no_norm.csv", index=False)


if __name__ == "__main__":
    main()
