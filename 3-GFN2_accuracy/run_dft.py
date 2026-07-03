"""Single-point B3LYP/6-31G(2df,p) calculation on GFN2-xTB optimized geometries.

Input geometry: 2D array where each row is [atomic_number, x, y, z] (Bohr)
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from pyscf import dft, gto
from pyscf.hessian import thermo
from pyscf.prop import polarizability
from rdkit import Chem
from utils import Files

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FUNCTIONAL = "b3lyp"
BASIS = "6-31g(2df,p)"

# HA_TO_EV = nist.HARTREE2EV
# HA_TO_KCAL = nist.HARTREE2KCALMOL
# AU_TO_D = nist.AU2DEBYE

# Ground-state spin (2S) per element for atomic reference calculations
ATOM_SPIN = {
    "H": 1,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 2,
    "F": 1,
    "Si": 2,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
}

# Cache for atomic energies so each element is computed only once
_atom_energy_cache: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Molecule setup
# ---------------------------------------------------------------------------
def build_mol(geometry: np.ndarray, symbol_map: dict[int, str], charge: int = 0) -> gto.Mole:
    """Build a PySCF Mole from a (N, 4) geometry array [an, x, y, z].

    Spin is set to 0 (even electrons) or 1 (odd electrons) automatically.
    """
    atoms = []
    for atom_idx, x, y, z in geometry:
        sym = symbol_map[int(atom_idx)]
        atoms.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")

    # Temporarily build with spin=0 just to count electrons
    mol = gto.Mole(atom="\n".join(atoms), basis=BASIS, charge=charge, spin=0, unit="Angstrom", verbose=0)
    mol.build()

    # Minimum valid spin: 0 for even, 1 for odd electron count
    mol.spin = mol.nelectron % 2
    mol.build()
    return mol


def run_dft(mol: gto.Mole):
    """Run RKS (closed-shell) or UKS (open-shell) and return converged object."""
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = FUNCTIONAL
    mf.grids.level = 4
    mf.conv_tol = 1e-9
    mf.verbose = 3
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge!")
    return mf


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
def homo_lumo_gap(mf) -> float:
    """HOMO-LUMO gap in eV."""
    mo_e, mo_o = mf.mo_energy, mf.mo_occ
    if isinstance(mo_e, np.ndarray):  # RKS
        homo = mo_e[mo_o > 0].max()
        lumo = mo_e[mo_o == 0].min()
    else:  # UKS
        homo = max(mo_e[s][mo_o[s] > 0].max() for s in (0, 1))
        lumo = min(mo_e[s][mo_o[s] == 0].min() for s in (0, 1))
    return lumo - homo


def zero_point_energy(mf) -> float:
    """ZPE in kcal/mol via analytical Hessian. Slowest step."""
    hessian = mf.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mf.mol, hessian)
    thermo_info = thermo.thermo(mf, freq_info["freq_au"], 300, 101325)
    return thermo_info["ZPE"][0]


def dipole_norm(mf) -> float:
    """Dipole moment norm in Debye."""
    return float(np.linalg.norm(mf.dip_moment(unit="AU", verbose=0)))


def mean_polarizability(mf) -> float:
    """Mean isotropic polarizability in Bohr^3 via CPHF/CPKS."""
    pol_cls = polarizability.rks.Polarizability if mf.mol.spin == 0 else polarizability.uks.Polarizability
    pol = pol_cls(mf)
    alpha = pol.polarizability()

    return float(np.trace(alpha) / 3.0)


def atomization_energy(mf) -> float:
    """Atomization energy in kcal/mol (positive = exothermic bond formation)."""
    symbols = [mf.mol.atom_symbol(i) for i in range(mf.mol.natm)]
    e_atoms = sum(_atomic_energy(s) for s in symbols)
    return e_atoms - mf.e_tot  # * HA_TO_KCAL


def _atomic_energy(symbol: str) -> float:
    """UKS energy of a single atom at the same level of theory. Cached."""
    if symbol not in _atom_energy_cache:
        mol = gto.Mole(atom=f"{symbol} 0 0 0", basis=BASIS, spin=ATOM_SPIN[symbol], verbose=0)
        mol.build()
        mf = dft.UKS(mol)
        mf.xc, mf.grids.level, mf.conv_tol = FUNCTIONAL, 4, 1e-9
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"Atomic SCF did not converge for {symbol}!")
        _atom_energy_cache[symbol] = mf.e_tot
    return _atom_energy_cache[symbol]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_properties(mol: gto.Mole, compute_zpe: bool = True) -> dict:
    """Parameters

    ----------
    geometry    : np.ndarray (N, 4) — [atomic_number, x, y, z] in Angstrom
    charge      : molecular charge (default 0)
    compute_zpe : compute ZPE via Hessian — slow, set False to skip

    Returns:
    -------
    dict with: homo_lumo_gap_eV, zpe_kcal_mol, dipole_norm_debye,
               mean_polarizability_bohr3, atomization_energy_kcal_mol,
               total_energy_hartree, spin
    """
    mf = run_dft(mol)

    results = {
        "total_energy_hartree": mf.e_tot,
        "spin": mol.spin,
        "homo_lumo_gap_eV": homo_lumo_gap(mf),
        "zpe_kcal_mol": zero_point_energy(mf) if compute_zpe else None,
        "dipole_norm_debye": dipole_norm(mf),
        "mean_polarizability_bohr3": mean_polarizability(mf),
        "atomization_energy_kcal_mol": atomization_energy(mf),
    }

    print(f"\n--- PySCF | {FUNCTIONAL.upper()}/{BASIS} ---")
    for k, v in results.items():
        if v is None:
            print(f"  {k:<35}: not computed")
        elif isinstance(v, float):
            print(f"  {k:<35}: {v:.6g}")
        else:
            print(f"  {k:<35}: {v}")

    return results


def single_molecule_calculation(molecule: NamedTuple) -> dict:
    smiles = molecule["output_smiles"]
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    charge = Chem.GetFormalCharge(mol)

    atom_symbol_map = {atom.GetAtomMapNum(): atom.GetSymbol() for atom in mol.GetAtoms()}
    geometry = molecule["xyz"]

    print(smiles)
    print(atom_symbol_map)

    pyscf_mol = build_mol(geometry, symbol_map=atom_symbol_map, charge=charge)

    return {"output_smiles": smiles, **compute_properties(pyscf_mol, compute_zpe=True)}


def main() -> None:
    mol_id = int(sys.argv[1])
    file = str(sys.argv[2])

    FILE_NAME: str = Path(__file__).stem + f"_id={mol_id}"
    FILES = Files(__file__, FILE_NAME)
    FILES.ensure_dirs()

    df = pd.read_json(file)

    molecule = df.iloc[mol_id]

    result = single_molecule_calculation(molecule)

    with FILES.RESULTS_FILE_JSON.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
