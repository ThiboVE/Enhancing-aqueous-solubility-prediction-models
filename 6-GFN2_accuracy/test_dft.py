"""Single-point B3LYP/6-31G(2df,p) calculation on GFN2-xTB optimized geometries.

Input geometry: 2D array where each row is [atomic_number, x, y, z] (Bohr)
"""

import json

import numpy as np
from pyscf import dft, gto
from pyscf.data import nist
from pyscf.hessian import thermo
from pyscf.prop import polarizability

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FUNCTIONAL = "b3lyp"
BASIS = "6-31g(2df,p)"

HA_TO_EV = nist.HARTREE2EV
HA_TO_KCAL = nist.HARTREE2KCALMOL
AU_TO_D = nist.AU2DEBYE

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
def build_mol(geometry: np.ndarray, charge: int = 0) -> gto.Mole:
    """Build a PySCF Mole from a (N, 4) geometry array [an, x, y, z].

    Spin is set to 0 (even electrons) or 1 (odd electrons) automatically.
    """
    atoms = []
    for an, x, y, z in geometry:
        sym = ATOMIC_SYMBOLS[int(an)]
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
    h = mf.Hessian().kernel()
    freqs = thermo.harmonic_analysis(mf.mol, h)["freq_au"]
    real_positive = freqs[freqs > 0].real
    return 0.5 * real_positive.sum() * HA_TO_KCAL


def dipole_norm(mf) -> float:
    """Dipole moment norm in Debye."""
    return float(np.linalg.norm(mf.dip_moment(unit="AU", verbose=0))) * AU_TO_D


def mean_polarizability(mf) -> float:
    """Mean isotropic polarizability in Bohr^3 via CPHF/CPKS."""
    pol_cls = polarizability.rhf.Polarizability if mf.mol.spin == 0 else polarizability.uhf.Polarizability
    return float(np.trace(pol_cls(mf).kernel()) / 3.0)


def atomization_energy(mf) -> float:
    """Atomization energy in kcal/mol (positive = exothermic bond formation)."""
    symbols = [mf.mol.atom_symbol(i) for i in range(mf.mol.natm)]
    e_atoms = sum(_atomic_energy(s) for s in symbols)
    return (e_atoms - mf.e_tot) * HA_TO_KCAL


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
def compute_properties(geometry: np.ndarray, charge: int = 0, compute_zpe: bool = True) -> dict:
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
    mol = build_mol(geometry, charge)
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

    print(f"\n--- Results ({FUNCTIONAL.upper()}/{BASIS}) ---")
    for k, v in results.items():
        print(f"  {k:<35}: {v:.6g}" if v is not None else f"  {k:<35}: not computed")

    return results


# ---------------------------------------------------------------------------

# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Methane (CH4) — spin auto-detected as 0 (10 electrons, even)
    geometry_ch4 = np.array(
        [
            [6, 0.000, 0.000, 0.000],
            [1, 0.629, 0.629, 0.629],
            [1, -0.629, -0.629, 0.629],
            [1, -0.629, 0.629, -0.629],
            [1, 0.629, -0.629, -0.629],
        ]
    )

    results = compute_properties(geometry_ch4, charge=0, compute_zpe=True)

    with open("dft_result_test.json", "w") as f:
        json.dump(results)
