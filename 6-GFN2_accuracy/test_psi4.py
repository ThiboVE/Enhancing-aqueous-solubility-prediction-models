"""Single-point B3LYP/6-31G(2df,p) calculation on GFN2-xTB optimized geometries.
Backend: Psi4

Computes:
  - HOMO/LUMO gap       (eV)
  - Zero-point energy   (kcal/mol)
  - Dipole norm         (Debye)
  - Mean polarizability (Bohr^3)
  - Atomization energy  (kcal/mol)

Input geometry : 2D array, each row = [atomic_number, x, y, z] in Angstrom

Dependencies:
    conda install psi4 -c psi4
"""

import numpy as np
import psi4

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FUNCTIONAL = "b3lyp"
BASIS = "6-31g(2df,p)"

HA_TO_EV = 27.211386245988
HA_TO_KCAL = 627.5094740631
AU_TO_D = 2.541746473
# 1 cm^-1 in Hartree
CM1_TO_HA = 4.5563352812122295e-6

ATOMIC_SYMBOLS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
}

# Ground-state multiplicity (2S+1) per element
ATOM_MULT = {
    "H": 2,
    "B": 2,
    "C": 3,
    "N": 4,
    "O": 3,
    "F": 2,
    "Si": 3,
    "P": 4,
    "S": 3,
    "Cl": 2,
    "Br": 2,
}

METHOD = f"{FUNCTIONAL}/{BASIS}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_spin(geometry: np.ndarray, charge: int) -> int:
    """Returns 2S: 0 for even electron count, 1 for odd."""
    n_elec = sum(int(r[0]) for r in geometry) - charge
    return n_elec % 2


def _build_psi4_mol(geometry: np.ndarray, charge: int, mult: int):
    """Build a psi4 molecule object from geometry array."""
    coord_lines = [f"{ATOMIC_SYMBOLS[int(r[0])]} {r[1]:.8f} {r[2]:.8f} {r[3]:.8f}" for r in geometry]
    geom_str = f"\n{charge} {mult}\n" + "\n".join(coord_lines) + "\nno_reorient\nnounits angstrom\n"
    return psi4.geometry(geom_str)


def _set_options(spin: int):
    """Apply psi4 options for a given spin (2S)."""
    psi4.set_options(
        {
            "reference": "rks" if spin == 0 else "uks",
            "e_convergence": 1e-9,
            "d_convergence": 1e-7,
            "dft_spherical_points": 590,
            "dft_radial_points": 99,
        }
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
def homo_lumo_gap(wfn, spin: int) -> float:
    """HOMO-LUMO gap in eV."""
    if spin == 0:
        eps = np.array(wfn.epsilon_a().to_array())
        occ = np.array(wfn.occupation_a().to_array())
        homo = eps[occ > 0.5].max()
        lumo = eps[occ < 0.5].min()
    else:
        eps_a = np.array(wfn.epsilon_a().to_array())
        eps_b = np.array(wfn.epsilon_b().to_array())
        occ_a = np.array(wfn.occupation_a().to_array())
        occ_b = np.array(wfn.occupation_b().to_array())
        homo = max(eps_a[occ_a > 0.5].max(), eps_b[occ_b > 0.5].max())
        lumo = min(eps_a[occ_a < 0.5].min(), eps_b[occ_b < 0.5].min())
    return (lumo - homo) * HA_TO_EV


def dipole_norm(wfn, mol) -> float:
    """Dipole moment norm in Debye."""
    psi4.oeprop(wfn, "DIPOLE", title="dip")
    dip_vec = wfn.variable("CURRENT DIPOLE")  # a.u., numpy array
    return float(np.linalg.norm(dip_vec)) * AU_TO_D


def mean_polarizability(mol, spin: int) -> float:
    """Mean isotropic polarizability in Bohr^3 via CPHF/CPKS."""
    _set_options(spin)
    psi4.properties(METHOD, properties=["DIPOLE_POLARIZABILITIES"], molecule=mol, return_wfn=False)
    alpha_xx = psi4.variable("DIPOLE POLARIZABILITY XX")
    alpha_yy = psi4.variable("DIPOLE POLARIZABILITY YY")
    alpha_zz = psi4.variable("DIPOLE POLARIZABILITY ZZ")
    return (alpha_xx + alpha_yy + alpha_zz) / 3.0


def zero_point_energy(mol, spin: int) -> float:
    """ZPE in kcal/mol from harmonic frequencies."""
    _set_options(spin)
    _, wfn_freq = psi4.frequency(METHOD, molecule=mol, return_wfn=True)
    freqs_cm1 = np.array(wfn_freq.frequency_analysis["omega"].data)
    freqs_ha = freqs_cm1[freqs_cm1 > 0] * CM1_TO_HA
    return 0.5 * freqs_ha.sum() * HA_TO_KCAL


def atomization_energy(geometry: np.ndarray, mol_energy: float, spin: int) -> float:
    """Atomization energy in kcal/mol (positive = exothermic bond formation)."""
    symbols = [ATOMIC_SYMBOLS[int(r[0])] for r in geometry]
    e_atoms = sum(_atomic_energy(sym) for sym in symbols)
    return (e_atoms - mol_energy) * HA_TO_KCAL


def _atomic_energy(symbol: str) -> float:
    """UKS energy of a single atom. Computed fresh each call (no cache needed
    for single-molecule use; add a module-level dict if batching many molecules).
    """
    psi4.core.clean()
    psi4.set_options(
        {
            "reference": "uks",
            "e_convergence": 1e-9,
            "d_convergence": 1e-7,
        }
    )
    atom_mol = psi4.geometry(f"\n0 {ATOM_MULT[symbol]}\n{symbol} 0.0 0.0 0.0\nno_reorient\n")
    e = psi4.energy(METHOD, molecule=atom_mol)
    psi4.core.clean()
    return e


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
    dict with: total_energy_hartree, spin, homo_lumo_gap_eV, zpe_kcal_mol,
               dipole_norm_debye, mean_polarizability_bohr3,
               atomization_energy_kcal_mol
    """
    psi4.core.be_quiet()  # suppress psi4 output; remove to see logs

    spin = _detect_spin(geometry, charge)
    mult = spin + 1
    mol = _build_psi4_mol(geometry, charge, mult)

    # --- energy + wavefunction ---
    _set_options(spin)
    energy, wfn = psi4.energy(METHOD, return_wfn=True, molecule=mol)

    results = {
        "total_energy_hartree": energy,
        "spin": spin,
        "homo_lumo_gap_eV": homo_lumo_gap(wfn, spin),
        "zpe_kcal_mol": zero_point_energy(mol, spin) if compute_zpe else None,
        "dipole_norm_debye": dipole_norm(wfn, mol),
        "mean_polarizability_bohr3": mean_polarizability(mol, spin),
        "atomization_energy_kcal_mol": atomization_energy(geometry, energy, spin),
    }

    print(f"\n--- Psi4 | {FUNCTIONAL.upper()}/{BASIS} ---")
    for k, v in results.items():
        if v is None:
            print(f"  {k:<35}: not computed")
        elif isinstance(v, float):
            print(f"  {k:<35}: {v:.6g}")
        else:
            print(f"  {k:<35}: {v}")
    return results


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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
