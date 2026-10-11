#!/usr/bin/env python3
"""Periodic-invariance test (hard requirement for any MD-ready model). Works with any ASE calculator.

Equivalent periodic representations of the same physical crystal; energy / forces / stress must be identical to numerical tolerance:
  R1 per-atom re-wrap        : every atom moved by a random integer lattice vector (incl. full wrap into the home cell)
  R2 molecule translation    : each whole molecule (17-atom block) moved by a random integer lattice vector
  R3 rigid system shift      : all atoms shifted by a random Cartesian vector (then wrapped)
  R4 cell-vector choice      : an equivalent unimodular cell (a' = a + b etc.; same lattice), positions wrapped into the new cell
  R5 supercell consistency   : E(n1 x n2 x n3) = n E(cell), forces identical on every image, stress identical
  R6 molecule permutation    : molecule blocks reordered (forces permute accordingly)
Tolerances (float64): |dE| <= 1e-6 eV/atom, |dF| <= 1e-5 eV/A, |dS| <= 1e-6 eV/A^3."""
import json, sys
import numpy as np
from ase import Atoms

TOL_E_PER_ATOM, TOL_F, TOL_S = 1e-6, 1e-5, 1e-6
APM = 17


def efs(atoms, calc):
    a = atoms.copy(); a.calc = calc
    return float(a.get_potential_energy()), np.array(a.get_forces()), np.array(a.get_stress(voigt=False))


def variants(atoms, rng):
    n = len(atoms); cell = atoms.cell.array; out = []
    # R1 per-atom random lattice shifts and full home-cell wrap
    a = atoms.copy(); a.positions = a.positions + rng.integers(-2, 3, size=(n, 3)) @ cell; out.append(("R1_atom_random_shift", a, None))
    a = atoms.copy(); a.wrap(); out.append(("R1_home_cell_wrap", a, None))
    # R2 whole-molecule translations
    a = atoms.copy(); p = a.positions.copy()
    for m in range(n // APM):
        p[m * APM:(m + 1) * APM] += rng.integers(-1, 2, size=3) @ cell
    a.positions = p; out.append(("R2_molecule_lattice_translation", a, None))
    # R3 rigid shift
    a = atoms.copy(); a.positions = a.positions + rng.normal(size=3) * 3.0; a.wrap(); out.append(("R3_rigid_shift_wrapped", a, None))
    # R4 equivalent cell choice: a'=a, b'=b, c'=c+a (unimodular)
    a = atoms.copy(); M = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]]); a.set_cell(M @ cell, scale_atoms=False); a.wrap(); out.append(("R4_cell_c_plus_a", a, None))
    # R6 molecule permutation
    nm = n // APM; perm = rng.permutation(nm); idx = np.concatenate([np.arange(m * APM, (m + 1) * APM) for m in perm])
    a = atoms[idx]; out.append(("R6_molecule_permutation", a, idx))
    return out


def run(atoms, calc, tag="", seed=0, supercells=((2, 1, 1), (1, 2, 1), (1, 1, 2))):
    rng = np.random.default_rng(seed); n = len(atoms)
    E0, F0, S0 = efs(atoms, calc); rows = []
    for name, a, idx in variants(atoms, rng):
        E, F, S = efs(a, calc)
        Fref = F0 if idx is None else F0[idx]
        rows.append(dict(test=name, dE_per_atom=abs(E - E0) / n, dF=float(np.abs(F - Fref).max()), dS=float(np.abs(S - S0).max())))
    for sc in supercells:
        a = atoms.repeat(sc); k = int(np.prod(sc)); E, F, S = efs(a, calc)
        rows.append(dict(test=f"R5_supercell_{sc[0]}x{sc[1]}x{sc[2]}", dE_per_atom=abs(E / k - E0) / n, dF=float(np.abs(F - np.tile(F0, (k, 1))).max()),
                         dS=float(np.abs(S - S0).max())))
    for r in rows:
        r["pass"] = bool(r["dE_per_atom"] <= TOL_E_PER_ATOM and r["dF"] <= TOL_F and r["dS"] <= TOL_S)
        r["structure"] = tag
    return rows


if __name__ == "__main__":
    print("library module; see run()")
