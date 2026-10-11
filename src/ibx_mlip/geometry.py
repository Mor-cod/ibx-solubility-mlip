#!/usr/bin/env python3
"""IBX crystal geometry utilities (fixed covalent topology, periodic-safe molecule unwrapping, contacts, rigid-body ops).

Atom order (verified on the stage-4 SCAN reference crystal and on every v22 structure): 4 molecules x 17 atoms, local order
  O0 O1 I2 C3..C9 O10 O11 H12..H16
  O0 = oxo (I=O 1.862 A), O1 = hydroxyl (I-O 1.996, O-H12), O10 = ring O (C9-O10 1.302, I2-O10 2.317), O11 = carbonyl (C9=O11 1.234).
The 18-bond template below defines molecular identity for the whole campaign (molecule id = atom // 17)."""
import numpy as np
from ase.neighborlist import neighbor_list

APM, NMOL = 17, 4
TEMPLATE_BONDS = [(0, 2), (1, 2), (1, 12), (2, 3), (2, 10), (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (5, 14), (6, 7), (6, 15),
                  (7, 8), (7, 16), (8, 9), (9, 10), (9, 11)]
O_ROLE = {0: "oxo", 1: "hydroxyl", 10: "ring", 11: "carbonyl"}
I_LOCAL, H_HYDROXYL = 2, 12
import os as _os
REF_EXTXYZ = _os.environ.get("IBX_REF_STRUCTURE", "ibx_reference.extxyz")   # path to the reference crystal (not distributed)


def mol_of(i):
    return i // APM


def unwrap_molecules(positions, cell):
    """Return positions with every molecule made contiguous (bond tree through the template, minimum-image bond vectors).
    Molecule COM is placed in the home cell. Pure function of the periodic structure (periodic-invariant up to a lattice shift per molecule)."""
    p = np.array(positions, dtype=float); cell = np.asarray(cell, dtype=float); inv = np.linalg.inv(cell)
    out = p.copy()
    adj = {k: [] for k in range(APM)}
    for a, b in TEMPLATE_BONDS:
        adj[a].append(b); adj[b].append(a)
    for m in range(len(p) // APM):
        base = m * APM; done = {I_LOCAL}; stack = [I_LOCAL]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in done: continue
                d = p[base + v] - out[base + u]; f = d @ inv; f -= np.round(f)
                out[base + v] = out[base + u] + f @ cell; done.add(v); stack.append(v)
        com = out[base:base + APM].mean(0); shift = np.floor(com @ inv) @ cell
        out[base:base + APM] -= shift
    return out


_REF_BOND_LEN = None


def ref_bond_lengths():
    global _REF_BOND_LEN
    if _REF_BOND_LEN is None:
        from ase.io import read
        r = read(REF_EXTXYZ); u = unwrap_molecules(r.positions, r.cell.array)
        _REF_BOND_LEN = np.array([np.linalg.norm(u[b] - u[a]) for a, b in TEMPLATE_BONDS])
    return _REF_BOND_LEN


def integrity(atoms, lo=0.75, hi=1.35, min_hx=1.15, min_heavy=1.90, min_io=2.10, top=None):
    """Covalent-integrity verdict on fixed molecular identity: every template bond (unwrapped, minimum image) within [lo, hi] x reference
    length; no intermolecular H...X < min_hx (proton transfer / merge), heavy-heavy < min_heavy, I...O < min_io (collapse)."""
    if top is None: top = np.arange(len(atoms)).reshape(-1, APM)
    u = unwrap_with_topology(atoms.positions, atoms.cell.array, top); ref = ref_bond_lengths(); worst = 0.0; ok = True
    for m in top:
        L = np.array([np.linalg.norm(u[m[b]] - u[m[a]]) for a, b in TEMPLATE_BONDS]) / ref
        worst = max(worst, float(np.max(np.abs(np.log(L)))))
        ok &= bool(np.all((L >= lo) & (L <= hi)))
    mins = min_inter_checks(atoms, top=top)
    hx = min([v for k, v in mins.items() if "H" in k] + [9.9]); heavy = min([v for k, v in mins.items() if "H" not in k] + [9.9])
    io = mins.get("IO", 9.9)
    verdict = ok and hx >= min_hx and heavy >= min_heavy and io >= min_io
    return dict(ok=bool(verdict), bonds_ok=bool(ok), max_abs_log_bond_ratio=worst, min_HX_inter=hx, min_heavy_inter=heavy, min_IO_inter=io)


def intermolecular_pairs(positions, cell, cutoff=4.0, symbols=None, top=None):
    """All intermolecular pairs within cutoff on molecules unwrapped through the topology (default: 17-atom blocks): a pair is intramolecular
    iff same molecule AND zero image shift between unwrapped coordinates (own periodic images count as intermolecular). Returns i, j, d, vec."""
    from ase import Atoms
    n = len(positions)
    if top is None: top = np.arange(n).reshape(-1, APM)
    u = unwrap_with_topology(positions, cell, top)
    mol = np.empty(n, dtype=np.int64)
    for k, m in enumerate(top): mol[m] = k
    a = Atoms(symbols if symbols is not None else ["X"] * n, positions=u, cell=cell, pbc=True)
    i, j, d, D, S = neighbor_list("ijdDS", a, cutoff)
    intra = (mol[i] == mol[j]) & (np.abs(S).sum(1) == 0)
    keep = ~intra
    return i[keep], j[keep], d[keep], D[keep]


def contact_summary(atoms, cutoff=4.0, top=None):
    """Min intermolecular distance per family (IO, OO, HO, HH, CO, other) + list of I...O contacts (i_I, j_O, d, O role)."""
    sym = np.array(atoms.get_chemical_symbols())
    i, j, d, D = intermolecular_pairs(atoms.positions, atoms.cell.array, cutoff, sym, top)
    fam = {}
    for x, y, dd in zip(i, j, d):
        k = "".join(sorted(sym[x] + sym[y]))
        fam[k] = min(fam.get(k, 9.9), dd)
    io = [(int(x), int(y), float(dd), O_ROLE.get(int(y) % APM, "?")) for x, y, dd in zip(i, j, d) if sym[x] == "I" and sym[y] == "O"]
    return fam, sorted(io, key=lambda t: t[2])


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def move_molecule(pos_unwrapped, m, R=None, t=None, center=None):
    p = pos_unwrapped.copy(); sl = slice(m * APM, (m + 1) * APM); c = p[sl].mean(0) if center is None else center
    if R is not None: p[sl] = (p[sl] - c) @ R.T + c
    if t is not None: p[sl] = p[sl] + t
    return p


def apply_strain(pos_unwrapped, cell, eps, rigid_molecules=False):
    """Symmetric strain eps applied to the cell (rows = lattice vectors: C' = C (1+eps)) and either all atoms (affine) or molecular COMs only."""
    F = np.eye(3) + eps; c2 = np.asarray(cell) @ F
    p = np.asarray(pos_unwrapped)
    if not rigid_molecules:
        return p @ F, c2
    out = p.copy()
    for m in range(len(p) // APM):
        sl = slice(m * APM, (m + 1) * APM); com = p[sl].mean(0); out[sl] = p[sl] - com + com @ F
    return out, c2


def random_rotation(rng, max_deg):
    ax = rng.normal(size=3); ang = np.radians(rng.uniform(0, max_deg))
    return rotation_matrix(ax, ang)


def min_inter_checks(atoms, exclude=None, top=None):
    """Return dict of minimum intermolecular distances per family excluding a given (i,j) pair."""
    sym = np.array(atoms.get_chemical_symbols())
    i, j, d, D = intermolecular_pairs(atoms.positions, atoms.cell.array, 3.2, sym, top)
    out = {}
    for x, y, dd in zip(i, j, d):
        if exclude is not None and {int(x), int(y)} == set(exclude): continue
        k = "".join(sorted(sym[x] + sym[y]))
        out[k] = min(out.get(k, 9.9), float(dd))
    return out


def topology_from_atoms(atoms):
    """Fixed covalent topology for an arbitrary periodic representation whose atoms keep their template local index (atom % 17), e.g. any
    supercell from ase.repeat, any re-wrapping, any molecule permutation. BFS from every iodine over the 18 template bonds; each bonded partner
    is the minimum-image NEAREST atom carrying the partner's local index. Returns mol_atoms (M, 17) int array in template local order.
    Intended to be computed ONCE at setup on an intact structure and then kept fixed (MD topology)."""
    pos = np.asarray(atoms.positions); cell = np.asarray(atoms.cell.array); inv = np.linalg.inv(cell); n = len(pos)
    loc = np.arange(n) % APM
    by_loc = {k: np.where(loc == k)[0] for k in range(APM)}
    adj = {k: [] for k in range(APM)}
    for a, b in TEMPLATE_BONDS:
        adj[a].append(b); adj[b].append(a)
    mols = []; used = np.zeros(n, bool)
    for i0 in by_loc[I_LOCAL]:
        if used[i0]: continue
        m = -np.ones(APM, int); m[I_LOCAL] = i0; stack = [I_LOCAL]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if m[v] >= 0: continue
                cand = by_loc[v]; d = pos[cand] - pos[m[u]]; f = d @ inv; f -= np.round(f); dist = np.linalg.norm(f @ cell, axis=1)
                m[v] = cand[np.argmin(dist)]; stack.append(v)
        if used[m].any() or len(set(m.tolist())) != APM:
            raise ValueError("topology assignment failed (structure not intact or ambiguous)")
        used[m] = True; mols.append(m)
    if not used.all(): raise ValueError("topology assignment left atoms unassigned")
    return np.array(mols)


def unwrap_with_topology(positions, cell, mol_atoms):
    """Unwrap molecules defined by an explicit topology (mol_atoms (M,17) in template local order)."""
    p = np.array(positions, dtype=float); cell = np.asarray(cell, dtype=float); inv = np.linalg.inv(cell); out = p.copy()
    adj = {k: [] for k in range(APM)}
    for a, b in TEMPLATE_BONDS:
        adj[a].append(b); adj[b].append(a)
    for m in mol_atoms:
        done = {I_LOCAL}; stack = [I_LOCAL]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in done: continue
                d = p[m[v]] - out[m[u]]; f = d @ inv; f -= np.round(f); out[m[v]] = out[m[u]] + f @ cell; done.add(v); stack.append(v)
        com = out[m].mean(0); out[m] -= np.floor(com @ inv) @ cell
    return out


def block_topology(n_atoms):
    return np.arange(n_atoms).reshape(-1, APM)
