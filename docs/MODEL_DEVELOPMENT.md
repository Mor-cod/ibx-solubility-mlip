# Model development

Two model families are currently compared on the same data, splits, training protocol and validation
suite. Selection is based on independent validation (source-separated development sets, periodic
invariance, basin recovery, MD stability), not on training error alone.

## Branch A — direct periodic equivariant potential

- MACE-type equivariant message passing (local cutoff ~5 Å, two interaction layers).
- Local, conservative (forces = −∇E), size-extensive; stress from the analytical strain derivative.
- Trained directly against SCAN-D3(BJ) energies, forces and stresses (no baseline potential).
- Screened nuclear (ZBL) repulsion plus an extrapolation safeguard for extreme atomic overlap.

## Branch B — modular intra/intermolecular potential

- E = Σ intramolecular atomic energies + Σ intermolecular atomic energies + short-range terms.
- The intramolecular channel sees only bonds/contacts within each molecule's own image (fixed covalent
  topology); the intermolecular channel sees all other neighbours, including periodic images of the same
  molecule, and is seeded with the intramolecular features.
- An optional, bounded repulsive I···O backstop acts only in the region where all reference data are
  repulsive (≤ 2.20 Å).
- Periodic, size-extensive, with the same short-range protection as Branch A.

## Short-range safeguards

Both families use (i) ZBL screened nuclear repulsion and (ii) an extrapolation guard: a stiff repulsive
pair term plus a smooth lower clamp of the distances seen by the learned channels, both placed below the
shortest interatomic distance present in the reference data, so that they are exactly inactive on every
training and validation structure.

## Code

`src/ibx_mlip/models.py` (model classes, safeguards, ASE calculator), `src/ibx_mlip/geometry.py`
(molecular topology, unwrapping, contact analysis, integrity checks), `src/ibx_mlip/periodic_invariance.py`
(invariance and supercell tests).
