# Methods

## Electronic structure
Periodic SCAN-D3(BJ) (CP2K, GPW 800/40 Ry, TZV-quality MOLOPT basis sets, GTH-SCAN pseudopotentials,
2×2×2 Monkhorst–Pack grid for the 68-atom cell, analytical stress, SCF convergence 1e-6). Supercells use
explicitly folded k-point sets that reproduce the unit-cell sampling (perfect supercells reproduce the
unit-cell energy to < 1 meV per cell).

## Ab initio MD
NVT (CSVR thermostat) at 250–450 K with independent velocity seeds; frames are subsampled after
equilibration and whole trajectories are assigned to training or validation.

## Active learning
Candidate pools from model-driven MD (300–500 K, strained cells), relaxations from independent perturbed
structures and compressed I···O contacts. Selection uses a cross-architecture committee (force and energy
disagreement, projected short-contact force uncertainty), instability-precursor quotas and farthest-point
sampling for diversity; selected structures are labelled with SCAN-D3(BJ).

## Data governance
Every structure belongs to a complete source group; splits are assigned per group before labels exist.
Development validation and challenge sets contain only source families not used for fitting.
Blind validation data are kept sealed until a candidate passes all development gates.

## Training
Weighted energy/force/stress loss with down-weighting of extreme-force structures, EMA weights, plateau
learning-rate schedule with a final energy-weighted stage, several random seeds per architecture,
data-parallel CPU training.
