# Progress

High-level timeline of the potential-development work. Ongoing research.

1. **Initial physics-guided model development.** Delta-learning corrections on top of a pretrained
   baseline reached good accuracy on near-equilibrium crystal data.
   *Lesson:* accuracy near equilibrium does not guarantee robustness under relaxation or MD.

2. **Identification of short-range and false-basin failures.** Relaxations from distorted crystals
   collapsed into compact, spurious low-energy structures; short intermolecular I···O contacts were
   systematically mis-described.
   *Lesson:* global or gated corrections can behave non-physically in compressed configurations, and
   the baseline itself contained spurious minima.

3. **Independent DFT contact and basin-validation campaign.** New SCAN-D3(BJ) data for compressed I···O
   contacts and collective-displacement paths, built from independent parent structures.
   *Lesson:* the I···O wall is universally repulsive below ~2.20 Å; along collective rigid displacements
   DFT shows no competing basin. The baseline was useful diagnostically but not sufficiently robust as a
   production component (it is also not supercell-consistent).

4. **Architecture rebuild.** Periodic, local, size-extensive models trained directly on SCAN-D3(BJ):
   a direct equivariant model and a modular intra/intermolecular model, both passing strict
   periodic-invariance and supercell tests.
   *Lesson:* local periodic architectures substantially improved robustness; delta-learning on the old
   baseline was consistently less accurate than direct training.

5. **AIMD and active-learning expansion.** Multi-temperature AIMD plus several committee-driven
   active-learning waves; committee disagreement decreased steadily from wave to wave.
   *Lesson:* targeted data for distorted and strongly compressed configurations reduced spurious local
   minima; extreme atomic overlaps additionally required explicit short-range safeguards.

6. **Current model generations.** Leading models recover the physical crystal basin from the large
   majority of independent perturbed starts, conserve energy in NVE and remain stable in NVT at
   300–400 K. Remaining failures arise from starting structures with unphysical atomic overlaps, from
   which even DFT relaxes into chemically reacted (oxygen-transfer) states outside the scope of a
   non-reactive crystal potential.

7. **Next: production validation and solubility applications.** Longer MD, final independent
   validation, then extension to derivatives and to solution/association studies.
