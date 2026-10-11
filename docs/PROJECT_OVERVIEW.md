# Project overview

IBX (2-iodoxybenzoic acid) is a widely used hypervalent-iodine oxidant whose poor solubility strongly
affects its practical use. Its crystal is held together by an unusual combination of hypervalent I···O
secondary bonds, hydrogen bonds and dispersion. Understanding these interactions quantitatively is the
basis for rationalising solubility and for designing more soluble derivatives.

The project builds a machine-learning interatomic potential trained on periodic SCAN-D3(BJ) data that
is accurate for the crystal, robust in long molecular dynamics and transferable to strained, thermally
disordered and compressed configurations. The workflow is:

1. generate diverse, source-separated DFT data (distortions, strain, contacts, AIMD, active learning);
2. train competing model families under an identical protocol;
3. validate with physics-based gates (invariance, basin recovery, MD stability) on independent data;
4. iterate data generation where diagnostics show gaps;
5. perform final independent validation before production simulations.

Planned applications include crystal free energies, solution-phase association and derivatives of IBX.
