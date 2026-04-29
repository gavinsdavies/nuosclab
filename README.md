# OscLib Explorer

Interactive PMNS + NSI neutrino oscillation probability viewer.

Drag sliders to see how NSI (Non-Standard Interaction) parameters deform
P(νμ→νe) and P(νμ→νμ) vs energy, relative to the standard PMNS curve.
Presets: NOvA (L=810 km, ρ=2.79 g/cm³), DUNE (1300 km), T2K (295 km).

## Setup (one-time)

```bash
python3 -m venv ~/venvs/osclib-explorer
~/venvs/osclib-explorer/bin/pip install -e ".[dev]"
~/venvs/osclib-explorer/bin/pip install ipykernel
~/venvs/osclib-explorer/bin/python -m ipykernel install --user \
    --name=osclib-explorer --display-name "osclib-explorer"
```

## Open the notebook

```bash
cd ~/github/osclib-explorer
~/venvs/osclib-explorer/bin/jupyter lab notebooks/explorer.ipynb
```

Select the **osclib-explorer** kernel, then run all cells.

## Run tests

```bash
~/venvs/osclib-explorer/bin/pytest tests/ -v
```

The `test_vs_osclib` test is skipped until `tests/test_vs_osclib.csv` is
populated — see `tools/osclib_oracle.cc` for instructions.

## Physics

The engine (`osclib_explorer/physics.py`) mirrors OscLib's `PMNS_NSI.cxx`:

1. **PMNS matrix** `U(θ₁₂, θ₁₃, θ₂₃, δ_CP)` — standard PDG convention.
2. **Vacuum Hamiltonian** `H_vac = U · diag(0, Δm²₂₁, Δm²₃₁) · U† / (2E)`.
3. **NSI matter potential** `V = √2 G_F N_e (diag(1,0,0) + ε)` where ε is
   the 3×3 Hermitian NSI matrix parameterised by |ε_eμ|, |ε_eτ|, |ε_μτ|
   and their phases.
4. **Propagator** via `numpy.linalg.eigh` — exact diagonalisation, vectorised
   over the energy array.
5. **Antineutrinos** — H_vac → conj(H_vac), V → −conj(V), matching OscLib.

All constants (G_F, N_A, ℏc) are taken from OscLib/Constants.h (PDG 2024).

## Validation

`tools/osclib_oracle.cc` is a standalone C++ program that links against the
real OscLib `OscCalcPMNS_NSI` and prints reference probabilities to CSV.
Build it on a machine with OscLib+ROOT (FNAL CVMFS), redirect output to
`tests/test_vs_osclib.csv`, and commit. The test `test_vs_osclib.py` then
asserts agreement to <10⁻⁴.
