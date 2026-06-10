# nuosclab

Interactive PMNS + NSI neutrino oscillation probability viewer.

Drag sliders to see how NSI (Non-Standard Interaction) parameters deform
P(νμ→νe) and P(νμ→νμ) vs energy, relative to the standard PMNS curve.
Presets: NOvA (L=810 km, ρ=2.79 g/cm³), DUNE (1300 km), T2K (295 km).

## Setup (one-time)

The package supports either `uv` or standard `venv`/`pip` workflows.

```bash
python3 -m venv /path/to/external/nuosclab-venv
source /path/to/external/nuosclab-venv/bin/activate
pip install -e ".[notebook,dev]"
python -m ipykernel install --user \
    --name=nuosclab --display-name "nuosclab"
```

With `uv`, point the project environment at an external path if you do not want
an in-repository `.venv`:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv sync --extra notebook --extra dev
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv run python -m ipykernel install --user \
    --name=nuosclab --display-name "nuosclab"
```

## Open the notebook

```bash
source /path/to/external/nuosclab-venv/bin/activate
jupyter lab notebooks/explorer.ipynb
```

Select the **nuosclab** kernel, then run all cells.

With `uv`, use the same `UV_PROJECT_ENVIRONMENT=... uv run jupyter lab
notebooks/explorer.ipynb` pattern from setup.

## Run the Panel app

Install the app extra, then serve the live scientific app:

```bash
source /path/to/external/nuosclab-venv/bin/activate
pip install -e ".[app,plot]"
panel serve tools/panel_app.py --show
```

The app provides live controls for experiment, engine, antineutrino mode,
δ_CP, representative NSI magnitudes and phase, energy-grid resolution,
experiment comparison, 3x3 channel inspection, and selected-experiment logo
inspired badges with experiment-matched base colors. Use the save icon in each
Bokeh plot toolbar to export that plot as a PNG.

## Run tests

```bash
source /path/to/external/nuosclab-venv/bin/activate
pytest tests/ -v
```

The `test_vs_osclib` test is skipped until `tests/test_vs_osclib.csv` is
populated — see `tools/osclib_oracle.cc` for instructions.

## Run lint

```bash
source /path/to/external/nuosclab-venv/bin/activate
ruff check .
```

## Frontend API

The notebook uses the same computation API intended for future web frontends:

```python
from nuosclab import ExplorerConfig, compute_curves

curves = compute_curves(ExplorerConfig(experiment="DUNE"))
```

`compute_curves` returns the energy grid plus live, standard NSI=0, and
nominal PMNS probability arrays. Use `curves.as_dict()` for JSON-friendly
payloads.

## Engine Adapters

`nuosclab` separates the explorer API from the underlying oscillation engine.
The default engine is `numpy_ref`, a vectorized NumPy implementation maintained
in this repository. A second built-in engine provides an always-available
cross-check, and optional adapters add independent validation when the
external software is available locally.

- [`NuFast`](https://github.com/PeterDenton/NuFast-LBL) (Denton & Parke,
  [arXiv:2405.02400](https://arxiv.org/abs/2405.02400), MIT) ships as a
  vendored pure-Python port in `nuosclab/nufast.py`, registered as the
  `nufast` engine. It covers standard PMNS in constant-density matter only —
  no NSI — so the app disables the NSI sliders while it is selected. Agreement
  with `numpy_ref` is bounded at ~3×10⁻⁵ in probability by the rounded
  physical constants hardcoded upstream, not by the algorithm.
- [`nuprobe`](https://github.com/shengfong/nuprobe) is used as an optional
  GPL-3.0-licensed second-engine cross-check. Install it separately from its
  GitHub repository, then run the optional adapter tests with `nuprobe`
  importable, for example:

  ```bash
  PYTHONPATH=/path/to/nuprobe pytest tests/test_nuprobe_adapter.py -q
  ```

  The adapter maps `nuosclab` PMNS/NSI parameters into `nuprobe`'s `NuSystem`,
  uses `nuprobe.probability.nuprobe` channel-by-channel, and keeps it out of
  the required dependency set.
- [OscLib](https://github.com/cafana/OscLib) is used as an external C++
  oracle for selected validation points. `nuosclab` does not vendor or require
  OscLib; `tools/osclib_oracle.cc` can be built on a machine that already has
  OscLib and ROOT available, then used to generate a local-only CSV for tests.

## Physics

The engine (`nuosclab/physics.py`) mirrors OscLib's `PMNS_NSI.cxx`:

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
Build it on a machine with OscLib+ROOT (FNAL CVMFS):

```bash
g++ -std=c++17 tools/osclib_oracle.cc \
    $(root-config --cflags --libs) \
    -I ${EIGEN_INC} -I ${OSCLIB_INC} \
    -L ${OSCLIB_LIB} -lOscLib \
    -o tools/osclib_oracle
```

Then generate and test the oracle CSV:

```bash
./tools/osclib_oracle > tests/test_vs_osclib.csv
pytest tests/test_vs_osclib.py -q
```

Keep `tests/test_vs_osclib.csv` local. The test skips when the CSV is absent
and asserts agreement to <10⁻⁴ when a locally generated oracle file exists.
