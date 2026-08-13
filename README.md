# nuosclab — Neutrino Oscillation Laboratory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21919616.svg)](https://doi.org/10.5281/zenodo.21919616)

Interactive PMNS + NSI neutrino oscillation probability viewer.

Drag sliders to see how NSI (Non-Standard Interaction) parameters deform
P(νμ→νe) and P(νμ→νμ) vs energy, relative to the standard PMNS curve.
Presets: NOvA (L=810 km, ρ=2.79 g/cm³), DUNE (1300 km), T2K (295 km).

**Validation status:** the default `numpy_ref` engine is cross-checked
against the vendored `nufast` engine (standard PMNS) and, optionally,
against `nuprobe` and an external OscLib oracle (see
[Validation](#validation)). As of v1.0.0 the public API
(`compute_curves`, `ExplorerConfig`, the engine registry) is considered
stable; systematic numerical validation across the full parameter space
against OscLib is ongoing.

## Setup (one-time)

`uv` is the primary, recommended workflow. Point the project environment at an
external path if you do not want an in-repository `.venv`:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv sync --extra notebook --extra dev
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv run python -m ipykernel install --user \
    --name=nuosclab --display-name "nuosclab"
```

<details>
<summary>Alternative: standard <code>venv</code>/<code>pip</code></summary>

```bash
python3 -m venv /path/to/external/nuosclab-venv
source /path/to/external/nuosclab-venv/bin/activate
pip install -e ".[notebook,dev]"
python -m ipykernel install --user \
    --name=nuosclab --display-name "nuosclab"
```

</details>

## Open the notebook

```bash
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv run jupyter lab notebooks/explorer.ipynb
```

Select the **nuosclab** kernel, then run all cells.

<details>
<summary>Alternative: <code>venv</code>/<code>pip</code></summary>

```bash
source /path/to/external/nuosclab-venv/bin/activate
jupyter lab notebooks/explorer.ipynb
```

</details>

## Run the Panel app

Install the app extra, then serve the live scientific app:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv sync --extra app --extra plot
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv run panel serve tools/panel_app.py --show
```

<details>
<summary>Alternative: <code>venv</code>/<code>pip</code></summary>

```bash
source /path/to/external/nuosclab-venv/bin/activate
pip install -e ".[app,plot]"
panel serve tools/panel_app.py --show
```

</details>

The app provides live controls for experiment, engine, antineutrino mode,
δ_CP, representative NSI magnitudes and phase, energy-grid resolution,
experiment comparison, 3x3 channel inspection, and selected-experiment logo
inspired badges with experiment-matched base colors. Use the save icon in each
Bokeh plot toolbar to export that plot as a PNG.

## Bokeh preview (no server)

The Panel app's Bokeh renderers (`nuosclab.plotting.make_bokeh_two_panel`,
`make_bokeh_probability_grid`) can be used standalone, without running
Panel, via `tools/bokeh_preview.py`. It computes one fixed DUNE + NSI
example and writes a static `bokeh-preview.html` you can open directly:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv sync --extra plot
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv run python tools/bokeh_preview.py
```

<details>
<summary>Alternative: <code>venv</code>/<code>pip</code></summary>

```bash
source /path/to/external/nuosclab-venv/bin/activate
pip install -e ".[plot]"
python tools/bokeh_preview.py
```

</details>

Useful for checking a Bokeh rendering change without the overhead of a
Panel session, or for embedding the renderers in your own script.

## Run tests

```bash
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv run pytest tests/ -v
```

<details>
<summary>Alternative: <code>venv</code>/<code>pip</code></summary>

```bash
source /path/to/external/nuosclab-venv/bin/activate
pytest tests/ -v
```

</details>

The `test_vs_osclib` test is skipped until `tests/test_vs_osclib.csv` is
populated — see `tools/osclib_oracle.cc` for instructions.

## Run lint

```bash
UV_PROJECT_ENVIRONMENT=/path/to/external/nuosclab-venv \
    uv run ruff check .
```

<details>
<summary>Alternative: <code>venv</code>/<code>pip</code></summary>

```bash
source /path/to/external/nuosclab-venv/bin/activate
ruff check .
```

</details>

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
  physical constants hardcoded upstream, not by the algorithm. Its original
  license notice is reproduced in
  [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
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
   the 3×3 Hermitian NSI matrix parameterized by |ε_eμ|, |ε_eτ|, |ε_μτ|
   and their phases.
4. **Propagator** via `numpy.linalg.eigh` — exact diagonalization, vectorized
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

## License

MIT — see [`LICENSE`](LICENSE). The vendored `nufast` engine carries its own
upstream MIT copyright notice, reproduced in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). The optional `nuprobe`
adapter depends on GPL-3.0-licensed software installed separately by the
user (see [Engine Adapters](#engine-adapters)); `nuprobe` is never required
to install or run `nuosclab`.

## Citation

See [`CITATION.cff`](CITATION.cff) for citation metadata. The v1.0.0 release
is archived on Zenodo: [10.5281/zenodo.21919616](https://doi.org/10.5281/zenodo.21919616).

## Roadmap

- **Web frontend** — a browser UI built on the existing frontend-neutral
  `compute_curves()` API, so the same computation layer serves notebooks,
  the Panel app, and a future web app without duplicating physics code.
  ([#31](https://github.com/gavinsdavies/nuosclab/issues/31))
- **Broader OscLib validation coverage** — move from selected validation
  points (`tools/osclib_oracle.cc`) to systematic agreement checks across
  the full PMNS + NSI parameter space, including antineutrino and
  varying-density scenarios.
  ([#32](https://github.com/gavinsdavies/nuosclab/issues/32))
- **Additional engine adapters** — evaluate further independent oscillation
  codes as optional cross-check engines, following the same
  `OscillationEngine` protocol used by `numpy_ref`, `nufast`, and `nuprobe`.
  ([#33](https://github.com/gavinsdavies/nuosclab/issues/33))

## Acknowledgements

Portions of this codebase were developed with Claude Code assistance. All
code and content are human-owned, human-reviewed, and human-validated
before release; see [Validation](#validation) for the physics cross-checks
applied.
