# Changelog

## v0.8.0 - 2026-06-10

- Add a vendored pure-Python port of the NuFast long-baseline algorithm
  (Denton & Parke, arXiv:2405.02400, MIT) as the always-available `nufast`
  engine for standard-PMNS cross-validation.
- Gate the Panel app NSI sliders when the selected engine lacks NSI support
  and note the gating in the status line.
- Document the measured `nufast` vs `numpy_ref` agreement floor (~3e-5,
  set by rounded upstream constants) in the README and tests.
- Update CI to `actions/checkout@v6` and `actions/setup-python@v6` to clear
  the Node.js 20 deprecation warnings.

## v0.7.2 - 2026-05-08

- Add selected-experiment logo-inspired badges to the Panel sidebar.
- Use experiment-matched base colors for NOvA, DUNE, and T2K in badges and
  plots.
- Update the badge when the selected experiment changes.

## v0.7.1 - 2026-05-08

- Hide unavailable optional engines from the live app selector so missing
  adapters cannot break callbacks.
- Add a live 3x3 probability tab and an experiment-comparison tab.
- Add experiment-aware colors and document per-plot PNG saving through the
  Bokeh toolbar.

## v0.7.0 - 2026-05-08

- Add a live Panel/Bokeh scientific app with experiment, engine,
  antineutrino, PMNS, NSI, and grid-resolution controls.
- Add residual plotting and tests that verify Panel control updates refresh
  Bokeh data sources.
- Document the normal `panel serve tools/panel_app.py --show` app command.

## v0.6.0 - 2026-05-08

- Add a Bokeh plotting extra and include Bokeh in the development test
  environment.
- Add frontend-neutral Bokeh renderers for the two-panel and 3x3 probability
  views.
- Add Bokeh renderer tests and a standalone HTML preview script for validation
  before Panel integration.

## v0.5.1 - 2026-05-07

- Add Python 3.13 and 3.14 to the GitHub Actions CI matrix while retaining
  Python 3.10, 3.11, and 3.12 coverage.

## v0.5.0 - 2026-05-07

- Add `nuprobe` as an optional second-engine validation adapter without making
  it a required dependency.
- Add optional `nuprobe` agreement tests that skip when the package is not
  importable.
- Document how `nuosclab` uses `nuprobe` and OscLib for validation.

## v0.4.0 - 2026-05-07

- Add an oscillation engine protocol, metadata, capability flags, and registry.
- Wrap the existing NumPy implementation as the default `numpy_ref` engine.
- Route explorer curve computation through the configured engine while preserving
  existing numerical behavior.

## v0.3.0 - 2026-05-06

- Rename the project and Python package from `osclib-explorer` / `osclib_explorer`
  to `nuosclab`.
- Update documentation, tests, and notebook imports for the new package name.
- Make a clean breaking import rename; no `osclib_explorer` compatibility shim is
  provided.

## v0.2.0 - 2026-05-06

- Add a frontend-neutral curve API for future web and app frontends.
- Document generic `venv`/`pip` and `uv` development workflows.
- Keep generated OscLib oracle CSV files local-only while preserving optional oracle validation.
- Add GitHub Actions CI for Python 3.10, 3.11, and 3.12.
- Add a conservative Ruff lint baseline.
