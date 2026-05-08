# Changelog

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
