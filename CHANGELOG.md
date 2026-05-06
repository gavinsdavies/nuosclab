# Changelog

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
