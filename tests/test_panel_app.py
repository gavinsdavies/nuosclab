"""Tests for the live Panel app wiring."""

from bokeh.models import ColumnDataSource

from nuosclab.app import build_panel_app


def test_panel_app_builds_with_controls_and_sources():
    state = build_panel_app()

    assert "experiment" in state.controls
    assert "engine" in state.controls
    assert "nuprobe" not in state.controls["engine"].options
    assert {"appearance", "disappearance", "residual", "comparison"} < set(
        state.sources
    )
    assert "grid_2_2" in state.sources
    assert isinstance(state.sources["appearance"], ColumnDataSource)
    assert len(state.sources["appearance"].data["energy_gev"]) == 300


def test_panel_app_updates_sources_when_controls_change():
    state = build_panel_app()
    before = len(state.sources["appearance"].data["energy_gev"])

    state.controls["n_points"].value = 100
    state.update()

    assert before == 300
    assert len(state.sources["appearance"].data["energy_gev"]) == 100


def test_panel_app_can_update_3x3_and_experiment_comparison_sources():
    state = build_panel_app()

    state.controls["compare_experiments"].value = True
    state.controls["experiment"].value = "DUNE"
    state.update()

    comparison = state.sources["comparison"].data
    assert {"NOvA", "DUNE", "T2K"} < set(comparison)
    assert {"energy_gev_NOvA", "energy_gev_DUNE", "energy_gev_T2K"} < set(
        comparison
    )
    assert len(state.sources["grid_0_1"].data["live"]) == 300
