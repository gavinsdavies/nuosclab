"""Tests for the live Panel app wiring."""

from bokeh.models import ColumnDataSource

from nuosclab.app import build_panel_app


def test_panel_app_builds_with_controls_and_sources():
    state = build_panel_app()

    assert "experiment" in state.controls
    assert "engine" in state.controls
    assert set(state.sources) == {"appearance", "disappearance", "residual"}
    assert isinstance(state.sources["appearance"], ColumnDataSource)
    assert len(state.sources["appearance"].data["energy_gev"]) == 300


def test_panel_app_updates_sources_when_controls_change():
    state = build_panel_app()
    before = len(state.sources["appearance"].data["energy_gev"])

    state.controls["n_points"].value = 100
    state.update()

    assert before == 300
    assert len(state.sources["appearance"].data["energy_gev"]) == 100
