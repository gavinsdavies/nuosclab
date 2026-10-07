"""Tests for the live Panel app wiring."""

import numpy as np
import pytest
from bokeh.models import ColumnDataSource

from nuosclab.app import EXPERIMENT_COLORS, build_panel_app


def test_panel_app_builds_with_controls_and_sources():
    state = build_panel_app()

    assert "experiment" in state.controls
    assert "experiment_logo" in state.controls
    assert "engine" in state.controls
    assert "nuprobe" not in state.controls["engine"].options
    assert {"appearance", "disappearance", "residual", "comparison"} < set(
        state.sources
    )
    assert "grid_2_2" in state.sources
    assert isinstance(state.sources["appearance"], ColumnDataSource)
    assert len(state.sources["appearance"].data["energy_gev"]) == 300
    assert "NOvA" in state.controls["experiment_logo"].object


def test_panel_app_updates_sources_when_controls_change():
    state = build_panel_app()
    before = len(state.sources["appearance"].data["energy_gev"])

    state.controls["n_points"].value = 100
    state.update()

    assert before == 300
    assert len(state.sources["appearance"].data["energy_gev"]) == 100


def test_panel_app_updates_selected_experiment_logo():
    state = build_panel_app()

    state.controls["experiment"].value = "DUNE"
    state.update()

    logo = state.controls["experiment_logo"].object
    assert "DUNE" in logo
    assert EXPERIMENT_COLORS["DUNE"] in logo


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


def test_panel_app_gates_nsi_controls_for_non_nsi_engine():
    state = build_panel_app()
    state.controls["eps_mutau"].value = 0.1

    state.controls["engine"].value = "nufast"
    curves = state.update()

    assert state.controls["eps_mutau"].disabled
    assert state.controls["phase_emu"].disabled
    assert np.allclose(curves.live, curves.standard)

    state.controls["engine"].value = "numpy_ref"
    curves = state.update()

    assert not state.controls["eps_mutau"].disabled
    assert not np.allclose(curves.live, curves.standard)


def test_panel_app_etau_degeneracy_tab_follows_controls():
    state = build_panel_app()
    assert {"sin2_th23", "dm32", "inverted", "etau_energy", "etau_delta_cp_nsi"} < set(
        state.controls
    )

    state.controls["experiment"].value = "DUNE"
    state.update()
    assert state.controls["etau_energy"].end == 10.0
    assert state.controls["etau_energy"].value == 2.5

    before = state.update().live.copy()
    state.controls["sin2_th23"].value = 0.45
    state.controls["dm32"].value = 2.4
    state.controls["etau_energy"].value = 3.0
    after = state.update().live
    assert not np.allclose(before, after)


def test_panel_app_nova_paper_point_button():
    from nuosclab.app import NOVA_NSI_PAPER_POINT

    state = build_panel_app()
    state.controls["experiment"].value = "DUNE"
    state.update()
    state.controls["nova_point"].clicks += 1

    assert state.controls["experiment"].value == "NOvA"
    for name, value in NOVA_NSI_PAPER_POINT.items():
        assert state.controls[name].value == value


def test_panel_app_mass_ordering_and_energy_average():
    from nuosclab.app import _compute_profile_from_controls, _pmns_from_controls

    state = build_panel_app()
    state.controls["dm32"].value = 2.4
    no = _pmns_from_controls(state.controls)
    state.controls["inverted"].value = True
    io = _pmns_from_controls(state.controls)
    # |Δm²₃₂| is the same in both orderings
    assert no.dm31 - no.dm21 == pytest.approx(-(io.dm31 - io.dm21))
    assert io.dm31 < 0

    state.controls["etau_average"].value = "Spectrum"
    pmap = _compute_profile_from_controls(state.controls)
    assert pmap.config.average == "spectrum"
    assert pmap.config.energy_sigma == state.controls["etau_sigma"].value

    state.controls["experiment"].value = "T2K"
    state.update()
    assert state.controls["etau_sigma"].value == pytest.approx(0.12)
