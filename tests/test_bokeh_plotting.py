"""Tests for frontend-neutral Bokeh figure builders."""

from bokeh.layouts import GridPlot
from bokeh.models import ColumnDataSource

from nuosclab import ExplorerConfig, compute_curves
from nuosclab.plotting import make_bokeh_probability_grid, make_bokeh_two_panel


def _sources(layout: GridPlot) -> list[ColumnDataSource]:
    return list(layout.select({"type": ColumnDataSource}))


def test_make_bokeh_two_panel_returns_data_backed_layout():
    curves = compute_curves(ExplorerConfig(n_points=8))
    layout = make_bokeh_two_panel(curves)
    sources = _sources(layout)

    assert isinstance(layout, GridPlot)
    assert len(sources) == 2
    for source in sources:
        assert list(source.data) == ["energy_gev", "live", "standard", "nominal"]
        assert len(source.data["energy_gev"]) == 8
        assert len(source.data["live"]) == 8


def test_make_bokeh_probability_grid_returns_all_channels():
    curves = compute_curves(
        ExplorerConfig(n_points=6, include_nominal=False, antineutrino=True)
    )
    layout = make_bokeh_probability_grid(curves)
    sources = _sources(layout)

    assert isinstance(layout, GridPlot)
    assert len(sources) == 9
    for source in sources:
        assert list(source.data) == ["energy_gev", "live", "standard"]
        assert len(source.data["energy_gev"]) == 6


def test_bokeh_etau_degeneracy_builds_and_updates():
    import numpy as np

    from nuosclab import EtauDegeneracyConfig, PMNSParams, compute_etau_degeneracy
    from nuosclab.plotting import (
        make_bokeh_etau_degeneracy,
        update_bokeh_etau_degeneracy,
    )

    config = EtauDegeneracyConfig(pmns=PMNSParams(delta_cp=0.8 * np.pi),
                                  delta_cp_nsi=-np.pi / 2)
    plot = make_bokeh_etau_degeneracy(compute_etau_degeneracy(config))
    joint = plot.select_one({"name": "etau_joint"}).data_source
    assert len(joint.data["eps"]) > 0

    update_bokeh_etau_degeneracy(
        plot, compute_etau_degeneracy(EtauDegeneracyConfig(experiment="DUNE"))
    )
    assert "DUNE" in plot.title.text


def test_bokeh_etau_profile_builds_and_updates():
    from nuosclab import EtauDegeneracyConfig, compute_etau_profile
    from nuosclab.plotting import make_bokeh_etau_profile, update_bokeh_etau_profile

    small = dict(n_eps=10, n_phase=9, n_delta_cp=6)
    plot = make_bokeh_etau_profile(
        compute_etau_profile(EtauDegeneracyConfig(), **small), tolerance=0.1
    )
    image = plot.select_one({"name": "profile_image"})
    assert image.data_source.data["image"][0].shape == (10, 9)

    update_bokeh_etau_profile(
        plot,
        compute_etau_profile(EtauDegeneracyConfig(experiment="DUNE"), **small),
        tolerance=0.05,
    )
    assert "DUNE" in plot.title.text and "5%" in plot.title.text
