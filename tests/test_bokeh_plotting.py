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
