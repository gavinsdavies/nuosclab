"""Panel app for interactive scientific exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .core import ExplorerConfig, ExplorerCurves, compute_curves
from .engines import ENGINE_REGISTRY
from .physics import NSIParams, PMNSParams
from .presets import PRESETS

if TYPE_CHECKING:
    from bokeh.models import ColumnDataSource


@dataclass(frozen=True)
class AppState:
    """Mutable Panel app components useful for tests and embedding."""

    layout: object
    sources: dict[str, object]
    controls: dict[str, object]
    update: object


def build_panel_app() -> AppState:
    """Build the live Panel/Bokeh scientific app."""
    import panel as pn

    pn.extension("bokeh")

    experiments = list(PRESETS)
    engines = ENGINE_REGISTRY.names()
    experiment = pn.widgets.Select(name="Experiment", value="NOvA", options=experiments)
    engine = pn.widgets.Select(name="Engine", value="numpy_ref", options=list(engines))
    antineutrino = pn.widgets.Checkbox(name="Antineutrino mode", value=False)
    delta_cp = pn.widgets.FloatSlider(
        name="delta CP",
        start=-np.pi,
        end=np.pi,
        step=0.05,
        value=PMNSParams().delta_cp,
    )
    eps_emu = pn.widgets.FloatSlider(name="|eps e-mu|", start=0.0, end=0.30, step=0.005)
    eps_etau = pn.widgets.FloatSlider(name="|eps e-tau|", start=0.0, end=0.30, step=0.005)
    eps_mutau = pn.widgets.FloatSlider(name="|eps mu-tau|", start=0.0, end=0.30, step=0.005)
    phase_emu = pn.widgets.FloatSlider(
        name="delta e-mu",
        start=-np.pi,
        end=np.pi,
        step=0.05,
        value=0.0,
    )
    n_points = pn.widgets.IntSlider(name="Points", start=50, end=800, step=50, value=300)

    controls = {
        "experiment": experiment,
        "engine": engine,
        "antineutrino": antineutrino,
        "delta_cp": delta_cp,
        "eps_emu": eps_emu,
        "eps_etau": eps_etau,
        "eps_mutau": eps_mutau,
        "phase_emu": phase_emu,
        "n_points": n_points,
    }

    initial = _compute_from_controls(controls)
    sources = _make_sources(initial)
    plots = _make_plots(initial, sources)
    residual = _make_residual_plot(initial, sources["residual"])
    status = pn.pane.Markdown(_status_text(initial), sizing_mode="stretch_width")

    def update(*_events: object) -> ExplorerCurves:
        curves = _compute_from_controls(controls)
        _update_sources(curves, sources)
        _update_plot_ranges(curves, plots, residual)
        status.object = _status_text(curves)
        return curves

    for widget in controls.values():
        widget.param.watch(update, "value")

    sidebar = pn.Column(
        "## nuosclab",
        experiment,
        engine,
        antineutrino,
        delta_cp,
        eps_emu,
        eps_etau,
        eps_mutau,
        phase_emu,
        n_points,
        width=280,
        sizing_mode="fixed",
    )
    main = pn.Column(
        status,
        pn.Row(*plots.values(), sizing_mode="stretch_width"),
        residual,
        sizing_mode="stretch_width",
    )
    layout = pn.Row(sidebar, main, sizing_mode="stretch_width")
    layout.servable(title="nuosclab")

    return AppState(layout=layout, sources=sources, controls=controls, update=update)


def _compute_from_controls(controls: dict[str, object]) -> ExplorerCurves:
    pmns = PMNSParams(delta_cp=controls["delta_cp"].value)
    nsi = NSIParams(
        eps_emu=controls["eps_emu"].value,
        eps_etau=controls["eps_etau"].value,
        eps_mutau=controls["eps_mutau"].value,
        delta_emu=controls["phase_emu"].value,
    )
    return compute_curves(
        ExplorerConfig(
            experiment=controls["experiment"].value,
            engine=controls["engine"].value,
            pmns=pmns,
            nsi=nsi,
            antineutrino=controls["antineutrino"].value,
            n_points=controls["n_points"].value,
        )
    )


def _make_sources(curves: ExplorerCurves) -> dict[str, object]:
    from bokeh.models import ColumnDataSource

    return {
        "appearance": ColumnDataSource(_channel_data(curves, 0)),
        "disappearance": ColumnDataSource(_channel_data(curves, 1)),
        "residual": ColumnDataSource(_residual_data(curves)),
    }


def _update_sources(curves: ExplorerCurves, sources: dict[str, object]) -> None:
    sources["appearance"].data = _channel_data(curves, 0)
    sources["disappearance"].data = _channel_data(curves, 1)
    sources["residual"].data = _residual_data(curves)


def _channel_data(curves: ExplorerCurves, beta: int) -> dict[str, np.ndarray]:
    src = 1
    return {
        "energy_gev": curves.energy_gev,
        "live": curves.live[:, beta, src],
        "standard": curves.standard[:, beta, src],
        "nominal": curves.nominal[:, beta, src],
    }


def _residual_data(curves: ExplorerCurves) -> dict[str, np.ndarray]:
    src = 1
    return {
        "energy_gev": curves.energy_gev,
        "appearance": curves.live[:, 0, src] - curves.standard[:, 0, src],
        "disappearance": curves.live[:, 1, src] - curves.standard[:, 1, src],
    }


def _make_plots(curves: ExplorerCurves, sources: dict[str, object]) -> dict[str, object]:
    plots = {
        "appearance": _make_channel_plot(
            "Appearance",
            "P(nu_mu -> nu_e)",
            sources["appearance"],
            curves,
        ),
        "disappearance": _make_channel_plot(
            "Disappearance",
            "P(nu_mu -> nu_mu)",
            sources["disappearance"],
            curves,
        ),
    }
    return plots


def _make_channel_plot(
    title: str,
    y_label: str,
    source: ColumnDataSource,
    curves: ExplorerCurves,
) -> object:
    from bokeh.models import Span
    from bokeh.plotting import figure

    plot = figure(
        title=title,
        x_axis_label="E (GeV)",
        y_axis_label=y_label,
        height=420,
        sizing_mode="stretch_width",
        x_range=curves.preset.E_range,
        y_range=(-0.02, 1.05),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.line(
        "energy_gev",
        "nominal",
        source=source,
        line_color="#8f8f8f",
        line_dash="dotted",
        line_width=1.5,
        legend_label="Nominal PMNS",
    )
    plot.line(
        "energy_gev",
        "standard",
        source=source,
        line_color="#b0b0b0",
        line_dash="dashed",
        line_width=1.8,
        legend_label="NSI = 0",
    )
    plot.line(
        "energy_gev",
        "live",
        source=source,
        line_color="#e25c2a",
        line_width=2.4,
        legend_label="Live",
    )
    plot.add_layout(
        Span(
            location=curves.preset.E_peak,
            dimension="height",
            line_color="#7d7d7d",
            line_dash="dotted",
            line_width=1,
        )
    )
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    return plot


def _make_residual_plot(curves: ExplorerCurves, source: ColumnDataSource) -> object:
    from bokeh.models import Span
    from bokeh.plotting import figure

    plot = figure(
        title="Live - NSI = 0 residuals",
        x_axis_label="E (GeV)",
        y_axis_label="Probability difference",
        height=300,
        sizing_mode="stretch_width",
        x_range=curves.preset.E_range,
        y_range=(-0.25, 0.25),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.line(
        "energy_gev",
        "appearance",
        source=source,
        line_color="#e25c2a",
        line_width=2.0,
        legend_label="Appearance",
    )
    plot.line(
        "energy_gev",
        "disappearance",
        source=source,
        line_color="#3a7abf",
        line_width=2.0,
        legend_label="Disappearance",
    )
    plot.add_layout(
        Span(location=0.0, dimension="width", line_color="#555555", line_width=1)
    )
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    return plot


def _update_plot_ranges(
    curves: ExplorerCurves,
    plots: dict[str, object],
    residual: object,
) -> None:
    for plot in list(plots.values()) + [residual]:
        plot.x_range.start = curves.preset.E_range[0]
        plot.x_range.end = curves.preset.E_range[1]


def _status_text(curves: ExplorerCurves) -> str:
    return (
        f"**{curves.preset.name}** | L = {curves.preset.L_km:g} km | "
        f"rho = {curves.preset.rho_gcc:g} g/cm^3 | "
        f"engine = `{curves.config.engine}`"
    )
