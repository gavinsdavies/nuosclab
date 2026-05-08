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

EXPERIMENT_COLORS = {
    "NOvA": "#183a67",
    "DUNE": "#f58220",
    "T2K": "#2a9e4f",
}
EXPERIMENT_LOGO_SUBTITLES = {
    "NOvA": "Off-axis long-baseline",
    "DUNE": "Deep underground neutrinos",
    "T2K": "Tokai to Kamioka",
}
FLAVOR_LABELS = ("e", "mu", "tau")


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
    engine_options = _available_engine_options()
    experiment = pn.widgets.Select(name="Experiment", value="NOvA", options=experiments)
    engine = pn.widgets.Select(name="Engine", value="numpy_ref", options=engine_options)
    antineutrino = pn.widgets.Checkbox(name="Antineutrino mode", value=False)
    compare_experiments = pn.widgets.Checkbox(name="Compare experiments", value=False)
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
        "compare_experiments": compare_experiments,
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
    grid_plots = _make_grid_plots(initial, sources)
    residual = _make_residual_plot(initial, sources["residual"])
    comparison = _make_comparison_plot(initial, sources["comparison"])
    logo = pn.pane.HTML(
        _experiment_logo_html(initial.config.experiment),
        height=132,
        sizing_mode="stretch_width",
    )
    controls["experiment_logo"] = logo
    status = pn.pane.Markdown(_status_text(initial), sizing_mode="stretch_width")

    def update(*_events: object) -> ExplorerCurves:
        curves = _compute_from_controls(controls)
        _update_sources(curves, sources, controls["compare_experiments"].value)
        _update_plot_ranges(curves, plots, residual, comparison, grid_plots)
        logo.object = _experiment_logo_html(curves.config.experiment)
        status.object = _status_text(curves, controls["compare_experiments"].value)
        return curves

    for widget in (
        experiment,
        engine,
        antineutrino,
        compare_experiments,
        delta_cp,
        eps_emu,
        eps_etau,
        eps_mutau,
        phase_emu,
        n_points,
    ):
        widget.param.watch(update, "value")

    sidebar = pn.Column(
        "## nuosclab",
        logo,
        experiment,
        engine,
        antineutrino,
        compare_experiments,
        delta_cp,
        eps_emu,
        eps_etau,
        eps_mutau,
        phase_emu,
        n_points,
        width=280,
        height=720,
        sizing_mode="fixed",
    )
    summary = pn.Column(
        pn.Row(*plots.values(), sizing_mode="stretch_width"),
        residual,
        sizing_mode="stretch_width",
    )
    grid = pn.GridBox(*grid_plots.values(), ncols=3, sizing_mode="stretch_width")
    compare = pn.Column(comparison, sizing_mode="stretch_width")
    main = pn.Column(
        status,
        pn.Tabs(
            ("Summary", summary),
            ("3x3", grid),
            ("Compare", compare),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
    layout = pn.Row(sidebar, main, sizing_mode="stretch_width")
    layout.servable(title="nuosclab")

    return AppState(layout=layout, sources=sources, controls=controls, update=update)


def _available_engine_options() -> list[str]:
    return [
        metadata.name
        for metadata in ENGINE_REGISTRY.metadata()
        if metadata.availability == "available"
    ]


def _experiment_logo_html(experiment: str) -> str:
    color = EXPERIMENT_COLORS.get(experiment, "#5b6472")
    subtitle = EXPERIMENT_LOGO_SUBTITLES.get(experiment, "Long-baseline neutrinos")
    return f"""
    <div style="
        box-sizing: border-box;
        width: 100%;
        min-height: 112px;
        border: 1px solid #d9dee8;
        border-left: 8px solid {color};
        border-radius: 8px;
        background: #ffffff;
        padding: 14px 16px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
      <div style="
          color: {color};
          font-size: 34px;
          line-height: 1;
          font-weight: 800;
          letter-spacing: 0;
      ">{experiment}</div>
      <div style="
          margin-top: 8px;
          color: #3d4654;
          font-size: 13px;
          font-weight: 600;
      ">{subtitle}</div>
      <div style="
          margin-top: 11px;
          height: 5px;
          border-radius: 999px;
          background: linear-gradient(90deg, {color}, #d7dde6);
      "></div>
    </div>
    """


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

    sources = {
        "appearance": ColumnDataSource(_channel_data(curves, 0)),
        "disappearance": ColumnDataSource(_channel_data(curves, 1)),
        "residual": ColumnDataSource(_residual_data(curves)),
        "comparison": ColumnDataSource(_comparison_data(curves, compare=False)),
    }
    for beta in range(3):
        for alpha in range(3):
            sources[f"grid_{beta}_{alpha}"] = ColumnDataSource(
                _grid_channel_data(curves, beta, alpha)
            )
    return sources


def _update_sources(
    curves: ExplorerCurves,
    sources: dict[str, object],
    compare: bool,
) -> None:
    sources["appearance"].data = _channel_data(curves, 0)
    sources["disappearance"].data = _channel_data(curves, 1)
    sources["residual"].data = _residual_data(curves)
    sources["comparison"].data = _comparison_data(curves, compare)
    for beta in range(3):
        for alpha in range(3):
            sources[f"grid_{beta}_{alpha}"].data = _grid_channel_data(
                curves,
                beta,
                alpha,
            )


def _channel_data(curves: ExplorerCurves, beta: int) -> dict[str, np.ndarray]:
    src = 1
    return {
        "energy_gev": curves.energy_gev,
        "live": curves.live[:, beta, src],
        "standard": curves.standard[:, beta, src],
        "nominal": curves.nominal[:, beta, src],
    }


def _grid_channel_data(
    curves: ExplorerCurves,
    beta: int,
    alpha: int,
) -> dict[str, np.ndarray]:
    return {
        "energy_gev": curves.energy_gev,
        "live": curves.live[:, beta, alpha],
        "standard": curves.standard[:, beta, alpha],
    }


def _residual_data(curves: ExplorerCurves) -> dict[str, np.ndarray]:
    src = 1
    return {
        "energy_gev": curves.energy_gev,
        "appearance": curves.live[:, 0, src] - curves.standard[:, 0, src],
        "disappearance": curves.live[:, 1, src] - curves.standard[:, 1, src],
    }


def _comparison_data(curves: ExplorerCurves, compare: bool) -> dict[str, np.ndarray]:
    src = 1
    data = {}
    names = PRESETS if compare else [curves.preset.name]
    for name in names:
        config = ExplorerConfig(
            experiment=name,
            engine=curves.config.engine,
            pmns=curves.config.pmns,
            nsi=curves.config.nsi,
            antineutrino=curves.config.antineutrino,
            n_points=curves.config.n_points,
            include_nominal=False,
        )
        experiment_curves = compute_curves(config)
        data[f"energy_gev_{name}"] = experiment_curves.energy_gev
        data[name] = experiment_curves.live[:, 0, src]
    for name in PRESETS:
        if name not in data:
            data[f"energy_gev_{name}"] = curves.energy_gev
            data[name] = np.full_like(curves.energy_gev, np.nan)
    return data


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
        line_color=EXPERIMENT_COLORS.get(curves.preset.name, "#e25c2a"),
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


def _make_grid_plots(curves: ExplorerCurves, sources: dict[str, object]) -> dict[str, object]:
    return {
        f"grid_{beta}_{alpha}": _make_grid_channel_plot(
            beta,
            alpha,
            sources[f"grid_{beta}_{alpha}"],
            curves,
        )
        for beta in range(3)
        for alpha in range(3)
    }


def _make_grid_channel_plot(
    beta: int,
    alpha: int,
    source: ColumnDataSource,
    curves: ExplorerCurves,
) -> object:
    from bokeh.models import Span
    from bokeh.plotting import figure

    title = f"P(nu_{FLAVOR_LABELS[alpha]} -> nu_{FLAVOR_LABELS[beta]})"
    plot = figure(
        title=title,
        x_axis_label="E (GeV)" if beta == 2 else "",
        y_axis_label="Probability" if alpha == 0 else "",
        height=250,
        sizing_mode="stretch_width",
        x_range=curves.preset.E_range,
        y_range=(-0.02, 1.05),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.line(
        "energy_gev",
        "standard",
        source=source,
        line_color="#b0b0b0",
        line_dash="dashed",
        line_width=1.4,
        legend_label="NSI = 0",
    )
    plot.line(
        "energy_gev",
        "live",
        source=source,
        line_color=EXPERIMENT_COLORS.get(curves.preset.name, "#e25c2a"),
        line_width=1.9,
        legend_label="Live",
    )
    plot.add_layout(
        Span(
            location=curves.preset.E_peak,
            dimension="height",
            line_color="#8f8f8f",
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


def _make_comparison_plot(curves: ExplorerCurves, source: ColumnDataSource) -> object:
    from bokeh.plotting import figure

    plot = figure(
        title="Appearance comparison",
        x_axis_label="E (GeV)",
        y_axis_label="P(nu_mu -> nu_e)",
        height=520,
        sizing_mode="stretch_width",
        x_range=curves.preset.E_range,
        y_range=(-0.02, 0.25),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    for name, color in EXPERIMENT_COLORS.items():
        plot.line(
            f"energy_gev_{name}",
            name,
            source=source,
            line_color=color,
            line_width=2.2,
            legend_label=name,
        )
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    return plot


def _update_plot_ranges(
    curves: ExplorerCurves,
    plots: dict[str, object],
    residual: object,
    comparison: object,
    grid_plots: dict[str, object],
) -> None:
    for plot in list(plots.values()) + [residual, comparison] + list(grid_plots.values()):
        plot.x_range.start = curves.preset.E_range[0]
        plot.x_range.end = curves.preset.E_range[1]
    color = EXPERIMENT_COLORS.get(curves.preset.name, "#e25c2a")
    for plot in list(plots.values()) + list(grid_plots.values()):
        for renderer in plot.renderers:
            glyph = getattr(renderer, "glyph", None)
            if glyph is not None and getattr(glyph, "line_width", None) in {1.9, 2.4}:
                glyph.line_color = color


def _status_text(curves: ExplorerCurves, compare: bool = False) -> str:
    compare_text = " | comparing experiments" if compare else ""
    return (
        f"**{curves.preset.name}** | L = {curves.preset.L_km:g} km | "
        f"rho = {curves.preset.rho_gcc:g} g/cm^3 | "
        f"engine = `{curves.config.engine}`{compare_text}"
    )
