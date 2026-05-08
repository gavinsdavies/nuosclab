"""Figure builders for the oscillation explorer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .core import ExplorerCurves
from .presets import FLAVOR_TEX, ExperimentPreset

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _channel_label(src: int, dst: int, anti: bool) -> str:
    bar = r"\bar" if anti else ""
    sf = FLAVOR_TEX[src].replace(r"\nu", rf"\{bar}\nu")
    df = FLAVOR_TEX[dst].replace(r"\nu", rf"\{bar}\nu")
    return rf"$P({sf} \to {df})$"


def _plain_channel_label(src: int, dst: int, anti: bool) -> str:
    flavors = ["e", "mu", "tau"]
    particle = "nubar" if anti else "nu"
    return f"P({particle}_{flavors[src]} -> {particle}_{flavors[dst]})"


def make_two_panel_figure(figsize=(11, 4)) -> tuple[Figure, list]:
    """Return (fig, [ax_appear, ax_disapp]) for the main 2-panel view."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    return fig, list(axes)


def make_3x3_figure(figsize=(13, 10)) -> tuple[Figure, np.ndarray]:
    """Return (fig, axes[3,3]) for the full probability grid."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True,
                             sharex=True)
    return fig, axes


def update_two_panel(
    axes: list,
    E: np.ndarray,
    P_live: np.ndarray,
    P_ref: np.ndarray | None,
    preset: ExperimentPreset,
    antineutrino: bool,
) -> None:
    """Redraw the two main panels in-place (clears and redraws).

    P_live and P_ref are shape (N, 3, 3); indexing is [E, beta, alpha].
    Appearance: alpha=1 (νμ), beta=0 (νe).
    Disappearance: alpha=1 (νμ), beta=1 (νμ).
    """
    src = 1  # νμ beam
    pairs = [(0, "Appearance"), (1, "Disappearance")]

    for ax, (dst, title) in zip(axes, pairs):
        ax.cla()
        label = _channel_label(src, dst, antineutrino)

        if P_ref is not None:
            ax.plot(E, P_ref[:, dst, src], color="0.65", lw=1.5,
                    ls="--", label="NSI = 0 (ref)", zorder=1)

        color = "#e25c2a" if not antineutrino else "#3a7abf"
        ax.plot(E, P_live[:, dst, src], color=color, lw=2,
                label=label, zorder=2)

        ax.axvline(preset.E_peak, color="0.5", lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("E  (GeV)")
        ax.set_ylabel("Probability")
        ax.set_xlim(preset.E_range)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=9)


def update_3x3(
    axes: np.ndarray,
    E: np.ndarray,
    P_live: np.ndarray,
    P_ref: np.ndarray | None,
    preset: ExperimentPreset,
    antineutrino: bool,
) -> None:
    """Redraw the 3×3 grid.  Row = destination flavor, col = source flavor."""
    flavors = ["e", "μ", "τ"]
    colors_live = ["#e25c2a", "#3a7abf", "#2a9e4f"]

    for beta in range(3):
        for alpha in range(3):
            ax = axes[beta, alpha]
            ax.cla()

            if P_ref is not None:
                ax.plot(E, P_ref[:, beta, alpha], color="0.72", lw=1.2,
                        ls="--", zorder=1)

            ax.plot(E, P_live[:, beta, alpha], color=colors_live[alpha],
                    lw=1.6, zorder=2)

            ax.axvline(preset.E_peak, color="0.5", lw=0.6, ls=":", alpha=0.6)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlim(preset.E_range)
            if beta == 2:
                ax.set_xlabel("E (GeV)")
            if alpha == 0:
                bar = "̄" if antineutrino else ""
                ax.set_ylabel(f"P(ν{bar}{flavors[alpha]} → ν{bar}{flavors[beta]})")
            else:
                ax.set_ylabel("")


def make_bokeh_two_panel(curves: ExplorerCurves):
    """Return a Bokeh two-panel appearance/disappearance layout.

    The returned object is a standalone Bokeh layout suitable for scripts,
    notebooks, and future Panel embedding. The data source includes live,
    standard, and nominal curves when present.
    """
    from bokeh.layouts import gridplot
    from bokeh.models import ColumnDataSource, Span
    from bokeh.plotting import figure

    config = curves.config
    src = 1
    panels = [(0, "Appearance"), (1, "Disappearance")]
    figures = []

    for dst, title in panels:
        data = {
            "energy_gev": curves.energy_gev,
            "live": curves.live[:, dst, src],
        }
        if curves.standard is not None:
            data["standard"] = curves.standard[:, dst, src]
        if curves.nominal is not None:
            data["nominal"] = curves.nominal[:, dst, src]

        source = ColumnDataSource(data)
        plot = figure(
            title=title,
            x_axis_label="E (GeV)",
            y_axis_label=_plain_channel_label(src, dst, config.antineutrino),
            width=420,
            height=300,
            x_range=curves.preset.E_range,
            y_range=(-0.02, 1.05),
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        if "nominal" in data:
            plot.line(
                "energy_gev",
                "nominal",
                source=source,
                line_color="#8f8f8f",
                line_dash="dotted",
                line_width=1.5,
                legend_label="Nominal PMNS",
            )
        if "standard" in data:
            plot.line(
                "energy_gev",
                "standard",
                source=source,
                line_color="#b0b0b0",
                line_dash="dashed",
                line_width=1.8,
                legend_label="NSI = 0",
            )

        live_color = "#e25c2a" if not config.antineutrino else "#3a7abf"
        plot.line(
            "energy_gev",
            "live",
            source=source,
            line_color=live_color,
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
        figures.append(plot)

    return gridplot([figures], sizing_mode="scale_width")


def make_bokeh_probability_grid(curves: ExplorerCurves):
    """Return a Bokeh 3x3 grid of all flavor-transition probabilities."""
    from bokeh.layouts import gridplot
    from bokeh.models import ColumnDataSource, Span
    from bokeh.plotting import figure

    figures = []
    colors = ["#e25c2a", "#3a7abf", "#2a9e4f"]

    for beta in range(3):
        row = []
        for alpha in range(3):
            data = {
                "energy_gev": curves.energy_gev,
                "live": curves.live[:, beta, alpha],
            }
            if curves.standard is not None:
                data["standard"] = curves.standard[:, beta, alpha]

            source = ColumnDataSource(data)
            plot = figure(
                title=_plain_channel_label(alpha, beta, curves.config.antineutrino),
                x_axis_label="E (GeV)" if beta == 2 else "",
                y_axis_label="Probability" if alpha == 0 else "",
                width=260,
                height=230,
                x_range=curves.preset.E_range,
                y_range=(-0.02, 1.05),
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            if "standard" in data:
                plot.line(
                    "energy_gev",
                    "standard",
                    source=source,
                    line_color="#b0b0b0",
                    line_dash="dashed",
                    line_width=1.4,
                )
            plot.line(
                "energy_gev",
                "live",
                source=source,
                line_color=colors[alpha],
                line_width=1.9,
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
            row.append(plot)
        figures.append(row)

    return gridplot(figures, sizing_mode="scale_width")
