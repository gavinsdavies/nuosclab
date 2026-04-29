"""Matplotlib figure builders for the oscillation explorer."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from .presets import FLAVOR_TEX, ExperimentPreset


def _channel_label(src: int, dst: int, anti: bool) -> str:
    bar = r"\bar" if anti else ""
    sf = FLAVOR_TEX[src].replace(r"\nu", rf"\{bar}\nu")
    df = FLAVOR_TEX[dst].replace(r"\nu", rf"\{bar}\nu")
    return rf"$P({sf} \to {df})$"


def make_two_panel_figure(figsize=(11, 4)) -> tuple[Figure, list]:
    """Return (fig, [ax_appear, ax_disapp]) for the main 2-panel view."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    return fig, list(axes)


def make_3x3_figure(figsize=(13, 10)) -> tuple[Figure, np.ndarray]:
    """Return (fig, axes[3,3]) for the full probability grid."""
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
