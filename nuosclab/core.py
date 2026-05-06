"""Reusable computation layer for notebook and web frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .physics import PMNSParams, NSIParams, oscillation_probabilities
from .presets import ExperimentPreset, PRESETS


@dataclass(frozen=True)
class ExplorerConfig:
    """Complete input state for one explorer calculation."""

    experiment: str = "NOvA"
    pmns: PMNSParams = field(default_factory=PMNSParams)
    nsi: NSIParams = field(default_factory=NSIParams)
    antineutrino: bool = False
    n_points: int = 400
    include_standard: bool = True
    include_nominal: bool = True


@dataclass(frozen=True)
class ExplorerCurves:
    """Computed oscillation curves for an explorer state.

    Arrays use the same probability convention as ``oscillation_probabilities``:
    ``P[n, beta, alpha] = P(nu_alpha -> nu_beta)``.
    """

    config: ExplorerConfig
    preset: ExperimentPreset
    energy_gev: np.ndarray
    live: np.ndarray
    standard: np.ndarray | None
    nominal: np.ndarray | None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly lists and scalar metadata for web frontends."""
        return {
            "config": {
                "experiment": self.config.experiment,
                "antineutrino": self.config.antineutrino,
                "n_points": self.config.n_points,
                "include_standard": self.config.include_standard,
                "include_nominal": self.config.include_nominal,
            },
            "preset": {
                "name": self.preset.name,
                "L_km": self.preset.L_km,
                "rho_gcc": self.preset.rho_gcc,
                "E_range": list(self.preset.E_range),
                "E_peak": self.preset.E_peak,
            },
            "energy_gev": self.energy_gev.tolist(),
            "live": self.live.tolist(),
            "standard": None if self.standard is None else self.standard.tolist(),
            "nominal": None if self.nominal is None else self.nominal.tolist(),
        }


def compute_curves(config: ExplorerConfig) -> ExplorerCurves:
    """Compute live, standard, and nominal curves for an explorer state.

    ``live`` uses the PMNS and NSI values in ``config``.
    ``standard`` uses the configured PMNS values with NSI set to zero.
    ``nominal`` uses default PMNS values with NSI set to zero.
    """
    if config.experiment not in PRESETS:
        known = ", ".join(PRESETS)
        raise ValueError(f"Unknown experiment {config.experiment!r}; expected one of: {known}")
    if config.n_points < 2:
        raise ValueError("n_points must be at least 2")

    preset = PRESETS[config.experiment]
    energy_gev = np.linspace(*preset.E_range, config.n_points)

    live = oscillation_probabilities(
        energy_gev,
        preset.L_km,
        preset.rho_gcc,
        config.pmns,
        config.nsi,
        config.antineutrino,
    )

    standard = None
    if config.include_standard:
        standard = oscillation_probabilities(
            energy_gev,
            preset.L_km,
            preset.rho_gcc,
            config.pmns,
            NSIParams(),
            config.antineutrino,
        )

    nominal = None
    if config.include_nominal:
        nominal = oscillation_probabilities(
            energy_gev,
            preset.L_km,
            preset.rho_gcc,
            PMNSParams(),
            NSIParams(),
            config.antineutrino,
        )

    return ExplorerCurves(
        config=config,
        preset=preset,
        energy_gev=energy_gev,
        live=live,
        standard=standard,
        nominal=nominal,
    )
