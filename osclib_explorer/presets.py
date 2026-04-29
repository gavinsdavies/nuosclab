"""Experiment geometry presets."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExperimentPreset:
    name:     str
    L_km:     float
    rho_gcc:  float
    E_range:  tuple[float, float]   # GeV
    E_peak:   float                  # GeV, for the reference line


NOVA  = ExperimentPreset("NOvA",  L_km=810,   rho_gcc=2.79,   E_range=(0.3, 5.0),  E_peak=1.9)
DUNE  = ExperimentPreset("DUNE",  L_km=1300,  rho_gcc=2.848,  E_range=(0.3, 10.0), E_peak=2.5)
T2K   = ExperimentPreset("T2K",   L_km=295,   rho_gcc=2.6,    E_range=(0.1, 2.0),  E_peak=0.6)

PRESETS: dict[str, ExperimentPreset] = {p.name: p for p in (NOVA, DUNE, T2K)}

FLAVOR_LABELS = ["e", "μ", "τ"]
FLAVOR_TEX    = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
