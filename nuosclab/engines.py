"""Oscillation engine registry and built-in NumPy reference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Protocol

import numpy as np

from .physics import NSIParams, PMNSParams, oscillation_probabilities


@dataclass(frozen=True)
class EngineCapabilities:
    """Feature flags advertised by an oscillation engine."""

    neutrino: bool = True
    antineutrino: bool = True
    nsi: bool = False
    channels: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        )
    )


@dataclass(frozen=True)
class EngineMetadata:
    """Descriptive metadata for selecting and reporting engine behavior."""

    name: str
    display_name: str
    capabilities: EngineCapabilities
    precision_note: str = ""
    availability: str = "available"
    unavailable_reason: str | None = None


class OscillationEngine(Protocol):
    """Protocol implemented by oscillation backends."""

    metadata: EngineMetadata

    def probabilities(
        self,
        energy_gev: np.ndarray,
        baseline_km: float,
        rho_gcc: float,
        pmns: PMNSParams,
        nsi: NSIParams,
        antineutrino: bool = False,
    ) -> np.ndarray:
        """Return P[n, beta, alpha] for the requested energy grid."""


@dataclass(frozen=True)
class NumpyReferenceEngine:
    """Reference engine backed by the existing vectorized NumPy implementation."""

    metadata: EngineMetadata = field(
        default_factory=lambda: EngineMetadata(
            name="numpy_ref",
            display_name="NumPy reference",
            capabilities=EngineCapabilities(nsi=True),
            precision_note="Vectorized three-flavour PMNS + off-diagonal NSI.",
        )
    )

    def probabilities(
        self,
        energy_gev: np.ndarray,
        baseline_km: float,
        rho_gcc: float,
        pmns: PMNSParams,
        nsi: NSIParams,
        antineutrino: bool = False,
    ) -> np.ndarray:
        return oscillation_probabilities(
            energy_gev,
            baseline_km,
            rho_gcc,
            pmns,
            nsi,
            antineutrino,
        )


@dataclass(frozen=True)
class NuprobeEngine:
    """Optional adapter for the external nuprobe package."""

    metadata: EngineMetadata = field(
        default_factory=lambda: EngineMetadata(
            name="nuprobe",
            display_name="NuProbe",
            capabilities=EngineCapabilities(nsi=True),
            precision_note=(
                "Analytic oscillation probabilities from the optional nuprobe "
                "package; evaluated channel-by-channel over the requested grid."
            ),
            availability=(
                "available" if find_spec("nuprobe") is not None else "unavailable"
            ),
            unavailable_reason=(
                None
                if find_spec("nuprobe") is not None
                else "Install nuprobe from https://github.com/shengfong/nuprobe."
            ),
        )
    )

    def probabilities(
        self,
        energy_gev: np.ndarray,
        baseline_km: float,
        rho_gcc: float,
        pmns: PMNSParams,
        nsi: NSIParams,
        antineutrino: bool = False,
    ) -> np.ndarray:
        try:
            from nuprobe.inputs import NuSystem, create_U_PMNS
            import nuprobe.probability as nuprobe_probability
        except ImportError as exc:
            raise RuntimeError(
                "The nuprobe engine requires the optional nuprobe package. "
                "Install it from https://github.com/shengfong/nuprobe."
            ) from exc

        energy_gev = np.asarray(energy_gev, dtype=float)
        nu_system = NuSystem(3)
        nu_system.set_theta(1, 2, pmns.th12)
        nu_system.set_theta(2, 3, pmns.th23)
        nu_system.set_theta(1, 3, pmns.th13)
        nu_system.set_delta(1, 3, pmns.delta_cp)
        nu_system.set_mass(1, 0.0)
        nu_system.set_mass(2, np.sqrt(pmns.dm21))
        nu_system.set_mass(3, np.sqrt(pmns.dm31))

        mixing = create_U_PMNS(nu_system.theta, nu_system.delta)
        nsi_matrix = _nuprobe_nsi_matrix(nsi)
        if antineutrino:
            nsi_matrix = np.conj(nsi_matrix)

        old_rho_const = nuprobe_probability.rho_const
        nuprobe_probability.rho_const = rho_gcc
        try:
            probabilities = np.empty((len(energy_gev), 3, 3), dtype=float)
            for i, energy in enumerate(energy_gev):
                for alpha in range(3):
                    for beta in range(3):
                        probabilities[i, beta, alpha] = nuprobe_probability.nuprobe(
                            alpha + 1,
                            beta + 1,
                            baseline_km,
                            float(energy),
                            nu_system.mass,
                            mixing,
                            antinu=antineutrino,
                            const_matter=True,
                            V_NSI=nsi_matrix,
                        )
        finally:
            nuprobe_probability.rho_const = old_rho_const

        return probabilities


def _nuprobe_nsi_matrix(nsi: NSIParams) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=complex)
    entries = (
        (0, 1, nsi.eps_emu, nsi.delta_emu),
        (0, 2, nsi.eps_etau, nsi.delta_etau),
        (1, 2, nsi.eps_mutau, nsi.delta_mutau),
    )
    for i, j, magnitude, phase in entries:
        matrix[i, j] = magnitude * np.exp(1j * phase)
        matrix[j, i] = np.conj(matrix[i, j])
    return matrix


class EngineRegistry:
    """Lookup table for available oscillation engines."""

    def __init__(self, engines: tuple[OscillationEngine, ...] = ()) -> None:
        self._engines: dict[str, OscillationEngine] = {}
        for engine in engines:
            self.register(engine)

    def register(self, engine: OscillationEngine) -> None:
        self._engines[engine.metadata.name] = engine

    def get(self, name: str) -> OscillationEngine:
        try:
            return self._engines[name]
        except KeyError as exc:
            known = ", ".join(self._engines)
            raise ValueError(f"Unknown engine {name!r}; expected one of: {known}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._engines)

    def metadata(self) -> tuple[EngineMetadata, ...]:
        return tuple(engine.metadata for engine in self._engines.values())


ENGINE_REGISTRY = EngineRegistry((NumpyReferenceEngine(), NuprobeEngine()))


def get_engine(name: str = "numpy_ref") -> OscillationEngine:
    """Return a registered oscillation engine by name."""
    return ENGINE_REGISTRY.get(name)
