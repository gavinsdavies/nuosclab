"""Oscillation engine registry and built-in NumPy reference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
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


ENGINE_REGISTRY = EngineRegistry((NumpyReferenceEngine(),))


def get_engine(name: str = "numpy_ref") -> OscillationEngine:
    """Return a registered oscillation engine by name."""
    return ENGINE_REGISTRY.get(name)
