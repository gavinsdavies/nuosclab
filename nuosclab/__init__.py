from .physics import PMNSParams, NSIParams, oscillation_probabilities, pmns_matrix
from .presets import NOVA, DUNE, T2K, PRESETS, ExperimentPreset
from .core import ExplorerConfig, ExplorerCurves, compute_curves
from .degeneracy import (
    EtauDegeneracyConfig,
    EtauDegeneracyMap,
    EtauProfileMap,
    compute_etau_degeneracy,
    compute_etau_profile,
)
from .engines import (
    ENGINE_REGISTRY,
    EngineCapabilities,
    EngineMetadata,
    EngineRegistry,
    NuFastEngine,
    NuprobeEngine,
    NumpyReferenceEngine,
    OscillationEngine,
    get_engine,
)

__all__ = [
    "PMNSParams", "NSIParams", "oscillation_probabilities", "pmns_matrix",
    "NOVA", "DUNE", "T2K", "PRESETS", "ExperimentPreset",
    "ExplorerConfig", "ExplorerCurves", "compute_curves",
    "EtauDegeneracyConfig", "EtauDegeneracyMap", "EtauProfileMap",
    "compute_etau_degeneracy", "compute_etau_profile",
    "ENGINE_REGISTRY", "EngineCapabilities", "EngineMetadata", "EngineRegistry",
    "NuFastEngine", "NuprobeEngine", "NumpyReferenceEngine", "OscillationEngine",
    "get_engine",
]
