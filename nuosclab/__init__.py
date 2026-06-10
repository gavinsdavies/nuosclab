from .physics import PMNSParams, NSIParams, oscillation_probabilities, pmns_matrix
from .presets import NOVA, DUNE, T2K, PRESETS, ExperimentPreset
from .core import ExplorerConfig, ExplorerCurves, compute_curves
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
    "ENGINE_REGISTRY", "EngineCapabilities", "EngineMetadata", "EngineRegistry",
    "NuFastEngine", "NuprobeEngine", "NumpyReferenceEngine", "OscillationEngine",
    "get_engine",
]
