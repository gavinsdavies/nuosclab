"""Tests for oscillation engine registration."""

import numpy as np
import pytest

from nuosclab import (
    ENGINE_REGISTRY,
    NSIParams,
    PMNSParams,
    NuprobeEngine,
    NumpyReferenceEngine,
    get_engine,
    oscillation_probabilities,
)


def test_default_registry_exposes_numpy_reference_engine():
    engine = get_engine("numpy_ref")

    assert engine.metadata.name == "numpy_ref"
    assert engine.metadata.capabilities.neutrino
    assert engine.metadata.capabilities.antineutrino
    assert engine.metadata.capabilities.nsi
    assert "numpy_ref" in ENGINE_REGISTRY.names()
    assert "nuprobe" in ENGINE_REGISTRY.names()


def test_nuprobe_engine_reports_optional_dependency_status():
    engine = get_engine("nuprobe")

    assert isinstance(engine, NuprobeEngine)
    assert engine.metadata.capabilities.nsi
    assert engine.metadata.availability in {"available", "unavailable"}
    if engine.metadata.availability == "unavailable":
        assert engine.metadata.unavailable_reason is not None


def test_numpy_reference_engine_matches_direct_function():
    engine = NumpyReferenceEngine()
    energy_gev = np.linspace(0.5, 5.0, 9)
    pmns = PMNSParams(delta_cp=-np.pi / 2)
    nsi = NSIParams(eps_etau=0.05, delta_etau=0.2)

    via_engine = engine.probabilities(energy_gev, 810.0, 2.84, pmns, nsi)
    direct = oscillation_probabilities(energy_gev, 810.0, 2.84, pmns, nsi)

    assert np.allclose(via_engine, direct)


def test_unknown_engine_reports_available_names():
    with pytest.raises(ValueError, match="numpy_ref"):
        get_engine("missing")
