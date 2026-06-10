"""Tests for the vendored NuFast engine."""

import numpy as np
import pytest

from nuosclab import NSIParams, NuFastEngine, PMNSParams, get_engine
from nuosclab.physics import oscillation_probabilities
from nuosclab.presets import PRESETS

# Measured worst-case |nufast - numpy_ref| across the presets below is
# 2.6e-5, identical for n_newton in {1, 2} and the exact cubic mode, so the
# floor is the rounded physical constants hardcoded upstream in
# NuFast_LBL.cpp (six significant figures), not the Newton approximation.
AGREEMENT_ATOL = 5e-5


@pytest.mark.parametrize("antineutrino", [False, True])
@pytest.mark.parametrize("experiment", list(PRESETS))
def test_nufast_agrees_with_numpy_ref(experiment, antineutrino):
    preset = PRESETS[experiment]
    energy_gev = np.linspace(*preset.E_range, 200)
    pmns = PMNSParams()
    nsi = NSIParams()
    engine = get_engine("nufast")

    got = engine.probabilities(
        energy_gev, preset.L_km, preset.rho_gcc, pmns, nsi, antineutrino
    )
    ref = oscillation_probabilities(
        energy_gev, preset.L_km, preset.rho_gcc, pmns, nsi, antineutrino
    )

    assert got.shape == ref.shape == (200, 3, 3)
    assert np.allclose(got, ref, atol=AGREEMENT_ATOL)


def test_nufast_is_registered_and_always_available():
    engine = get_engine("nufast")

    assert isinstance(engine, NuFastEngine)
    assert engine.metadata.availability == "available"
    assert engine.metadata.capabilities.neutrino
    assert engine.metadata.capabilities.antineutrino
    assert not engine.metadata.capabilities.nsi


def test_nufast_rejects_nonzero_nsi():
    engine = get_engine("nufast")
    energy_gev = np.linspace(0.5, 5.0, 9)

    with pytest.raises(ValueError, match="standard PMNS only"):
        engine.probabilities(
            energy_gev,
            810.0,
            2.79,
            PMNSParams(),
            NSIParams(eps_mutau=0.05),
        )


def test_nufast_rows_are_unitary():
    engine = get_engine("nufast")
    energy_gev = np.linspace(0.3, 10.0, 150)

    probs = engine.probabilities(
        energy_gev, 1300.0, 2.848, PMNSParams(), NSIParams()
    )

    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(probs.sum(axis=2), 1.0, atol=1e-9)
