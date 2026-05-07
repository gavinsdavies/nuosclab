"""Optional agreement tests for the nuprobe engine adapter."""

import numpy as np
import pytest

from nuosclab import (
    NSIParams,
    PMNSParams,
    get_engine,
    oscillation_probabilities,
)

pytest.importorskip("nuprobe")


def test_nuprobe_engine_matches_numpy_ref_for_standard_pmns():
    engine = get_engine("nuprobe")
    energy_gev = np.linspace(0.5, 5.0, 8)
    pmns = PMNSParams(delta_cp=0.0)
    nsi = NSIParams()

    via_nuprobe = engine.probabilities(energy_gev, 810.0, 2.79, pmns, nsi)
    via_numpy = oscillation_probabilities(energy_gev, 810.0, 2.79, pmns, nsi)

    assert np.allclose(via_nuprobe, via_numpy, atol=1e-12)


@pytest.mark.parametrize("antineutrino", [False, True])
def test_nuprobe_engine_matches_numpy_ref_for_nsi(antineutrino):
    engine = get_engine("nuprobe")
    energy_gev = np.linspace(0.7, 4.0, 6)
    pmns = PMNSParams(delta_cp=-np.pi / 2)
    nsi = NSIParams(
        eps_emu=0.03,
        delta_emu=0.4,
        eps_etau=0.02,
        delta_etau=-0.7,
        eps_mutau=0.01,
        delta_mutau=1.1,
    )

    via_nuprobe = engine.probabilities(
        energy_gev,
        810.0,
        2.84,
        pmns,
        nsi,
        antineutrino=antineutrino,
    )
    via_numpy = oscillation_probabilities(
        energy_gev,
        810.0,
        2.84,
        pmns,
        nsi,
        antineutrino=antineutrino,
    )

    assert np.allclose(via_nuprobe, via_numpy, atol=1e-12)
