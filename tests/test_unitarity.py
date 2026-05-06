"""Unitarity: transition probabilities must sum to 1 over all final flavors."""

import numpy as np
import pytest
from nuosclab import PMNSParams, NSIParams, oscillation_probabilities, NOVA, DUNE, T2K

_E = np.linspace(0.4, 5.0, 200)
_DEFAULT_PMNS = PMNSParams()


@pytest.mark.parametrize("preset", [NOVA, DUNE, T2K])
@pytest.mark.parametrize("antineutrino", [False, True])
def test_unitarity_no_nsi(preset, antineutrino):
    P = oscillation_probabilities(_E, preset.L_km, preset.rho_gcc,
                                   _DEFAULT_PMNS, NSIParams(), antineutrino)
    row_sums = P.sum(axis=1)   # sum over beta for each (E, alpha)
    col_sums = P.sum(axis=2)   # sum over alpha for each (E, beta)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f"Row sums off: {row_sums.max()}"
    assert np.allclose(col_sums, 1.0, atol=1e-10), f"Col sums off: {col_sums.max()}"


@pytest.mark.parametrize("antineutrino", [False, True])
def test_unitarity_with_nsi(antineutrino):
    nsi = NSIParams(eps_emu=0.1, eps_etau=0.05, eps_mutau=0.08,
                    delta_emu=1.2, delta_etau=-0.7, delta_mutau=2.1)
    P = oscillation_probabilities(_E, NOVA.L_km, NOVA.rho_gcc,
                                   _DEFAULT_PMNS, nsi, antineutrino)
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-10)
    assert np.allclose(col_sums, 1.0, atol=1e-10)
