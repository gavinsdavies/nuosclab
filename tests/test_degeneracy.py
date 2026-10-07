"""Tests for the NSI ε_eτ degeneracy map."""

import numpy as np
import pytest

from nuosclab import (
    EtauDegeneracyConfig,
    NSIParams,
    PMNSParams,
    PRESETS,
    compute_etau_degeneracy,
    oscillation_probabilities,
)
from nuosclab.degeneracy import _engine_grid, _numpy_grid
from nuosclab.engines import get_engine


def _p_mue(energy, preset, pmns, nsi, anti):
    return oscillation_probabilities(
        np.array([energy]), preset.L_km, preset.rho_gcc, pmns, nsi, anti
    )[0, 0, 1]


@pytest.mark.parametrize("anti", [False, True])
def test_batched_grid_matches_engine_loop(anti):
    preset = PRESETS["DUNE"]
    pmns = PMNSParams(delta_cp=0.3)
    eps = np.linspace(0.05, 0.4, 4)
    phi = np.linspace(-np.pi, np.pi, 5)

    fast = _numpy_grid(2.2, preset, pmns, eps, phi, anti)
    slow = _engine_grid(get_engine("numpy_ref"), 2.2, preset, pmns, eps, phi, anti)

    assert np.allclose(fast, slow, atol=1e-12)


def test_map_values_match_direct_probabilities():
    pmns = PMNSParams(delta_cp=-np.pi / 2, th23=np.arcsin(np.sqrt(0.45)), dm31=2.45e-3)
    config = EtauDegeneracyConfig(
        experiment="NOvA", energy_gev=1.6, pmns=pmns, delta_cp_nsi=0.4,
        n_eps=6, n_phase=7,
    )
    dmap = compute_etau_degeneracy(config)
    preset = PRESETS["NOvA"]

    i, j = 3, 4
    nsi = NSIParams(
        eps_etau=dmap.eps[j], delta_etau=dmap.phase_sum[i] - config.delta_cp_nsi
    )
    pmns_nsi = PMNSParams(th23=pmns.th23, dm31=pmns.dm31, delta_cp=0.4)
    for anti, delta, p_si in (
        (False, dmap.delta_nu, dmap.p_si_nu),
        (True, dmap.delta_nubar, dmap.p_si_nubar),
    ):
        assert p_si == pytest.approx(_p_mue(1.6, preset, pmns, NSIParams(), anti))
        expected = _p_mue(1.6, preset, pmns_nsi, nsi, anti) - p_si
        assert delta[i, j] == pytest.approx(expected, abs=1e-12)


def test_grid_shapes_and_defaults():
    dmap = compute_etau_degeneracy(EtauDegeneracyConfig(n_eps=10, n_phase=13))

    assert dmap.energy_gev == PRESETS["NOvA"].E_peak
    assert dmap.eps.shape == (10,)
    assert dmap.eps[0] > 0 and dmap.eps[-1] == pytest.approx(0.5)
    assert dmap.phase_sum[0] == 0 and dmap.phase_sum[-1] == pytest.approx(2 * np.pi)
    assert dmap.delta_nu.shape == dmap.delta_nubar.shape == (13, 10)


def test_joint_cells_are_degenerate_for_both_beams():
    # Standard δ_CP = 0.8π vs an NSI hypothesis at δ_CP = −π/2: the ν and ν̄
    # curves cross near |ε_eτ| ≈ 0.37 for NOvA at its peak energy.
    pmns = PMNSParams(delta_cp=0.8 * np.pi)
    config = EtauDegeneracyConfig(pmns=pmns, delta_cp_nsi=-np.pi / 2, n_eps=200,
                                  n_phase=241)
    dmap = compute_etau_degeneracy(config)
    cells = dmap.joint_cells()
    assert len(cells) > 0

    preset = PRESETS["NOvA"]
    pmns_nsi = PMNSParams(delta_cp=-np.pi / 2)
    for eps, phase_sum in cells:
        nsi = NSIParams(eps_etau=eps, delta_etau=phase_sum + np.pi / 2)
        for anti, p_si in ((False, dmap.p_si_nu), (True, dmap.p_si_nubar)):
            p_nsi = _p_mue(dmap.energy_gev, preset, pmns_nsi, nsi, anti)
            assert abs(p_nsi - p_si) < 2e-3


def test_same_delta_cp_gives_no_crossing():
    # With δ_CP^NSI = δ_CP^SM, ΔP is O(|ε|) with near-horizontal zero lines
    # at different δ_CP + φ_eτ for ν and ν̄, so they never cross.
    pmns = PMNSParams(delta_cp=-np.pi / 2)
    dmap = compute_etau_degeneracy(
        EtauDegeneracyConfig(pmns=pmns, delta_cp_nsi=-np.pi / 2)
    )
    assert len(dmap.joint_cells()) == 0


def test_rejects_invalid_config():
    with pytest.raises(ValueError, match="Unknown experiment"):
        compute_etau_degeneracy(EtauDegeneracyConfig(experiment="MINOS"))
    with pytest.raises(ValueError, match="does not support NSI"):
        compute_etau_degeneracy(EtauDegeneracyConfig(engine="nufast"))
    with pytest.raises(ValueError, match="energy_gev"):
        compute_etau_degeneracy(EtauDegeneracyConfig(energy_gev=0.0))
