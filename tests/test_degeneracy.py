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
    assert dmap.eps[0] > 0 and dmap.eps[-1] == pytest.approx(3.0)
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


def test_same_delta_cp_crosses_only_at_large_eps():
    # With δ_CP^NSI = δ_CP^SM, ΔP is O(|ε|) at small |ε|, with near-straight
    # zero lines at different δ_CP + φ_eτ for ν and ν̄: no crossing below
    # |ε| = 0.5. The O(|ε|²) terms bend them into a crossing near |ε| ~ 1.5.
    pmns = PMNSParams(delta_cp=-np.pi / 2)
    small = compute_etau_degeneracy(
        EtauDegeneracyConfig(pmns=pmns, delta_cp_nsi=-np.pi / 2, eps_max=0.5)
    )
    assert len(small.joint_cells()) == 0

    full = compute_etau_degeneracy(
        EtauDegeneracyConfig(pmns=pmns, delta_cp_nsi=-np.pi / 2)
    )
    assert len(full.joint_cells()) > 0
    assert full.joint_cells()[:, 0].min() > 1.0


def test_rejects_invalid_config():
    with pytest.raises(ValueError, match="Unknown experiment"):
        compute_etau_degeneracy(EtauDegeneracyConfig(experiment="MINOS"))
    with pytest.raises(ValueError, match="does not support NSI"):
        compute_etau_degeneracy(EtauDegeneracyConfig(engine="nufast"))
    with pytest.raises(ValueError, match="energy_gev"):
        compute_etau_degeneracy(EtauDegeneracyConfig(energy_gev=0.0))


NOVA_SO = PMNSParams(
    th23=np.arcsin(np.sqrt(0.57)), dm31=2.41e-3 + 7.42e-5, delta_cp=0.82 * np.pi
)


def test_profile_finds_nova_island_near_three_pi_over_two():
    # NOvA NSI Letter (arXiv:2403.07266): at the SO best fit, an isolated
    # region at large |eps_etau| near delta_CP + phi_etau = 3pi/2 reproduces
    # both nu and nubar appearance. At a single energy (1.9 GeV) it sits at
    # |eps_etau| ~ 1.6-1.7; the paper's full-spectrum fit puts it near 1.8.
    from nuosclab import compute_etau_profile

    pmap = compute_etau_profile(
        EtauDegeneracyConfig(pmns=NOVA_SO), n_eps=150, n_phase=121, n_delta_cp=48
    )
    large = pmap.eps > 1.0
    sub = pmap.mismatch[:, large]
    i, j = np.unravel_index(sub.argmin(), sub.shape)

    assert sub[i, j] < 0.01
    assert pmap.phase_sum[i] / np.pi == pytest.approx(1.5, abs=0.06)
    assert 1.4 < pmap.eps[large][j] < 1.9


def test_profile_best_delta_cp_reproduces_mismatch():
    from nuosclab import compute_etau_profile

    pmap = compute_etau_profile(
        EtauDegeneracyConfig(pmns=NOVA_SO), n_eps=20, n_phase=13, n_delta_cp=36
    )
    preset = PRESETS["NOvA"]
    i, j = 9, 12
    dcp = pmap.best_delta_cp_nsi[i, j]
    nsi = NSIParams(eps_etau=pmap.eps[j], delta_etau=pmap.phase_sum[i] - dcp)
    pmns_nsi = PMNSParams(th23=NOVA_SO.th23, dm31=NOVA_SO.dm31, delta_cp=dcp)
    terms = []
    for anti in (False, True):
        p_si = _p_mue(pmap.energy_gev, preset, NOVA_SO, NSIParams(), anti)
        p_nsi = _p_mue(pmap.energy_gev, preset, pmns_nsi, nsi, anti)
        terms.append(((p_nsi - p_si) / p_si) ** 2)
    direct = np.sqrt(0.5 * sum(terms))

    # Parabolic refinement: the reported minimum is close to, and not far
    # above, the value at the reported delta CP.
    assert pmap.mismatch[i, j] == pytest.approx(direct, rel=0.05, abs=1e-3)


def test_profile_rejects_other_engines():
    from nuosclab import compute_etau_profile

    with pytest.raises(ValueError, match="numpy_ref"):
        compute_etau_profile(EtauDegeneracyConfig(engine="nuprobe"))


def test_closed_form_p_mue_matches_eigh_propagator():
    from nuosclab.degeneracy import _p_mue
    from nuosclab.physics import _propagate

    rng = np.random.default_rng(7)
    H = (rng.normal(size=(500, 3, 3)) + 1j * rng.normal(size=(500, 3, 3))) * 1e-13
    H = H + np.conj(H.swapaxes(-1, -2))
    assert np.allclose(_p_mue(H, 810.0), _propagate(H, 810.0)[:, 0, 1], atol=1e-12)


def test_energy_average_rate_keeps_island_spectrum_removes_it():
    # With a Gaussian "flux" (σ = 0.4 GeV around 1.9 GeV), matching only the
    # flux-averaged rates keeps the NOvA island; demanding the unsmeared
    # spectrum match point by point removes it at the 10% level.
    from nuosclab import compute_etau_profile

    def island_min(average):
        pmap = compute_etau_profile(EtauDegeneracyConfig(
            pmns=NOVA_SO, energy_sigma=0.4, n_energy=7, average=average,
        ), n_delta_cp=36)
        sel = (pmap.eps > 1.2) & (pmap.eps < 2.0)
        near = np.abs(pmap.phase_sum / np.pi - 1.5) < 0.15
        return pmap.mismatch[np.ix_(near, sel)].min()

    assert island_min("rate") < 0.05
    assert island_min("spectrum") > 0.1


def test_averaged_fixed_map_is_weighted_mean_of_single_energies():
    from nuosclab.degeneracy import _energy_points

    base = dict(pmns=NOVA_SO, delta_cp_nsi=0.9 * np.pi, n_eps=5, n_phase=7)
    avg_cfg = EtauDegeneracyConfig(energy_sigma=0.3, n_energy=5, **base)
    avg = compute_etau_degeneracy(avg_cfg)
    energies, weights = _energy_points(PRESETS["NOvA"].E_peak, avg_cfg)
    singles = [
        compute_etau_degeneracy(EtauDegeneracyConfig(energy_gev=e, **base))
        for e in energies
    ]
    expected = sum(w * m.delta_nu for w, m in zip(weights, singles))
    assert np.allclose(avg.delta_nu, expected, atol=1e-12)
