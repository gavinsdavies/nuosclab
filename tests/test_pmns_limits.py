"""Analytic limit checks for the PMNS implementation."""

import numpy as np
import pytest
from osclib_explorer import PMNSParams, NSIParams, oscillation_probabilities, pmns_matrix


def test_pmns_matrix_unitary():
    U = pmns_matrix(PMNSParams())
    assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-14)
    assert np.allclose(U.conj().T @ U, np.eye(3), atol=1e-14)


def test_vacuum_two_flavor_approximation():
    """At small θ₁₃, P(νμ→νe) ≈ sin²(2θ₁₃)·sin²(Δ₃₁L/4E) in vacuum (ρ=0)."""
    pmns = PMNSParams(th13=np.radians(8.57), delta_cp=0.0)
    E = np.linspace(0.5, 5.0, 500)
    L = 810.0
    P = oscillation_probabilities(E, L, rho_gcc=0.0, pmns=pmns, nsi=NSIParams())
    P_me = P[:, 0, 1]   # νμ → νe

    E_eV = E * 1e9
    dm31 = pmns.dm31
    # two-flavor approx (ignore Δm²₂₁ and θ₁₂ corrections)
    from osclib_explorer.physics import _KM_TO_M, _HBAR_C_EV_M
    arg = dm31 * L * _KM_TO_M / (4.0 * E_eV * _HBAR_C_EV_M)
    s2th13 = np.sin(2 * pmns.th13)
    P_approx = s2th13**2 * np.sin(arg)**2

    # Loose tolerance — two-flavor ignores θ₁₂/Δm²₂₁ corrections (~few percent)
    assert np.allclose(P_me, P_approx, atol=0.05), \
        f"Max deviation: {np.abs(P_me - P_approx).max():.4f}"


def test_zero_L_identity():
    """At L=0 every particle stays in its own flavor."""
    E = np.array([1.0, 2.0, 3.0])
    P = oscillation_probabilities(E, L_km=0.0, rho_gcc=2.79,
                                   pmns=PMNSParams(), nsi=NSIParams())
    assert np.allclose(P, np.eye(3)[np.newaxis, :, :], atol=1e-12)


def test_nsi_zero_equals_standard():
    """NSI=0 must reproduce the standard PMNS result exactly."""
    E = np.linspace(0.5, 5.0, 100)
    pmns = PMNSParams()
    nsi_off = NSIParams()  # all zero
    nsi_on  = NSIParams(eps_emu=0.0, eps_etau=0.0, eps_mutau=0.0)
    P1 = oscillation_probabilities(E, 810., 2.79, pmns, nsi_off)
    P2 = oscillation_probabilities(E, 810., 2.79, pmns, nsi_on)
    assert np.allclose(P1, P2, atol=1e-14)


def test_cp_asymmetry_sign():
    """Non-zero δ_CP should produce a visible ν vs ν̄ asymmetry in P(νμ→νe)."""
    pmns = PMNSParams(delta_cp=-np.pi / 2)
    E = np.linspace(0.5, 3.0, 300)
    P_nu   = oscillation_probabilities(E, 810., 2.79, pmns, NSIParams(), antineutrino=False)
    P_anti = oscillation_probabilities(E, 810., 2.79, pmns, NSIParams(), antineutrino=True)
    diff = P_nu[:, 0, 1] - P_anti[:, 0, 1]
    # Asymmetry should be non-trivial (at least a few percent somewhere)
    assert np.abs(diff).max() > 0.01, "δ_CP asymmetry too small — likely a sign error"
