"""Pure-Python port of the NuFast long-baseline oscillation algorithm.

Ported from ``NuFast_LBL.cpp`` by Peter B. Denton and Stephen J. Parke,
NuFast: Fast and accurate neutrino oscillation probabilities in matter
(arXiv:2405.02400), MIT License, https://github.com/PeterDenton/NuFast-LBL.
The original copyright and permission notice is reproduced in
``THIRD_PARTY_LICENSES.md`` at the repository root, per the MIT license.

The port vectorizes the original per-energy loop over a NumPy energy grid
and keeps the upstream conventions: mixing inputs are sin^2 of the angles,
``Dmsq31`` is positive/negative for normal/inverted ordering, and negative
energies select antineutrinos. Standard three-flavour PMNS in constant-density
matter only; no NSI.
"""

from __future__ import annotations

import numpy as np

# Upstream constants, kept bit-identical to NuFast_LBL.cpp so the port can be
# compared digit-for-digit against the reference implementation.
_EVSQKM_TO_GEV_OVER4 = 1e-9 / 1.97327e-7 * 1e3 / 4
_YERHOE2A = 1.52588e-4


def probability_matter_lbl(
    s12sq: float,
    s13sq: float,
    s23sq: float,
    delta: float,
    dmsq21: float,
    dmsq31: float,
    baseline_km: float,
    energy_gev: np.ndarray,
    rho_ye: float,
    n_newton: int = 1,
) -> np.ndarray:
    """Return P[n, alpha, beta] = P(nu_alpha -> nu_beta) over an energy grid.

    Parameters
    ----------
    s12sq, s13sq, s23sq : sin^2 of the PMNS mixing angles
    delta       : Dirac CP phase in radians
    dmsq21      : Delta m^2_21 in eV^2
    dmsq31      : Delta m^2_31 in eV^2; sign selects the mass ordering
    baseline_km : baseline in km
    energy_gev  : energy grid in GeV; negative values select antineutrinos
    rho_ye      : matter density times electron fraction, g/cm^3
    n_newton    : Newton iterations refining lambda3; negative uses the exact
                  (slower) cubic-root expression

    Returns
    -------
    P : ndarray, shape (N, 3, 3), real
        Note the (alpha, beta) index order follows upstream NuFast and is the
        transpose of the ``oscillation_probabilities`` convention.
    """
    energy_gev = np.asarray(energy_gev, dtype=float)

    # Energy-independent functions of the oscillation parameters
    c13sq = 1 - s13sq

    ue2sq = c13sq * s12sq
    ue3sq = s13sq

    um3sq = c13sq * s23sq
    ut2sq = s13sq * s12sq * s23sq
    um2sq = (1 - s12sq) * (1 - s23sq)

    jrr = np.sqrt(um2sq * ut2sq)
    sind = np.sin(delta)
    cosd = np.cos(delta)

    um2sq = um2sq + ut2sq - 2 * jrr * cosd
    jmatter_vac = 8 * jrr * c13sq * sind
    dmsqee = dmsq31 - s12sq * dmsq21

    a_vac = dmsq21 + dmsq31
    see = a_vac - dmsq21 * ue2sq - dmsq31 * ue3sq
    tmm_vac = dmsq21 * dmsq31
    tee = tmm_vac * (1 - ue3sq - ue2sq)

    # Energy-dependent coefficients of the characteristic cubic
    amatter = rho_ye * energy_gev * _YERHOE2A
    big_c = amatter * tee
    big_a = a_vac + amatter
    big_b = tmm_vac + amatter * see

    if n_newton < 0:
        # Exact lambda3 from the cubic root, valid for both mass orderings
        root_asq_b = np.sqrt(big_a * big_a - 3 * big_b)
        ss0 = np.arccos(
            (big_a**3 - 4.5 * big_a * big_b + 13.5 * big_c) / root_asq_b**3
        )
        if dmsq31 < 0:
            ss0 = ss0 + 2 * np.pi
        lambda3 = (big_a + 2 * root_asq_b * np.cos(ss0 / 3)) / 3
    else:
        # lambda+ of MP/DMP as the seed, refined by Newton iterations
        xmat = amatter / dmsqee
        tmp = 1 - xmat
        lambda3 = dmsq31 + 0.5 * dmsqee * (
            xmat - 1 + np.sqrt(tmp * tmp + 4 * s13sq * xmat)
        )
        for _ in range(n_newton):
            lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - big_a) + big_c) / (
                lambda3 * (2 * (lambda3 - big_a) + lambda3) + big_b
            )

    # Eigenvalue differences
    tmp = big_a - lambda3
    dlambda21 = np.sqrt(tmp * tmp - 4 * big_c / lambda3)
    lambda2 = 0.5 * (big_a - lambda3 + dlambda21)
    dlambda32 = lambda3 - lambda2
    dlambda31 = dlambda32 + dlambda21

    # Matter mixing-matrix elements squared via the eigenvector-eigenvalue
    # identity ("Rosetta")
    pi_dlambda_inv = 1 / (dlambda31 * dlambda32 * dlambda21)
    xp3 = pi_dlambda_inv * dlambda21
    xp2 = -pi_dlambda_inv * dlambda31

    ue3sq_m = (lambda3 * (lambda3 - see) + tee) * xp3
    ue2sq_m = (lambda2 * (lambda2 - see) + tee) * xp2

    smm = big_a - dmsq21 * um2sq - dmsq31 * um3sq
    tmm = tmm_vac * (1 - um3sq - um2sq) + amatter * (see + smm - big_a)

    um3sq_m = (lambda3 * (lambda3 - smm) + tmm) * xp3
    um2sq_m = (lambda2 * (lambda2 - smm) + tmm) * xp2

    # Jarlskog factor in matter (NHS identity)
    jmatter = jmatter_vac * dmsq21 * dmsq31 * (dmsq31 - dmsq21) * pi_dlambda_inv

    ue1sq_m = 1 - ue3sq_m - ue2sq_m
    um1sq_m = 1 - um3sq_m - um2sq_m

    ut3sq_m = 1 - um3sq_m - ue3sq_m
    ut2sq_m = 1 - um2sq_m - ue2sq_m
    ut1sq_m = 1 - um1sq_m - ue1sq_m

    # Kinematic terms
    lover4e = _EVSQKM_TO_GEV_OVER4 * baseline_km / energy_gev

    d21 = dlambda21 * lover4e
    d32 = dlambda32 * lover4e

    sin_d21 = np.sin(d21)
    sin_d31 = np.sin(d32 + d21)
    sin_d32 = np.sin(d32)

    triple_sin = sin_d21 * sin_d31 * sin_d32

    sinsq_d21_2 = 2 * sin_d21 * sin_d21
    sinsq_d31_2 = 2 * sin_d31 * sin_d31
    sinsq_d32_2 = 2 * sin_d32 * sin_d32

    # The three independent probabilities, separating the T-conserving and
    # T-violating parts of P(nu_mu -> nu_e)
    pme_tc = (
        (ut3sq_m - um2sq_m * ue1sq_m - um1sq_m * ue2sq_m) * sinsq_d21_2
        + (ut2sq_m - um3sq_m * ue1sq_m - um1sq_m * ue3sq_m) * sinsq_d31_2
        + (ut1sq_m - um3sq_m * ue2sq_m - um2sq_m * ue3sq_m) * sinsq_d32_2
    )
    pme_tv = -jmatter * triple_sin

    pmm = 1 - 2 * (
        um2sq_m * um1sq_m * sinsq_d21_2
        + um3sq_m * um1sq_m * sinsq_d31_2
        + um3sq_m * um2sq_m * sinsq_d32_2
    )

    pee = 1 - 2 * (
        ue2sq_m * ue1sq_m * sinsq_d21_2
        + ue3sq_m * ue1sq_m * sinsq_d31_2
        + ue3sq_m * ue2sq_m * sinsq_d32_2
    )

    # Fill the full matrix from the three probabilities and unitarity
    probs = np.empty((energy_gev.size, 3, 3), dtype=float)
    probs[:, 0, 0] = pee
    probs[:, 0, 1] = pme_tc - pme_tv
    probs[:, 0, 2] = 1 - pee - probs[:, 0, 1]

    probs[:, 1, 0] = pme_tc + pme_tv
    probs[:, 1, 1] = pmm
    probs[:, 1, 2] = 1 - probs[:, 1, 0] - pmm

    probs[:, 2, 0] = 1 - pee - probs[:, 1, 0]
    probs[:, 2, 1] = 1 - probs[:, 0, 1] - pmm
    probs[:, 2, 2] = 1 - probs[:, 0, 2] - probs[:, 1, 2]
    return probs
