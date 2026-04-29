"""Three-flavour PMNS + NSI oscillation engine.

Units: L in km, E in GeV, masses in eV², densities in g/cm³.
All constants from PDG 2024 via OscLib/Constants.h.
"""

import numpy as np
from dataclasses import dataclass, field

# ── Physical constants (matching OscLib/Constants.h exactly) ─────────────────
_GEV_TO_EV      = 1e9
_KM_TO_M        = 1e3
_G_F            = 1.1663788e-5        # Fermi constant, GeV^-2
_N_A            = 6.02214076e23       # Avogadro, mol^-1
_C_LIGHT        = 299792458.0         # m/s
_HBAR           = 6.582119569e-25     # GeV·s
_HBAR_C_EV_M    = _C_LIGHT * _HBAR * _GEV_TO_EV   # ℏc in eV·m  ≈ 1.97327e-7
_HBAR_C_EV_CM   = _HBAR_C_EV_M * 100              # ℏc in eV·cm
_Y_E            = 0.5                 # electrons per nucleon (standard rock)

# Matter density conversion:  V_CC [eV] = _KMATTER × Ne [mol/cm³]
# = √2 · G_F[eV⁻²] · N_A · (ℏc[eV·cm])³
_KMATTER = (
    np.sqrt(2) * (_G_F / _GEV_TO_EV**2)
    * _N_A
    * _HBAR_C_EV_CM**3
)  # ≈ 7.65e-14  eV·cm³/mol


# ── Parameter containers ──────────────────────────────────────────────────────

@dataclass
class PMNSParams:
    """Standard 3-flavour PMNS parameters.

    Mixing angles in radians, mass-squared differences in eV².
    PDG-2024 normal-ordering best-fit values as defaults.
    """
    th12:    float = np.radians(33.44)
    th13:    float = np.radians(8.57)
    th23:    float = np.radians(49.2)
    dm21:    float = 7.42e-5    # Δm²₂₁  eV²
    dm31:    float = 2.515e-3   # Δm²₃₁  eV²  (NH)
    delta_cp: float = -1.601    # δ_CP  radians


@dataclass
class NSIParams:
    """Off-diagonal NSI ε parameters (v1: diagonals fixed to zero)."""
    eps_emu:    float = 0.0   # |ε_eμ|
    eps_etau:   float = 0.0   # |ε_eτ|
    eps_mutau:  float = 0.0   # |ε_μτ|
    delta_emu:   float = 0.0  # phase δ_eμ  (radians)
    delta_etau:  float = 0.0  # phase δ_eτ
    delta_mutau: float = 0.0  # phase δ_μτ


# ── PMNS matrix ───────────────────────────────────────────────────────────────

def pmns_matrix(p: PMNSParams) -> np.ndarray:
    """Return 3×3 complex PMNS matrix U[alpha, i].

    Row = flavor (e, μ, τ), column = mass eigenstate (1, 2, 3).
    Matches OscLib/PMNSOpt.cxx:319-330 column layout.
    """
    s12, c12 = np.sin(p.th12), np.cos(p.th12)
    s13, c13 = np.sin(p.th13), np.cos(p.th13)
    s23, c23 = np.sin(p.th23), np.cos(p.th23)
    eid = np.exp(1j * p.delta_cp)

    U = np.array([
        [c12*c13,                          s12*c13,                          s13*np.conj(eid)   ],
        [-s12*c23 - c12*s23*s13*eid,       c12*c23 - s12*s23*s13*eid,       s23*c13            ],
        [ s12*s23 - c12*c23*s13*eid,      -c12*s23 - s12*c23*s13*eid,       c23*c13            ],
    ], dtype=complex)
    return U


# ── NSI matter potential ──────────────────────────────────────────────────────

def _nsi_potential(rho: float, nsi: NSIParams, antineutrino: bool) -> np.ndarray:
    """Return the full 3×3 matter+NSI Hamiltonian contribution in eV.

    Mirrors OscLib/PMNS_NSI.cxx:88-114.  For antineutrinos the off-diagonal
    epsilon terms are conjugated and the whole matrix is negated.
    """
    Ne = rho * _Y_E                # mol/cm³
    V0 = _KMATTER * Ne             # eV

    e_emu   = nsi.eps_emu   * np.exp(1j * nsi.delta_emu)
    e_etau  = nsi.eps_etau  * np.exp(1j * nsi.delta_etau)
    e_mutau = nsi.eps_mutau * np.exp(1j * nsi.delta_mutau)

    if antineutrino:
        # Upper-triangle off-diagonals: use conj(epsilon); overall sign −1
        V = -V0 * np.array([
            [1.0,                np.conj(e_emu),  np.conj(e_etau)  ],
            [e_emu,              0.0,             np.conj(e_mutau) ],
            [e_etau,             e_mutau,         0.0              ],
        ], dtype=complex)
    else:
        V = +V0 * np.array([
            [1.0,                e_emu,           e_etau           ],
            [np.conj(e_emu),     0.0,             e_mutau          ],
            [np.conj(e_etau),    np.conj(e_mutau), 0.0             ],
        ], dtype=complex)

    return V


# ── Total Hamiltonian ─────────────────────────────────────────────────────────

def _build_hamiltonian(
    E_GeV: np.ndarray,
    rho: float,
    pmns: PMNSParams,
    nsi: NSIParams,
    antineutrino: bool,
) -> np.ndarray:
    """Return H[n,3,3] in eV, vectorised over E_GeV (shape N)."""
    E_eV = E_GeV * _GEV_TO_EV          # (N,)
    U = pmns_matrix(pmns)               # (3,3)

    # Vacuum eigenvalues for each E: (N,3)
    d = np.stack([
        np.zeros_like(E_eV),
        pmns.dm21 / (2.0 * E_eV),
        pmns.dm31 / (2.0 * E_eV),
    ], axis=-1)

    if antineutrino:
        # H_vac_anti = U* @ diag @ Uᵀ  (= conj of H_vac_nu)
        H_vac = np.einsum('ai,ni,bi->nab', U.conj(), d, U, optimize=True)
    else:
        H_vac = np.einsum('ai,ni,bi->nab', U, d, U.conj(), optimize=True)

    V = _nsi_potential(rho, nsi, antineutrino)   # (3,3)
    return H_vac + V[np.newaxis, :, :]            # (N,3,3)


# ── Propagator ───────────────────────────────────────────────────────────────

def oscillation_probabilities(
    E_GeV: np.ndarray,
    L_km: float,
    rho_gcc: float,
    pmns: PMNSParams,
    nsi: NSIParams,
    antineutrino: bool = False,
) -> np.ndarray:
    """Return P[n, beta, alpha] = P(ν_alpha → ν_beta) for each energy.

    Parameters
    ----------
    E_GeV     : energy array, shape (N,), in GeV
    L_km      : baseline in km
    rho_gcc   : matter density in g/cm³
    pmns      : PMNS mixing parameters
    nsi       : NSI epsilon parameters (off-diagonal only in v1)
    antineutrino : if True, compute antineutrino probabilities

    Returns
    -------
    P : ndarray, shape (N, 3, 3), real
        P[n, beta, alpha] is the probability ν_alpha → ν_beta at E_GeV[n].
        Flavor ordering: 0=e, 1=μ, 2=τ.
    """
    E_GeV = np.asarray(E_GeV, dtype=float)
    H = _build_hamiltonian(E_GeV, rho_gcc, pmns, nsi, antineutrino)  # (N,3,3)

    # Diagonalise: eigenvalues (N,3) in eV, eigenvectors (N,3,3)
    eigvals, eigvecs = np.linalg.eigh(H)   # eigh guarantees real eigenvalues

    # Phase: exp(-i λ L / ℏc)
    L_m = L_km * _KM_TO_M
    phases = np.exp(-1j * eigvals * L_m / _HBAR_C_EV_M)   # (N,3)

    # Propagator: U_evol = V @ diag(phase) @ V†
    # Efficient: (V * phase[:,np.newaxis,:]) @ Vh
    Vh = np.conj(eigvecs).swapaxes(-1, -2)                  # (N,3,3)
    U_evol = np.einsum('nai,ni,nib->nab', eigvecs, phases, Vh, optimize=True)

    return np.abs(U_evol)**2
