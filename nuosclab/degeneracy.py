"""NSI ε_eτ degeneracy maps for the νμ → νe appearance channel.

For a "true" standard three-flavour point (ε = 0), find where an NSI
hypothesis with ε_eτ = |ε_eτ| e^{iφ_eτ} gives exactly the same appearance
probability, separately for neutrinos and antineutrinos. To leading order
the ε_eτ interference term depends on φ_eτ only through δ_CP + φ_eτ, so the
map is drawn in the (δ_CP + φ_eτ, |ε_eτ|) plane. Where the neutrino and
antineutrino zero contours cross, the NSI hypothesis cannot be told apart
from the standard one by either beam mode at that energy.

``compute_etau_degeneracy`` fixes the NSI hypothesis's δ_CP.
``compute_etau_profile`` instead minimises the ν + ν̄ mismatch over it at
each (δ_CP + φ_eτ, |ε_eτ|), the way NOvA profiles over δ_CP − φ_eτ in its
NSI analysis (arXiv:2403.07266). That is what exposes the isolated
large-|ε_eτ| "island" near δ_CP + φ_eτ = 3π/2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .engines import get_engine
from .physics import (
    NSIParams,
    PMNSParams,
    _HBAR_C_EV_M,
    _KM_TO_M,
    _build_hamiltonian,
    _nsi_potential,
    oscillation_probabilities,
)
from .presets import PRESETS, ExperimentPreset

_SRC, _DST = 1, 0  # νμ → νe


@dataclass(frozen=True)
class EtauDegeneracyConfig:
    """Input state for one ε_eτ degeneracy map.

    ``pmns`` is the standard (ε = 0) reference point. The NSI hypothesis uses
    the same PMNS values except δ_CP, which is ``delta_cp_nsi``; its φ_eτ is
    then (δ_CP + φ_eτ) − δ_CP^NSI along the phase axis. With
    ``delta_cp_nsi = pmns.delta_cp``, ΔP is O(|ε|) at small |ε| and the ν
    and ν̄ zero curves are near-straight lines that only meet at large |ε|.
    ``energy_gev`` defaults to the experiment's peak energy.

    ``energy_sigma > 0`` replaces the single energy by ``n_energy`` points
    across ±2σ around it, weighted by a Gaussian. That is a stand-in for a
    beam flux, not a real one. The fixed-δ_CP map then compares
    flux-weighted rates. The profile compares flux-weighted rates when
    ``average == "rate"`` and the spectra point by point when
    ``average == "spectrum"`` (no detector smearing, so a much stricter
    test than a real fit).
    """

    experiment: str = "NOvA"
    energy_gev: float | None = None
    pmns: PMNSParams = field(default_factory=PMNSParams)
    delta_cp_nsi: float = 0.0
    eps_max: float = 3.0
    n_eps: int = 100
    n_phase: int = 121
    engine: str = "numpy_ref"
    energy_sigma: float = 0.0
    n_energy: int = 9
    average: str = "rate"


@dataclass(frozen=True)
class EtauDegeneracyMap:
    """ΔP = P_NSI − P_SI for νμ → νe on the (|ε_eτ|, δ_CP + φ_eτ) grid.

    ``delta_nu`` and ``delta_nubar`` have shape ``(n_phase, n_eps)``, rows
    following ``phase_sum`` and columns following ``eps``. Their zero level
    sets are the degeneracy curves.
    """

    config: EtauDegeneracyConfig
    preset: ExperimentPreset
    energy_gev: float
    eps: np.ndarray
    phase_sum: np.ndarray
    p_si_nu: float
    p_si_nubar: float
    delta_nu: np.ndarray
    delta_nubar: np.ndarray

    def joint_cells(self) -> np.ndarray:
        """Return (|ε_eτ|, δ_CP + φ_eτ) centres of grid cells where both the
        neutrino and antineutrino ΔP change sign.

        These approximate the crossings of the two zero contours, to the
        resolution of the grid. Shape ``(k, 2)``.
        """
        mask = _sign_change_cells(self.delta_nu) & _sign_change_cells(
            self.delta_nubar
        )
        rows, cols = np.nonzero(mask)
        eps_mid = 0.5 * (self.eps[cols] + self.eps[cols + 1])
        phase_mid = 0.5 * (self.phase_sum[rows] + self.phase_sum[rows + 1])
        return np.column_stack([eps_mid, phase_mid])


def compute_etau_degeneracy(config: EtauDegeneracyConfig) -> EtauDegeneracyMap:
    """Compute the ε_eτ degeneracy map for ``config``."""
    if config.experiment not in PRESETS:
        known = ", ".join(PRESETS)
        raise ValueError(
            f"Unknown experiment {config.experiment!r}; expected one of: {known}"
        )
    if config.n_eps < 2 or config.n_phase < 2:
        raise ValueError("n_eps and n_phase must be at least 2")
    if config.eps_max <= 0:
        raise ValueError("eps_max must be positive")
    engine = get_engine(config.engine)
    if not engine.metadata.capabilities.nsi:
        raise ValueError(f"Engine {config.engine!r} does not support NSI")

    preset = PRESETS[config.experiment]
    energy = preset.E_peak if config.energy_gev is None else config.energy_gev
    if energy <= 0:
        raise ValueError("energy_gev must be positive")

    # Skip |ε| = 0, where ΔP vanishes identically along the whole column and
    # would add a spurious zero contour on the axis.
    eps = np.linspace(0.0, config.eps_max, config.n_eps + 1)[1:]
    phase_sum = np.linspace(0.0, 2.0 * np.pi, config.n_phase)
    pmns_nsi = replace(config.pmns, delta_cp=config.delta_cp_nsi)
    phi = phase_sum - config.delta_cp_nsi

    energies, weights = _energy_points(energy, config)

    deltas = {}
    p_si = {}
    for anti in (False, True):
        p_si_k = engine.probabilities(
            energies, preset.L_km, preset.rho_gcc, config.pmns, NSIParams(), anti
        )[:, _DST, _SRC]
        if config.engine == "numpy_ref":
            p_nsi = _numpy_grid(energies, preset, pmns_nsi, eps, phi, anti)
        else:
            p_nsi = np.stack([
                _engine_grid(engine, e, preset, pmns_nsi, eps, phi, anti)
                for e in energies
            ])
        # Flux-weighted means; a single energy has weight 1
        p_si[anti] = weights @ p_si_k
        deltas[anti] = np.tensordot(weights, p_nsi, axes=1) - p_si[anti]

    return EtauDegeneracyMap(
        config=config,
        preset=preset,
        energy_gev=float(energy),
        eps=eps,
        phase_sum=phase_sum,
        p_si_nu=float(p_si[False]),
        p_si_nubar=float(p_si[True]),
        delta_nu=deltas[False],
        delta_nubar=deltas[True],
    )


@dataclass(frozen=True)
class EtauProfileMap:
    """ν + ν̄ mismatch after profiling over the NSI hypothesis's δ_CP.

    ``mismatch`` is the smallest relative RMS difference
    sqrt(½[(ΔP_ν/P_ν)² + (ΔP_ν̄/P_ν̄)²]) over δ_CP^NSI (a flux-weighted
    spectral version of it when averaging over energy), at fixed
    δ_CP + φ_eτ and |ε_eτ|; ``best_delta_cp_nsi`` is where it is reached.
    Both have shape ``(n_phase, n_eps)``. Zero means both beam modes see
    exactly the standard probability.
    """

    config: EtauDegeneracyConfig
    preset: ExperimentPreset
    energy_gev: float
    eps: np.ndarray
    phase_sum: np.ndarray
    mismatch: np.ndarray
    best_delta_cp_nsi: np.ndarray


def compute_etau_profile(
    config: EtauDegeneracyConfig,
    n_eps: int = 100,
    n_phase: int = 61,
    n_delta_cp: int = 24,
) -> EtauProfileMap:
    """Profile the ν + ν̄ mismatch over δ_CP^NSI on a coarse grid.

    ``config.delta_cp_nsi``, ``n_eps`` and ``n_phase`` are ignored in favour
    of the arguments here, which default to a grid small enough for live
    sliders (well under a second). The δ_CP^NSI scan is refined with a parabola
    through the best scan point and its two neighbours. Only ``numpy_ref``
    is supported, since the scan needs batched evaluation.
    """
    if config.engine != "numpy_ref":
        raise ValueError("compute_etau_profile supports the numpy_ref engine only")
    if config.experiment not in PRESETS:
        known = ", ".join(PRESETS)
        raise ValueError(
            f"Unknown experiment {config.experiment!r}; expected one of: {known}"
        )
    if min(n_eps, n_phase) < 2 or n_delta_cp < 3:
        raise ValueError("need n_eps, n_phase >= 2 and n_delta_cp >= 3")

    preset = PRESETS[config.experiment]
    energy = preset.E_peak if config.energy_gev is None else config.energy_gev
    if energy <= 0:
        raise ValueError("energy_gev must be positive")

    eps = np.linspace(0.0, config.eps_max, n_eps + 1)[1:]
    phase_sum = np.linspace(0.0, 2.0 * np.pi, n_phase)
    step = 2.0 * np.pi / n_delta_cp
    delta_cps = -np.pi + step * np.arange(n_delta_cp)

    energies, weights = _energy_points(energy, config)

    if config.average not in ("rate", "spectrum"):
        raise ValueError("average must be 'rate' or 'spectrum'")

    # Per beam, "rate": (ΔR/R)² with R = Σ_k w_k P_k. "spectrum":
    # Σ_k w_k ΔP_k² / P_k  /  Σ_k w_k P_k, a Poisson-like χ² over energy
    # points. Both reduce to (ΔP/P)² at a single energy.
    chi2 = np.zeros((n_delta_cp, n_phase, n_eps))
    for anti in (False, True):
        p_si = oscillation_probabilities(
            energies, preset.L_km, preset.rho_gcc, config.pmns, NSIParams(), anti
        )[:, _DST, _SRC]
        rate_si = weights @ p_si
        for k, dcp in enumerate(delta_cps):
            pmns = replace(config.pmns, delta_cp=dcp)
            p = _numpy_grid(energies, preset, pmns, eps, phase_sum - dcp, anti)
            if config.average == "rate":
                term = ((np.tensordot(weights, p, axes=1) - rate_si) / rate_si) ** 2
            else:
                dp2 = (p - p_si[:, np.newaxis, np.newaxis]) ** 2
                term = np.tensordot(weights / p_si, dp2, axes=1) / rate_si
            chi2[k] += 0.5 * term

    # Parabolic refinement around the best scan point (δ_CP is periodic)
    k0 = np.argmin(chi2, axis=0)
    c0 = np.take_along_axis(chi2, k0[np.newaxis], 0)[0]
    cm = np.take_along_axis(chi2, ((k0 - 1) % n_delta_cp)[np.newaxis], 0)[0]
    cp = np.take_along_axis(chi2, ((k0 + 1) % n_delta_cp)[np.newaxis], 0)[0]
    curv = cm - 2.0 * c0 + cp
    safe = curv > 0
    offset = np.where(safe, 0.5 * (cm - cp) / np.where(safe, curv, 1.0), 0.0)
    cmin = np.where(safe, c0 - 0.25 * (cm - cp) * offset, c0)
    best = delta_cps[k0] + offset * step
    best = (best + np.pi) % (2.0 * np.pi) - np.pi

    return EtauProfileMap(
        config=config,
        preset=preset,
        energy_gev=float(energy),
        eps=eps,
        phase_sum=phase_sum,
        mismatch=np.sqrt(np.clip(cmin, 0.0, None)),
        best_delta_cp_nsi=best,
    )


def _energy_points(
    energy: float, config: EtauDegeneracyConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Energies and normalised weights: one point, or a Gaussian over ±2σ."""
    if config.energy_sigma <= 0:
        return np.array([energy]), np.array([1.0])
    if config.n_energy < 2:
        raise ValueError("n_energy must be at least 2 when averaging")
    offsets = config.energy_sigma * np.linspace(-2.0, 2.0, config.n_energy)
    energies = energy + offsets
    energies = energies[energies > 0.05]
    weights = np.exp(-0.5 * ((energies - energy) / config.energy_sigma) ** 2)
    return energies, weights / weights.sum()


def _numpy_grid(
    energies: float | np.ndarray,
    preset: ExperimentPreset,
    pmns: PMNSParams,
    eps: np.ndarray,
    phi: np.ndarray,
    antineutrino: bool,
) -> np.ndarray:
    """Batched numpy_ref P(νμ → νe) over energies × phases × |ε|.

    Returns shape ``(n_phase, n_eps)`` for a scalar energy, otherwise
    ``(n_energy, n_phase, n_eps)``.
    """
    scalar = np.ndim(energies) == 0
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    H0 = _build_hamiltonian(energies, preset.rho_gcc, pmns, NSIParams(), antineutrino)
    # Signed V_CC: +V0 for neutrinos, −V0 for antineutrinos
    v_cc = _nsi_potential(preset.rho_gcc, NSIParams(), antineutrino)[0, 0].real
    e_etau = eps[np.newaxis, :] * np.exp(1j * phi[:, np.newaxis])  # (P, E)
    if antineutrino:
        e_etau = np.conj(e_etau)

    shape = (energies.size,) + e_etau.shape
    H = np.broadcast_to(H0[:, np.newaxis, np.newaxis], shape + (3, 3)).copy()
    H[..., 0, 2] += v_cc * e_etau
    H[..., 2, 0] += v_cc * np.conj(e_etau)
    P = _p_mue(H, preset.L_km)
    return P[0] if scalar else P


def _p_mue(H: np.ndarray, L_km: float) -> np.ndarray:
    """|exp(−iHL)[e, μ]|² for a stack of Hermitian 3×3 H (eV), without eigh.

    Closed-form (trigonometric) eigenvalues plus Sylvester's formula
    exp(−iK) = Σ_i e^{−iλ_i} (K − λ_j)(K − λ_k) / ((λ_i − λ_j)(λ_i − λ_k)),
    applied to the traceless, dimensionless K = (H − tr H/3)·L/ħc, forming
    only the [e, μ] element. Much faster than ``np.linalg.eigh`` on many
    small matrices; agrees with ``physics._propagate`` to ~1e-12.
    """
    K = H * (L_km * _KM_TO_M / _HBAR_C_EV_M)
    tr = (K[..., 0, 0] + K[..., 1, 1] + K[..., 2, 2]).real / 3.0
    a = K[..., 0, 0].real - tr
    b = K[..., 1, 1].real - tr
    c = K[..., 2, 2].real - tr
    x01, x02, x12 = K[..., 0, 1], K[..., 0, 2], K[..., 1, 2]
    n01, n02, n12 = abs(x01) ** 2, abs(x02) ** 2, abs(x12) ** 2

    p = np.sqrt((a * a + b * b + c * c + 2.0 * (n01 + n02 + n12)) / 6.0)
    det = (
        a * b * c + 2.0 * (x01 * x12 * np.conj(x02)).real
        - a * n12 - b * n02 - c * n01
    )
    t = np.arccos(np.clip(det / (2.0 * p**3), -1.0, 1.0)) / 3.0
    lam = [2.0 * p * np.cos(t + 2.0 * np.pi * n / 3.0) for n in range(3)]

    # [e, μ] element of K and of K² (the diagonal shift cancels off-diagonal)
    k2_01 = (a + b) * x01 + x02 * np.conj(x12)
    amp = 0.0
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        num = k2_01 - (lam[j] + lam[k]) * x01
        den = (lam[i] - lam[j]) * (lam[i] - lam[k])
        amp = amp + np.exp(-1j * lam[i]) * num / den
    return np.abs(amp) ** 2


def _engine_grid(
    engine,
    energy: float,
    preset: ExperimentPreset,
    pmns: PMNSParams,
    eps: np.ndarray,
    phi: np.ndarray,
    antineutrino: bool,
) -> np.ndarray:
    """Generic (slow) path: one engine call per grid point."""
    out = np.empty((phi.size, eps.size))
    energy_arr = np.array([energy])
    for i, phase in enumerate(phi):
        for j, magnitude in enumerate(eps):
            out[i, j] = engine.probabilities(
                energy_arr,
                preset.L_km,
                preset.rho_gcc,
                pmns,
                NSIParams(eps_etau=magnitude, delta_etau=phase),
                antineutrino,
            )[0, _DST, _SRC]
    return out


def _sign_change_cells(z: np.ndarray) -> np.ndarray:
    """True for each grid cell whose four corners do not share one sign."""
    corners = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]])
    return (corners.min(axis=0) <= 0) & (corners.max(axis=0) >= 0)
