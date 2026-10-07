"""NSI ε_eτ degeneracy maps for the νμ → νe appearance channel.

For a "true" standard three-flavour point (ε = 0), find where an NSI
hypothesis with ε_eτ = |ε_eτ| e^{iφ_eτ} gives exactly the same appearance
probability, separately for neutrinos and antineutrinos. To leading order
the ε_eτ interference term depends on φ_eτ only through δ_CP + φ_eτ, so the
map is drawn in the (|ε_eτ|, δ_CP + φ_eτ) plane. Where the neutrino and
antineutrino zero contours cross, the NSI hypothesis cannot be told apart
from the standard one by either beam mode at that energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .engines import get_engine
from .physics import (
    NSIParams,
    PMNSParams,
    _build_hamiltonian,
    _nsi_potential,
    _propagate,
)
from .presets import PRESETS, ExperimentPreset

_SRC, _DST = 1, 0  # νμ → νe


@dataclass(frozen=True)
class EtauDegeneracyConfig:
    """Input state for one ε_eτ degeneracy map.

    ``pmns`` is the standard (ε = 0) reference point. The NSI hypothesis uses
    the same PMNS values except δ_CP, which is ``delta_cp_nsi``; its φ_eτ is
    then (δ_CP + φ_eτ) − δ_CP^NSI along the vertical axis. Setting
    ``delta_cp_nsi = pmns.delta_cp`` gives the trivial case where ΔP is
    O(|ε|) and its zeros are near-horizontal lines that never cross between
    ν and ν̄. ``energy_gev`` defaults to the experiment's peak energy.
    """

    experiment: str = "NOvA"
    energy_gev: float | None = None
    pmns: PMNSParams = field(default_factory=PMNSParams)
    delta_cp_nsi: float = 0.0
    eps_max: float = 0.5
    n_eps: int = 100
    n_phase: int = 121
    engine: str = "numpy_ref"


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

    deltas = {}
    p_si = {}
    for anti in (False, True):
        p_si[anti] = engine.probabilities(
            np.array([energy]),
            preset.L_km,
            preset.rho_gcc,
            config.pmns,
            NSIParams(),
            anti,
        )[0, _DST, _SRC]
        if config.engine == "numpy_ref":
            p_nsi = _numpy_grid(energy, preset, pmns_nsi, eps, phi, anti)
        else:
            p_nsi = _engine_grid(engine, energy, preset, pmns_nsi, eps, phi, anti)
        deltas[anti] = p_nsi - p_si[anti]

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


def _numpy_grid(
    energy: float,
    preset: ExperimentPreset,
    pmns: PMNSParams,
    eps: np.ndarray,
    phi: np.ndarray,
    antineutrino: bool,
) -> np.ndarray:
    """Batched numpy_ref evaluation: one eigh call for the whole grid."""
    H0 = _build_hamiltonian(
        np.array([energy]), preset.rho_gcc, pmns, NSIParams(), antineutrino
    )[0]
    # Signed V_CC: +V0 for neutrinos, −V0 for antineutrinos
    v_cc = _nsi_potential(preset.rho_gcc, NSIParams(), antineutrino)[0, 0].real
    e_etau = eps[np.newaxis, :] * np.exp(1j * phi[:, np.newaxis])  # (P, E)
    if antineutrino:
        e_etau = np.conj(e_etau)

    H = np.broadcast_to(H0, e_etau.shape + (3, 3)).copy()
    H[..., 0, 2] += v_cc * e_etau
    H[..., 2, 0] += v_cc * np.conj(e_etau)
    P = _propagate(H.reshape(-1, 3, 3), preset.L_km)
    return P[:, _DST, _SRC].reshape(e_etau.shape)


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
