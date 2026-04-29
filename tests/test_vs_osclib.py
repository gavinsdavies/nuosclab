"""Validate against OscLib reference values.

Run tools/osclib_oracle.cc on a machine with OscLib+ROOT available, redirect
stdout to tests/test_vs_osclib.csv, then commit.  This test skips gracefully
if the CSV is absent.
"""

import numpy as np
import pytest
from pathlib import Path
from osclib_explorer import PMNSParams, NSIParams, oscillation_probabilities, NOVA

CSV = Path(__file__).parent / "test_vs_osclib.csv"

# Parameters matching osclib_oracle.cc
_PMNS = PMNSParams(
    th12 = np.arcsin(np.sqrt(0.307)),
    th13 = np.arcsin(np.sqrt(0.0220)),
    th23 = np.arcsin(np.sqrt(0.545)),
    dm21 = 7.42e-5,
    dm31 = 2.515e-3,
    delta_cp = -1.601,
)

_NSI_MAP = {
    "standard":               NSIParams(),
    "eps_emu_0.1_phase_0":    NSIParams(eps_emu=0.1),
    "eps_emu_0.1_phase_-pi2": NSIParams(eps_emu=0.1, delta_emu=-np.pi/2),
    "eps_etau_0.1_phase_0":   NSIParams(eps_etau=0.1),
    "eps_mutau_0.1_phase_0":  NSIParams(eps_mutau=0.1),
}


@pytest.mark.skipif(not CSV.exists(), reason="oracle CSV not generated yet")
def test_vs_osclib():
    rows = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
    tol = 1e-4

    for row in rows:
        scenario = str(row["scenario"])
        E = float(row["E_GeV"])
        Pme_ref = float(row["Pme"])
        Pmm_ref = float(row["Pmm"])

        nsi = _NSI_MAP[scenario]
        P = oscillation_probabilities(
            np.array([E]), NOVA.L_km, NOVA.rho_gcc, _PMNS, nsi
        )
        Pme = float(P[0, 0, 1])   # beta=e, alpha=mu
        Pmm = float(P[0, 1, 1])   # beta=mu, alpha=mu

        assert abs(Pme - Pme_ref) < tol, (
            f"Pme mismatch at E={E}, scenario={scenario}: "
            f"got {Pme:.6f}, expected {Pme_ref:.6f}"
        )
        assert abs(Pmm - Pmm_ref) < tol, (
            f"Pmm mismatch at E={E}, scenario={scenario}: "
            f"got {Pmm:.6f}, expected {Pmm_ref:.6f}"
        )
