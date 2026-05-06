"""Tests for the reusable explorer computation layer."""

import numpy as np
import pytest

from osclib_explorer import (
    ExplorerConfig,
    NSIParams,
    PMNSParams,
    compute_curves,
    oscillation_probabilities,
)


def test_compute_curves_matches_physics_engine():
    config = ExplorerConfig(
        experiment="DUNE",
        pmns=PMNSParams(delta_cp=-np.pi / 2),
        nsi=NSIParams(eps_emu=0.08, delta_emu=0.4),
        n_points=25,
    )

    curves = compute_curves(config)
    expected = oscillation_probabilities(
        curves.energy_gev,
        curves.preset.L_km,
        curves.preset.rho_gcc,
        config.pmns,
        config.nsi,
        config.antineutrino,
    )

    assert curves.preset.name == "DUNE"
    assert curves.energy_gev.shape == (25,)
    assert curves.live.shape == (25, 3, 3)
    assert np.allclose(curves.live, expected)


def test_standard_and_nominal_reference_definitions():
    config = ExplorerConfig(
        pmns=PMNSParams(th23=np.radians(45.0)),
        nsi=NSIParams(eps_mutau=0.1),
        n_points=10,
    )

    curves = compute_curves(config)
    standard_expected = oscillation_probabilities(
        curves.energy_gev,
        curves.preset.L_km,
        curves.preset.rho_gcc,
        config.pmns,
        NSIParams(),
    )
    nominal_expected = oscillation_probabilities(
        curves.energy_gev,
        curves.preset.L_km,
        curves.preset.rho_gcc,
        PMNSParams(),
        NSIParams(),
    )

    assert np.allclose(curves.standard, standard_expected)
    assert np.allclose(curves.nominal, nominal_expected)


def test_compute_curves_can_skip_reference_curves():
    curves = compute_curves(
        ExplorerConfig(include_standard=False, include_nominal=False)
    )

    assert curves.standard is None
    assert curves.nominal is None


def test_compute_curves_rejects_invalid_config():
    with pytest.raises(ValueError, match="Unknown experiment"):
        compute_curves(ExplorerConfig(experiment="MINOS"))

    with pytest.raises(ValueError, match="n_points"):
        compute_curves(ExplorerConfig(n_points=1))


def test_curves_as_dict_is_json_friendly():
    curves = compute_curves(ExplorerConfig(n_points=4))
    payload = curves.as_dict()

    assert payload["config"]["experiment"] == "NOvA"
    assert payload["preset"]["name"] == "NOvA"
    assert isinstance(payload["energy_gev"], list)
    assert isinstance(payload["live"], list)
    assert isinstance(payload["standard"], list)
    assert isinstance(payload["nominal"], list)
