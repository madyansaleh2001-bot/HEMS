from pathlib import Path

import pytest

from pvforecast.config import (
    ConfigError,
    Installation,
    PanelSpec,
    felicity_ivem4024_ii,
    load_config,
    percent_per_c_to_per_c,
    resolve_panel,
    validate_refresh_minutes,
)

from conftest import make_installation

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example_installation.json"


@pytest.mark.parametrize("minutes", [5, 10, 15, 20, 60, 120])
def test_refresh_accepts_positive_multiples_of_five(minutes):
    assert validate_refresh_minutes(minutes) == minutes


@pytest.mark.parametrize("minutes", [0, -5, 7, 12, 2.5, "15", True])
def test_refresh_rejects_everything_else(minutes):
    with pytest.raises(ConfigError):
        validate_refresh_minutes(minutes)


def test_installation_rejects_bad_refresh():
    inst = make_installation()
    with pytest.raises(ConfigError):
        Installation(**{**inst.__dict__, "refresh_minutes": 7})


def test_temperature_coefficient_conversion():
    assert percent_per_c_to_per_c(-0.30) == pytest.approx(-0.0030)


def test_temperature_coefficient_in_percent_units_is_rejected():
    with pytest.raises(ConfigError):
        resolve_panel(PanelSpec(source="manual", rated_power_w=400, gamma_pmp_per_c=-0.30))


def test_missing_coefficient_is_an_explicit_assumption():
    resolved = resolve_panel(PanelSpec(source="manual", rated_power_w=400))
    assert any(n.kind == "assumption" and "temperature coefficient" in n.text for n in resolved.notes)


def test_rated_power_is_not_multiplied_by_efficiency():
    resolved = resolve_panel(PanelSpec(source="manual", rated_power_w=500, efficiency=0.22, length_m=2.0, width_m=1.134))
    assert resolved.rated_power_w == 500
    inst = make_installation(panel_count=6, rated_w=500, efficiency=0.22)
    assert inst.stc_capacity_w == 3000


def test_rated_power_derived_from_area_and_efficiency_is_labelled():
    resolved = resolve_panel(PanelSpec(source="manual", efficiency=0.20, length_m=2.0, width_m=1.0))
    assert resolved.rated_power_w == pytest.approx(400.0)
    assert any(n.kind == "derived" for n in resolved.notes)


def test_inconsistent_efficiency_is_flagged():
    resolved = resolve_panel(PanelSpec(source="manual", rated_power_w=500, efficiency=0.15, length_m=2.0, width_m=1.134))
    assert any(n.kind == "warning" for n in resolved.notes)


def test_missing_rating_and_area_is_an_error():
    with pytest.raises(ConfigError):
        resolve_panel(PanelSpec(source="manual", efficiency=0.2))


def test_felicity_preset_uses_confirmed_ratings():
    inv = felicity_ivem4024_ii()
    assert inv.model == "IVEM4024-II"
    assert inv.ac_rated_w == 4000.0
    assert inv.battery_nominal_v == 24.0
    assert inv.pv_input_limit_w is None  # not applied until validated for the unit


def test_example_config_loads_and_resolves_timezone():
    place, inst = load_config(EXAMPLE)
    assert place.timezone == "America/Denver"
    assert inst.inverter.ac_rated_w == 4000.0
    assert inst.sub_arrays[0].panel.gamma_pmp_per_c == pytest.approx(-0.0030)
    assert inst.losses.user_set == frozenset()
