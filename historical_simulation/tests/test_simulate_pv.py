"""Tests for simulate_pv.py. All weather here is SYNTHETIC (tests/synthetic_weather.py)."""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import simulate_pv as sp  # noqa: E402
from synthetic_weather import synthetic_frame, write_houses  # noqa: E402

LITERAL_HEADER = (
    "Timestamp,House_ID,Panel_Tilt_rad,Panel_Azimuth_rad,Temperature,Relative Humidity,GHI,DNI,DHI,Wind Speed,"
    "Solar_Zenith_rad,Solar_Azimuth_rad,PV_Power_Generation_W,PV_AC_Power_W,Panel_Count,Panel_STC_Power_W,"
    "Array_STC_Power_W,Inverter_Unit_Count,Inverter_AC_Rated_Power_W,Inverter_Nominal_Efficiency"
)

# The house table from the specification: id, count, Pmax, tilt, gamma, type, units, AC per unit, efficiency.
SPEC_TABLE = [
    (1, 10, 505, 36, -0.0034, "hybrid", 1, 4000, 0.93),
    (2, 12, 540, 36, -0.0035, "string", 1, 5000, 0.97),
    (3, 12, 460, 36, -0.0035, "micro", 12, 400, 0.965),
    (4, 8, 445, 36, -0.0035, "string", 1, 3000, 0.97),
    (5, 9, 425, 36, -0.0035, "hybrid", 1, 3000, 0.93),
    (6, 10, 405, 36, -0.0035, "micro", 10, 350, 0.965),
    (7, 11, 405, 39, -0.0035, "string", 1, 3600, 0.97),
    (8, 11, 445, 38, -0.0035, "hybrid", 1, 4000, 0.93),
    (9, 12, 505, 35, -0.0034, "string", 1, 5000, 0.97),
    (10, 8, 460, 35, -0.0035, "micro", 8, 400, 0.965),
    (11, 12, 540, 32, -0.0035, "hybrid", 1, 5000, 0.93),
    (12, 10, 505, 31, -0.0034, "string", 1, 4000, 0.97),
    (13, 12, 405, 32, -0.0035, "micro", 12, 350, 0.965),
]


@pytest.fixture(scope="module")
def config() -> sp.Config:
    return sp.load_config(ROOT / "simulation_config.json")


@pytest.fixture
def inputs(tmp_path) -> Path:
    write_houses(tmp_path / "in", houses=13, days=2)
    return tmp_path / "in"


def test_header_and_schema_file(config):
    assert ",".join(sp.FINAL_COLUMNS) == LITERAL_HEADER
    schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in schema["columns"]] == sp.FINAL_COLUMNS
    assert schema["header"] == LITERAL_HEADER
    assert set(schema["not_exported"]) == set(sp.REMOVED_COLUMNS)


def test_config_matches_specification_table(config):
    assert sorted(config.houses) == list(range(1, 14))
    for hid, count, pmax, tilt, gamma, kind, units, ac_unit, eta in SPEC_TABLE:
        h = config.houses[hid]
        assert (h.panel_count, h.panel_stc_power_w, h.tilt_deg, h.gamma_pdc_per_c) == (count, pmax, tilt, gamma)
        assert (h.inverter_type, h.inverter_unit_count, h.inverter_ac_power_w_per_unit, h.inverter_nominal_efficiency) == (kind, units, ac_unit, eta)
        assert h.azimuth_deg == 180
        assert h.array_stc_power_w == count * pmax
        assert h.inverter_ac_rated_power_w == units * ac_unit
    p = config.physics
    assert (p.transposition_model, p.temperature_model_parameters, p.dc_loss_fraction, p.inverter_eta_inv_ref) == (
        "haydavies", "open_rack_glass_polymer", 0.10, 0.9637)
    assert 1.0 - p.dc_loss_fraction == 0.9


def test_end_to_end_all_houses(inputs, tmp_path):
    out = tmp_path / "out"
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(out), "--energy-summary", "--zip"]) == 0
    files = sorted((out / "model_datasets").glob("house_*_model_dataset.csv"))
    assert [f.name for f in files] == [f"house_{h:02d}_model_dataset.csv" for h in range(1, 14)]
    for f in files:
        assert f.read_text(encoding="utf-8").split("\n", 1)[0] == LITERAL_HEADER
    report = json.loads((out / "metadata" / "validation_report.json").read_text(encoding="utf-8"))
    assert len(report) == 13
    for checks in report.values():
        assert all(c["passed"] for c in checks.values() if isinstance(c, dict) and "passed" in c)
    meta = json.loads((out / "metadata" / "run_metadata.json").read_text(encoding="utf-8"))
    assert "SIMULATED" in meta["provenance"] and "not measured" in meta["provenance"]
    assert meta["houses"]["3"]["inverter_type"] == "micro"  # architecture kept in metadata only
    assert all(len(f["source_sha256"]) == 64 for f in meta["files"].values())
    names = set(zipfile.ZipFile(out / "pv_model_datasets.zip").namelist())
    assert {"simulate_pv.py", "simulation_config.json", "requirements.txt", "schema.json", "README.md",
            "energy_summary.csv", "metadata/run_metadata.json", "model_datasets/house_13_model_dataset.csv"} <= names


def test_output_preserves_rows_timestamps_and_source_values(inputs, tmp_path):
    out = tmp_path / "out"
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(out), "--houses", "5"]) == 0
    src = pd.read_csv(inputs / "Weather House 5.csv", float_precision="round_trip")
    res = pd.read_csv(out / "model_datasets" / "house_05_model_dataset.csv", float_precision="round_trip")
    assert len(res) == len(src) and res["Timestamp"].iloc[-1] == src["Timestamp"].iloc[-1]
    assert (res["Timestamp"] == src["Timestamp"]).all()
    for s, t in sp.PASSTHROUGH.items():
        assert np.array_equal(res[t].to_numpy(float), src[s].to_numpy(float))
    assert not any(c in res.columns for c in sp.REMOVED_COLUMNS + ("Fill Flag", "Surface Albedo"))
    night = src["Sun Zenith (rad)"] >= math.pi / 2
    assert (res.loc[night, ["PV_Power_Generation_W", "PV_AC_Power_W"]] == 0).all().all()
    assert (res["PV_AC_Power_W"] <= res["PV_Power_Generation_W"]).all()
    assert res["Panel_Azimuth_rad"].eq(math.pi).all() and res["Panel_Tilt_rad"].eq(math.radians(36)).all()


def test_physics_matches_the_specified_sequence(config):
    frame = synthetic_frame(periods=96 * 3 + 1, seed=7)
    values = frame.drop(columns="Timestamp").astype(float)
    times = pd.DatetimeIndex(pd.to_datetime(frame["Timestamp"]))
    house, physics = config.houses[3], config.physics  # microinverter scenario
    sim = sp.simulate(values, times, house, physics)
    zen, az = np.degrees(values["Sun Zenith (rad)"]), np.degrees(values["Sun Azimuth (rad)"])
    irr = pvlib.irradiance.get_total_irradiance(
        house.tilt_deg, 180, zen, az, values["DNI"], values["GHI"], values["DHI"],
        dni_extra=pvlib.irradiance.get_extra_radiation(times).to_numpy(), albedo=values["Surface Albedo"], model="haydavies")
    poa = np.maximum(irr["poa_global"].to_numpy(), 0)
    poa[zen >= 90] = 0
    np.testing.assert_array_equal(sim["poa_global_wm2"], poa)
    tc = pvlib.temperature.sapm_cell(poa, values["Temperature"], values["Wind Speed"], a=-3.56, b=-0.075, deltaT=3)
    dc = np.maximum(pvlib.pvsystem.pvwatts_dc(poa, tc, house.array_stc_power_w, house.gamma_pdc_per_c), 0) * 0.9
    np.testing.assert_allclose(sim["pdc_w"], dc, rtol=0, atol=1e-9)
    ac = 12 * pvlib.inverter.pvwatts(dc / 12, pdc0=400 / 0.965, eta_inv_nom=0.965, eta_inv_ref=0.9637)
    np.testing.assert_allclose(sim["pac_w"], np.maximum(ac, 0), rtol=0, atol=1e-9)
    # equal per-panel split: every unit sees the same part-load ratio, so no MPPT/shading advantage is invented
    single = pvlib.inverter.pvwatts(dc, pdc0=4800 / 0.965, eta_inv_nom=0.965, eta_inv_ref=0.9637)
    np.testing.assert_allclose(single, sim["pac_w"], rtol=1e-12, atol=1e-9)
    assert (sim["pac_w"] <= house.inverter_ac_rated_power_w + 1e-9).all()


def test_dc_is_not_capped_at_array_rating(config):
    house = config.houses[4]
    values = pd.DataFrame({
        "Sun Zenith (rad)": [math.radians(20)], "Sun Azimuth (rad)": [math.pi], "GHI": [1150.0], "DNI": [1000.0],
        "DHI": [200.0], "Surface Albedo": [0.3], "Temperature": [-5.0], "Wind Speed": [6.0],
    })
    sim = sp.simulate(values, pd.DatetimeIndex([pd.Timestamp("2022-04-01 12:00")]), house, config.physics)
    assert sim["pdc_w"].iloc[0] > 0.9 * house.array_stc_power_w  # cold + bright: above rating before loss
    assert sim["pac_w"].iloc[0] == pytest.approx(house.inverter_ac_rated_power_w)


@pytest.mark.parametrize(
    "mutate, check",
    [
        (lambda lines: lines.__setitem__(10, lines[10].replace(",0.0,", ",,", 1)), "empty_or_whitespace_cell"),
        (lambda lines: lines.__setitem__(10, lines[10].replace(",0.0,", ",   ,", 1)), "empty_or_whitespace_cell"),
        (lambda lines: lines.__setitem__(10, lines[10].replace(",0.0,", ",NA,", 1)), "missing_value_token"),
        (lambda lines: lines.__setitem__(10, lines[10].replace(",0.0,", ",abc,", 1)), "invalid_numeric"),
        (lambda lines: lines.__setitem__(10, lines[10].replace(",0.0,", ",inf,", 1)), "infinity"),
        (lambda lines: lines.__setitem__(10, lines[10].replace(",0.0,", ",-9999,", 1)), "placeholder_candidate"),
        (lambda lines: lines.__setitem__(11, lines[10]), "duplicate_timestamp"),
        (lambda lines: lines.pop(20), "timestamp_gap"),
        (lambda lines: lines.__setitem__(12, lines[12].replace(":45:00", ":44:00", 1)), "off_grid_timestamp"),
        (lambda lines: lines.insert(15, ""), "blank_line"),
        (lambda lines: lines.__setitem__(10, lines[10] + ",1"), "wrong_field_count"),
        (lambda lines: lines.__setitem__(10, lines[10].replace("2021-01-01", "2021-13-01", 1)), "invalid_timestamp"),
    ],
)
def test_audit_blocks_defective_house_only(inputs, tmp_path, mutate, check):
    path = inputs / "Weather House 2.csv"
    lines = path.read_text(encoding="utf-8").split("\n")
    mutate(lines)
    path.write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "out"
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(out), "--houses", "1,2"]) == 1
    audit = json.loads((out / "metadata" / "audit_report.json").read_text(encoding="utf-8"))
    assert check in [d["check"] for d in audit["2"]["defects"]]
    assert audit["1"]["ok"] is True
    assert not (out / "model_datasets" / "house_02_model_dataset.csv").exists()
    assert (out / "model_datasets" / "house_01_model_dataset.csv").exists()


def test_night_zeros_and_zero_celsius_are_valid(inputs, config):
    audit, parsed = sp.audit_file(inputs / "Weather House 1.csv", 1, config)
    assert audit.ok and parsed is not None
    assert (parsed.values["Temperature"] == 0.0).any() and (parsed.values["GHI"] == 0.0).any()


def test_long_decimals_are_parsed_exactly(tmp_path, config):
    frame = synthetic_frame(periods=5, seed=3)
    tricky = ["0.30000000000000004", "2.8394668115675203", "123456.78901234567", "1e-300", "0.1"]
    frame["Sun Azimuth (rad)"] = tricky
    path = tmp_path / "Weather House 1.csv"
    frame.to_csv(path, index=False)
    audit, parsed = sp.audit_file(path, 1, config)
    assert audit.ok
    assert parsed.values["Sun Azimuth (rad)"].tolist() == [float(t) for t in tricky]


def test_energy_summary_uses_n_minus_1_trapezoids(config):
    times = pd.DatetimeIndex(pd.date_range("2021-12-31 23:00", periods=9, freq="15min"))
    sim = pd.DataFrame({"pdc_w": np.arange(9, dtype=float) * 100, "pac_w": np.arange(9, dtype=float) * 90})
    rows = sp.energy_summary(config.houses[1], times, sim)
    total = next(r for r in rows if r["Period"] == "all")
    assert total["Intervals"] == 8
    assert total["DC_Energy_kWh"] == pytest.approx(sum((i * 100 + (i + 1) * 100) / 2 * 0.25 / 1000 for i in range(8)))
    by_year = {r["Period"]: r for r in rows}
    assert by_year["2021"]["Intervals"] == 4 and by_year["2022"]["Intervals"] == 4
    assert by_year["2021"]["Coverage"] == "partial"


def test_reference_comparison_and_input_protection(inputs, tmp_path):
    first = tmp_path / "first"
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(first), "--houses", "6"]) == 0
    second = tmp_path / "second"
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(second), "--houses", "6",
                    "--reference-dir", str(first / "model_datasets")]) == 0
    ref = json.loads((second / "metadata" / "validation_report.json").read_text(encoding="utf-8"))["6"]["reference_comparison"]
    assert ref["passed"] is True and ref["PV_AC_Power_W"]["max_abs_diff_w"] == 0.0
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(inputs)]) == 2


def test_missing_input_is_reported(tmp_path):
    assert sp.main(["--input-dir", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "o")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    assert sp.main(["--input-dir", str(empty), "--output-dir", str(out), "--houses", "1"]) == 1
    audit = json.loads((out / "metadata" / "audit_report.json").read_text(encoding="utf-8"))
    assert audit["1"]["defects"][0]["check"] == "file_missing"


def test_existing_results_are_not_silently_mixed(inputs, tmp_path):
    out = tmp_path / "out"
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(out), "--houses", "1,2"]) == 0
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(out), "--houses", "1"]) == 2  # refused
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(out), "--houses", "1", "--overwrite"]) == 0
    assert not (out / "model_datasets" / "house_02_model_dataset.csv").exists()  # stale result removed
    assert (out / "model_datasets" / "house_01_model_dataset.csv").exists()


def _load_house_file(house_id: int):
    import importlib.util

    path = ROOT / f"house_{house_id:02d}.py"
    spec = importlib.util.spec_from_file_location(f"house_{house_id:02d}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the __main__ guard keeps it from running
    return module


@pytest.mark.parametrize("house_id", range(1, 14))
def test_house_files_match_the_configuration(config, house_id):
    module = _load_house_file(house_id)
    reference = config.houses[house_id]
    assert {k: getattr(reference, k) for k in sp.REQUIRED_HOUSE_KEYS} == module.HOUSE
    assert str(module.INPUT_DIR) == r"C:\Users\MI Electronics\Downloads" or module.INPUT_DIR.name == "Downloads"


def test_single_house_run_matches_the_full_run(inputs, tmp_path, config):
    full = tmp_path / "full"
    assert sp.main(["--input-dir", str(inputs), "--output-dir", str(full), "--houses", "3,8"]) == 0
    single = tmp_path / "single"
    for house_id in (3, 8):
        assert sp.run_single_house(_load_house_file(house_id).HOUSE, inputs, single) == 0
        name = f"model_datasets/house_{house_id:02d}_model_dataset.csv"
        assert (single / name).read_bytes() == (full / name).read_bytes()
    meta = json.loads((single / "metadata" / "house_08_run_metadata.json").read_text(encoding="utf-8"))
    assert meta["settings_differ_from_simulation_config"] == {}
    assert meta["house"]["notes"] == list(config.houses[8].notes)
    assert (single / "house_08_energy_summary.csv").exists()


def test_single_house_rerun_replaces_only_its_own_results(inputs, tmp_path):
    out = tmp_path / "out"
    h1, h2 = _load_house_file(1).HOUSE, _load_house_file(2).HOUSE
    assert sp.run_single_house(h1, inputs, out) == 0
    assert sp.run_single_house(h2, inputs, out) == 0
    edited = {**h2, "panel_count": 6}
    assert sp.run_single_house(edited, inputs, out) == 0
    meta = json.loads((out / "metadata" / "house_02_run_metadata.json").read_text(encoding="utf-8"))
    assert meta["settings_differ_from_simulation_config"]["panel_count"] == {"house_file": 6, "simulation_config": 12}
    res = pd.read_csv(out / "model_datasets" / "house_02_model_dataset.csv")
    assert res["Array_STC_Power_W"].eq(6 * 540).all()
    # a defect in house 2's source now blocks it and removes its stale dataset, leaving house 1 untouched
    path = inputs / "Weather House 2.csv"
    lines = path.read_text(encoding="utf-8").split("\n")
    lines.pop(20)
    path.write_text("\n".join(lines), encoding="utf-8")
    assert sp.run_single_house(h2, inputs, out) == 1
    assert not (out / "model_datasets" / "house_02_model_dataset.csv").exists()
    assert (out / "model_datasets" / "house_01_model_dataset.csv").exists()
    assert sp.run_single_house(h2, inputs, inputs) == 2
    missing = tmp_path / "no_such_folder"
    assert sp.run_single_house(h2, missing, missing / "out") == 2
    assert not missing.exists()  # nothing is created under a wrong INPUT_DIR


def test_house_settings_are_checked():
    with pytest.raises(sp.ConfigError):
        sp.make_house({**_load_house_file(3).HOUSE, "inverter_unit_count": 1})  # micro needs one unit per panel
    with pytest.raises(sp.ConfigError):
        sp.make_house({**_load_house_file(1).HOUSE, "inverter_nominal_efficiency": 93})
    with pytest.raises(sp.ConfigError):
        sp.make_house({**_load_house_file(1).HOUSE, "panel_count": 10.5})
