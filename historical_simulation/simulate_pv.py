#!/usr/bin/env python3
"""Historical available-PV-power simulation for the 13 "Weather House" CSVs (pvlib).

Weather becomes power in five steps, independently at every source timestamp:

  1. Source solar zenith/azimuth (radians) -> degrees for pvlib only.
  2. Extraterrestrial DNI from the timestamp's day of year (pvlib defaults).
  3. Front-side plane-of-array irradiance, Hay-Davies, from the supplied GHI/DNI/DHI,
     Surface Albedo and solar angles; POA >= 0 and POA = 0 when zenith >= 90 deg.
  4. Cell temperature, SAPM 'open_rack_glass_polymer' (generic for all houses).
  5. PVWatts DC at the array STC rating, clipped at 0, times 0.90 (10 % aggregate
     DC loss) -> PV_Power_Generation_W; then the PVWatts inverter (part-load curve
     and AC clipping) per inverter unit -> PV_AC_Power_W.

All power values are SIMULATED available PV potential, not measured generation.
Nothing in the source files is filled, interpolated, dropped or replaced: a file
with any audit defect is reported and its output is not generated.

Run:
    python simulate_pv.py --input-dir "C:\\Users\\MI Electronics\\Downloads" \\
        --output-dir pv_simulation_output --energy-summary --zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pvlib

SCRIPT_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent

FINAL_COLUMNS = [
    "Timestamp",
    "House_ID",
    "Panel_Tilt_rad",
    "Panel_Azimuth_rad",
    "Temperature",
    "Relative Humidity",
    "GHI",
    "DNI",
    "DHI",
    "Wind Speed",
    "Solar_Zenith_rad",
    "Solar_Azimuth_rad",
    "PV_Power_Generation_W",
    "PV_AC_Power_W",
    "Panel_Count",
    "Panel_STC_Power_W",
    "Array_STC_Power_W",
    "Inverter_Unit_Count",
    "Inverter_AC_Rated_Power_W",
    "Inverter_Nominal_Efficiency",
]

# Columns deliberately not exported (their physics is still computed internally).
REMOVED_COLUMNS = ("Fill_Flag", "Record_Type", "POA_Global_Wm2", "PV_Cell_Temperature_C", "Inverter_Type")
# Lower-case fragments that would indicate one of the removed columns under an alias.
REMOVED_ALIAS_FRAGMENTS = ("fill", "record", "poa", "cell", "inverter_type", "architecture")

SOURCE_COLUMNS = [
    "Timestamp", "Temperature", "Clearsky DHI", "Clearsky DNI", "Clearsky GHI", "Cloud Type", "Dew Point",
    "DHI", "DNI", "Fill Flag", "GHI", "Ozone", "Relative Humidity", "Surface Albedo", "Pressure",
    "Precipitable Water", "Wind Direction", "Wind Speed", "Sun Azimuth (rad)", "Sun Zenith (rad)",
]
NUMERIC_SOURCE_COLUMNS = [c for c in SOURCE_COLUMNS if c != "Timestamp"]

# Source column -> exported column (values copied unchanged).
PASSTHROUGH = {
    "Temperature": "Temperature",
    "Relative Humidity": "Relative Humidity",
    "GHI": "GHI",
    "DNI": "DNI",
    "DHI": "DHI",
    "Wind Speed": "Wind Speed",
    "Sun Zenith (rad)": "Solar_Zenith_rad",
    "Sun Azimuth (rad)": "Solar_Azimuth_rad",
}

MISSING_TOKENS = {
    "na", "n/a", "n.a.", "nan", "null", "none", "nil", "missing", "-", "--", "---", "?", "#n/a", "#na",
    "#value!", "#div/0!", "#ref!", "#num!", "#null!", "undefined", "empty",
}
# Common "no data" sentinels. 999 is deliberately absent: 999 mbar is a valid pressure.
PLACEHOLDER_VALUES = (-99999.0, -9999.0, -999.9, -999.0, -99.9, -99.0, 9999.0, 99999.0)
INTEGER_TEXT = r"[+-]?\d+"
OUTPUT_TIMESTAMP_PATTERN = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

REFERENCE_VERSIONS = {"python": "3.12", "pvlib": "0.16.1", "numpy": "2.3.5", "pandas": "3.0.1", "scipy": "1.18.1"}

PROVENANCE_STATEMENT = (
    "PV_Power_Generation_W and PV_AC_Power_W are SIMULATED available PV potential computed with pvlib from "
    "the source weather files and scenario installation settings. They are not measured generation, include no "
    "battery, load, grid or MPPT curtailment, and cannot establish real forecasting accuracy."
)

POWER_TOLERANCE_W = 1e-7
RATING_TOLERANCE_W = 1e-9
MAX_EXAMPLES = 5


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class House:
    house_id: int
    panel_identification: str
    panel_count: int
    panel_stc_power_w: float
    tilt_deg: float
    azimuth_deg: float
    gamma_pdc_per_c: float
    inverter_type: str  # internal only: hybrid | string | micro
    inverter_unit_count: int
    inverter_ac_power_w_per_unit: float
    inverter_nominal_efficiency: float
    notes: tuple[str, ...] = ()

    @property
    def array_stc_power_w(self) -> float:
        return self.panel_count * self.panel_stc_power_w

    @property
    def inverter_ac_rated_power_w(self) -> float:
        """Aggregate AC rating across all inverter units."""
        return self.inverter_unit_count * self.inverter_ac_power_w_per_unit


@dataclass(frozen=True)
class Physics:
    transposition_model: str
    temperature_model: str
    temperature_model_parameters: str
    dc_loss_fraction: float
    temp_ref_c: float
    inverter_model: str
    inverter_eta_inv_ref: float
    horizon_zenith_deg: float
    bifacial_rear_gain: float


@dataclass
class Config:
    raw: dict[str, Any]
    houses: dict[int, House]
    physics: Physics
    io: dict[str, Any]
    expected_source: dict[str, Any]


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    physics = Physics(**raw["physics"])
    if physics.transposition_model != "haydavies" or physics.temperature_model != "sapm" or physics.inverter_model != "pvwatts":
        raise ConfigError("this script implements haydavies transposition, SAPM cell temperature and the PVWatts inverter only")
    if not 0.0 <= physics.dc_loss_fraction < 1.0:
        raise ConfigError("dc_loss_fraction must be in [0, 1)")
    if physics.bifacial_rear_gain != 0.0:
        raise ConfigError("rear-side (bifacial) gain is not modelled; bifacial_rear_gain must be 0")
    if physics.temperature_model_parameters not in pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]:
        raise ConfigError(f"unknown SAPM parameter set {physics.temperature_model_parameters!r}")

    houses: dict[int, House] = {}
    for h in raw["houses"]:
        house = make_house(h)
        if house.house_id in houses:
            raise ConfigError(f"house {house.house_id}: duplicate house_id")
        houses[house.house_id] = house
    return Config(raw, houses, physics, raw["io"], raw.get("expected_source", {}))


REQUIRED_HOUSE_KEYS = (
    "house_id", "panel_identification", "panel_count", "panel_stc_power_w", "tilt_deg", "azimuth_deg",
    "gamma_pdc_per_c", "inverter_type", "inverter_unit_count", "inverter_ac_power_w_per_unit",
    "inverter_nominal_efficiency",
)


def make_house(settings: Mapping[str, Any], default_notes: tuple[str, ...] = ()) -> House:
    """Build and check one house's settings (from simulation_config.json or a house_XX.py file)."""
    missing = [k for k in REQUIRED_HOUSE_KEYS if k not in settings]
    unknown = sorted(set(settings) - set(House.__dataclass_fields__))
    if missing or unknown:
        raise ConfigError(f"house settings: missing {missing}, unknown {unknown}")
    house = House(**{**settings, "notes": tuple(settings.get("notes", default_notes))})
    problems = []
    for key in ("house_id", "panel_count", "inverter_unit_count"):
        value = getattr(house, key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            problems.append(f"{key} must be a positive whole number")
    if house.inverter_type not in ("hybrid", "string", "micro"):
        problems.append("inverter_type must be hybrid, string or micro")
    elif house.inverter_type == "micro" and house.inverter_unit_count != house.panel_count:
        problems.append("microinverter scenarios use one unit per panel")
    elif house.inverter_type != "micro" and house.inverter_unit_count != 1:
        problems.append("string/hybrid scenarios use one inverter unit")
    if not house.panel_stc_power_w > 0 or not house.inverter_ac_power_w_per_unit > 0:
        problems.append("panel and inverter ratings must be positive")
    if not 0.0 < house.inverter_nominal_efficiency <= 1.0:
        problems.append("inverter_nominal_efficiency must be a fraction, e.g. 0.93")
    if not -0.02 <= house.gamma_pdc_per_c <= 0.0:
        problems.append("gamma_pdc_per_c must be a fraction per degC, e.g. -0.0035")
    if not 0.0 <= house.tilt_deg <= 90.0 or not 0.0 <= house.azimuth_deg < 360.0:
        problems.append("tilt must be 0-90 deg and azimuth 0-360 deg")
    if problems:
        raise ConfigError(f"house {house.house_id}: " + "; ".join(problems))
    return house


# --------------------------------------------------------------------------- audit


@dataclass
class Finding:
    check: str
    column: str | None
    count: int
    examples: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""


@dataclass
class FileAudit:
    house_id: int
    path: str
    exists: bool
    sha256: str | None = None
    size_bytes: int | None = None
    data_rows: int | None = None
    header: list[str] | None = None
    timestamp_format: str | None = None
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    defects: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exists and not self.defects


@dataclass
class ParsedWeather:
    labels: pd.Series  # original timestamp text (stripped)
    times: pd.DatetimeIndex  # naive: labels are preserved, never localized or shifted
    values: pd.DataFrame  # numeric source columns, parsed exactly


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _examples(mask: np.ndarray, labels: pd.Series, values: pd.Series | None = None) -> list[dict[str, Any]]:
    out = []
    for i in np.flatnonzero(mask)[:MAX_EXAMPLES]:
        item: dict[str, Any] = {"line": int(i) + 2, "timestamp": str(labels.iloc[i])}
        if values is not None:
            item["value"] = str(values.iloc[i])
        out.append(item)
    return out


def _exact_float(text: pd.Series) -> np.ndarray:
    """Correctly rounded parsing (pandas' fast parser can differ in the last bit)."""
    return np.fromiter((float(t) for t in text), dtype=np.float64, count=len(text))


def audit_file(path: Path, house_id: int, config: Config) -> tuple[FileAudit, ParsedWeather | None]:
    audit = FileAudit(house_id, str(path), path.is_file())
    if not audit.exists:
        audit.defects.append(Finding("file_missing", None, 1, detail=f"{path} not found"))
        return audit, None
    audit.sha256 = sha256_file(path)
    audit.size_bytes = path.stat().st_size
    encoding = config.io.get("input_encoding", "utf-8-sig")

    # Structure: every line must have exactly the header's number of fields.
    with path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            audit.defects.append(Finding("empty_file", None, 1))
            return audit, None
        audit.header = header
        width, bad_lines, blank_lines, n = len(header), [], [], 0
        for n, row in enumerate(reader, start=2):
            if not row:
                blank_lines.append(n)
            elif len(row) != width:
                bad_lines.append((n, len(row)))
    if blank_lines:
        audit.defects.append(Finding("blank_line", None, len(blank_lines), [{"line": l} for l in blank_lines[:MAX_EXAMPLES]],
                                     "a blank line is a missing record"))
    if bad_lines:
        audit.defects.append(Finding("wrong_field_count", None, len(bad_lines),
                                     [{"line": l, "fields": f, "expected": width} for l, f in bad_lines[:MAX_EXAMPLES]]))
    duplicates = sorted({c for c in header if header.count(c) > 1})
    missing = [c for c in SOURCE_COLUMNS if c not in header]
    extra = [c for c in header if c not in SOURCE_COLUMNS]
    if duplicates:
        audit.defects.append(Finding("duplicate_column_name", ", ".join(duplicates), len(duplicates)))
    if missing:
        audit.defects.append(Finding("missing_column", ", ".join(missing), len(missing)))
    if extra:
        audit.warnings.append(Finding("extra_column", ", ".join(extra), len(extra), detail="not used"))
    if bad_lines or duplicates or missing:
        return audit, None

    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, skip_blank_lines=False,
                      index_col=False, encoding=encoding)
    audit.data_rows = len(raw)
    stripped = {c: raw[c].str.strip() for c in SOURCE_COLUMNS}
    labels = stripped["Timestamp"]

    # --- cell-level checks
    values: dict[str, np.ndarray | pd.Series] = {}
    for column in SOURCE_COLUMNS:
        s = stripped[column]
        empty = (s == "").to_numpy()
        token = s.str.lower().isin(MISSING_TOKENS).to_numpy() & ~empty
        if empty.any():
            audit.defects.append(Finding("empty_or_whitespace_cell", column, int(empty.sum()), _examples(empty, labels, raw[column])))
        if token.any():
            audit.defects.append(Finding("missing_value_token", column, int(token.sum()), _examples(token, labels, s)))
        if column == "Timestamp":
            continue
        candidate = ~(empty | token)
        coerced = pd.to_numeric(s.where(candidate, "0"), errors="coerce").to_numpy(dtype="float64")
        invalid = np.isnan(coerced) & candidate
        exact = np.full(len(s), np.nan)
        ok = candidate & ~invalid
        try:
            exact[ok] = _exact_float(s[ok])
        except ValueError:  # accepted by pandas but not by float(): treat as invalid
            for i in np.flatnonzero(ok):
                try:
                    exact[i] = float(s.iloc[i])
                except ValueError:
                    invalid[i] = True
        if invalid.any():
            audit.defects.append(Finding("invalid_numeric", column, int(invalid.sum()), _examples(invalid, labels, s)))
        infinite = np.isinf(exact)
        if infinite.any():
            audit.defects.append(Finding("infinity", column, int(infinite.sum()), _examples(infinite, labels, s)))
        placeholder = np.isin(exact, PLACEHOLDER_VALUES)
        if placeholder.any():
            audit.defects.append(Finding("placeholder_candidate", column, int(placeholder.sum()), _examples(placeholder, labels, s),
                                         f"values in {PLACEHOLDER_VALUES}"))
        if ok.all() and s.str.fullmatch(INTEGER_TEXT).all():
            values[column] = s.astype("int64").to_numpy()
        else:
            values[column] = exact

    # --- timestamps
    formats = config.io["timestamp_formats"]
    best, best_bad = None, None
    for fmt in formats:
        parsed = pd.to_datetime(labels, format=fmt, errors="coerce")
        bad = int(parsed.isna().sum())
        if best_bad is None or bad < best_bad:
            best, best_bad, audit.timestamp_format = parsed, bad, fmt
        if bad == 0:
            break
    times = pd.DatetimeIndex(best)
    invalid_ts = times.isna()
    if invalid_ts.any():
        audit.defects.append(Finding("invalid_timestamp", "Timestamp", int(invalid_ts.sum()), _examples(invalid_ts, labels),
                                     f"best matching format {audit.timestamp_format!r}; add the correct format to io.timestamp_formats"))
    else:
        step = pd.Timedelta(minutes=config.io["timestep_minutes"])
        audit.first_timestamp, audit.last_timestamp = str(times[0]), str(times[-1])
        dup = times.duplicated(keep=False)
        if dup.any():
            audit.defects.append(Finding("duplicate_timestamp", "Timestamp", int(dup.sum()), _examples(dup, labels)))
        diffs_ns = np.diff(times.as_unit("ns").asi8)
        step_ns = step.value  # Timedelta.value is always nanoseconds
        backwards = np.concatenate([[False], diffs_ns < 0])
        if backwards.any():
            audit.defects.append(Finding("non_monotonic_timestamp", "Timestamp", int(backwards.sum()), _examples(backwards, labels)))
        gap = np.concatenate([[False], (diffs_ns > step_ns) & (diffs_ns % step_ns == 0)])
        if gap.any():
            missing_rows = int(((diffs_ns[gap[1:]] // step_ns) - 1).sum())
            audit.defects.append(Finding("timestamp_gap", "Timestamp", missing_rows, _examples(gap, labels),
                                         "missing whole rows (count = missing records); example is the record after each gap"))
        off_step = np.concatenate([[False], (diffs_ns > 0) & (diffs_ns % step_ns != 0)])
        misaligned = ((times.minute % config.io["timestep_minutes"]) != 0) | (times.second != 0) | (times.microsecond != 0)
        off_grid = np.asarray(off_step | misaligned)
        if off_grid.any():
            audit.defects.append(Finding("off_grid_timestamp", "Timestamp", int(off_grid.sum()), _examples(off_grid, labels)))

        exp = config.expected_source
        mismatches = []
        if exp.get("rows_per_file") is not None and audit.data_rows != exp["rows_per_file"]:
            mismatches.append(f"rows {audit.data_rows} != expected {exp['rows_per_file']}")
        if exp.get("first_timestamp") and times[0] != pd.Timestamp(exp["first_timestamp"]):
            mismatches.append(f"first {times[0]} != expected {exp['first_timestamp']}")
        if exp.get("last_timestamp") and times[-1] != pd.Timestamp(exp["last_timestamp"]):
            mismatches.append(f"last {times[-1]} != expected {exp['last_timestamp']}")
        if mismatches:
            audit.warnings.append(Finding("differs_from_earlier_audit", None, len(mismatches), detail="; ".join(mismatches)))

    if audit.defects:
        return audit, None
    frame = pd.DataFrame({c: values[c] for c in NUMERIC_SOURCE_COLUMNS})
    audit.diagnostics = diagnostics(frame, times, labels)
    for name, finding in audit.diagnostics.pop("_warnings", {}).items():
        audit.warnings.append(finding)
    return audit, ParsedWeather(labels.reset_index(drop=True), times, frame)


def diagnostics(frame: pd.DataFrame, times: pd.DatetimeIndex, labels: pd.Series) -> dict[str, Any]:
    """Informational checks. They never change data and never block generation."""
    zen, az = frame["Sun Zenith (rad)"].to_numpy(float), frame["Sun Azimuth (rad)"].to_numpy(float)
    out: dict[str, Any] = {"_warnings": {}}
    ranges = {
        "negative_irradiance_GHI": int((frame["GHI"] < 0).sum()),
        "negative_irradiance_DNI": int((frame["DNI"] < 0).sum()),
        "negative_irradiance_DHI": int((frame["DHI"] < 0).sum()),
        "relative_humidity_outside_0_100": int(((frame["Relative Humidity"] < 0) | (frame["Relative Humidity"] > 100)).sum()),
        "albedo_outside_0_1": int(((frame["Surface Albedo"] < 0) | (frame["Surface Albedo"] > 1)).sum()),
        "negative_wind_speed": int((frame["Wind Speed"] < 0).sum()),
        "zenith_outside_0_pi": int(((zen < 0) | (zen > math.pi)).sum()),
        "azimuth_outside_0_2pi": int(((az < 0) | (az > 2 * math.pi)).sum()),
    }
    out["range_checks"] = ranges
    for name, count in ranges.items():
        if count:
            out["_warnings"][name] = Finding("value_range", name, count, detail="reported only; values are used as supplied")

    night = zen >= math.radians(90.0)
    ghi = frame["GHI"].to_numpy(float)
    out["records_with_zenith_ge_90_but_ghi_gt_0"] = int((night & (ghi > 0)).sum())
    out["zero_or_below_zero_celsius_records"] = int((frame["Temperature"] <= 0).sum())

    # Timestamp semantics: clock time of the daily minimum zenith (solar noon) and the azimuth there.
    df = pd.DataFrame({"zen": zen, "az": az, "t": times})
    df["date"] = df["t"].dt.date
    idx = df.groupby("date")["zen"].idxmin()
    noon = df.loc[idx]
    minutes = noon["t"].dt.hour * 60 + noon["t"].dt.minute
    month = noon["t"].dt.month
    fmt = lambda m: None if pd.isna(m) else f"{int(m) // 60:02d}:{int(m) % 60:02d}"
    out["solar_noon_label_time_median"] = fmt(minutes.median())
    out["solar_noon_label_time_median_dec_feb"] = fmt(minutes[month.isin([12, 1, 2])].median())
    out["solar_noon_label_time_median_jun_aug"] = fmt(minutes[month.isin([6, 7, 8])].median())
    az_noon = float(np.degrees(np.median(noon["az"])))
    out["azimuth_at_solar_noon_median_deg"] = round(az_noon, 2)
    out["interpretation"] = (
        "Solar-noon clock times show how timestamps are labelled (a ~60 min summer/winter difference suggests local "
        "civil time with daylight saving). An azimuth near 180 deg at solar noon is consistent with the north = 0, "
        "clockwise convention pvlib expects. Nothing is shifted or re-computed from these diagnostics."
    )
    if not 150.0 <= az_noon <= 210.0:
        out["_warnings"]["azimuth_convention"] = Finding(
            "azimuth_convention", "Sun Azimuth (rad)", 1,
            detail=f"median azimuth at solar noon is {az_noon:.1f} deg, not ~180 deg; the source convention may differ "
                   "from pvlib's (north = 0, clockwise) and POA would then be wrong",
        )
    return out


# --------------------------------------------------------------------------- physics


def simulate(values: pd.DataFrame, times: pd.DatetimeIndex, house: House, physics: Physics) -> pd.DataFrame:
    """DC and AC power at every source timestamp. POA and cell temperature are internal only."""
    index = pd.RangeIndex(len(values))
    zenith_deg = np.degrees(values["Sun Zenith (rad)"].to_numpy(dtype="float64"))
    azimuth_deg = np.degrees(values["Sun Azimuth (rad)"].to_numpy(dtype="float64"))
    as_series = lambda col: pd.Series(values[col].to_numpy(dtype="float64"), index=index)

    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=house.tilt_deg,
        surface_azimuth=house.azimuth_deg,
        solar_zenith=pd.Series(zenith_deg, index=index),
        solar_azimuth=pd.Series(azimuth_deg, index=index),
        dni=as_series("DNI"),
        ghi=as_series("GHI"),
        dhi=as_series("DHI"),
        dni_extra=pd.Series(np.asarray(dni_extra, dtype="float64"), index=index),
        albedo=as_series("Surface Albedo"),
        model=physics.transposition_model,
    )
    poa = np.maximum(irradiance["poa_global"].to_numpy(dtype="float64"), 0.0)
    poa[zenith_deg >= physics.horizon_zenith_deg] = 0.0

    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][physics.temperature_model_parameters]
    temp_cell = np.asarray(pvlib.temperature.sapm_cell(
        poa, values["Temperature"].to_numpy(dtype="float64"), values["Wind Speed"].to_numpy(dtype="float64"), **params
    ), dtype="float64")

    pdc_gross = np.asarray(pvlib.pvsystem.pvwatts_dc(
        poa, temp_cell, pdc0=house.array_stc_power_w, gamma_pdc=house.gamma_pdc_per_c, temp_ref=physics.temp_ref_c
    ), dtype="float64")
    pdc_w = np.maximum(pdc_gross, 0.0) * (1.0 - physics.dc_loss_fraction)

    n = house.inverter_unit_count
    pdc_per_unit = pdc_w / n
    inverter_dc_reference_w = house.inverter_ac_power_w_per_unit / house.inverter_nominal_efficiency
    pac_w = n * np.asarray(pvlib.inverter.pvwatts(
        pdc_per_unit,
        pdc0=inverter_dc_reference_w,
        eta_inv_nom=house.inverter_nominal_efficiency,
        eta_inv_ref=physics.inverter_eta_inv_ref,
    ), dtype="float64")
    pac_w = np.maximum(pac_w, 0.0)

    return pd.DataFrame({
        "zenith_deg": zenith_deg,
        "poa_global_wm2": poa,
        "temp_cell_c": temp_cell,
        "pdc_gross_w": pdc_gross,
        "pdc_w": pdc_w,
        "pac_w": pac_w,
    })


def build_output(weather: ParsedWeather, house: House, sim: pd.DataFrame, out_format: str) -> pd.DataFrame:
    n = len(sim)
    columns: dict[str, Any] = {
        "Timestamp": weather.times.strftime(out_format),
        "House_ID": np.full(n, house.house_id, dtype="int64"),
        "Panel_Tilt_rad": np.full(n, math.radians(house.tilt_deg)),
        "Panel_Azimuth_rad": np.full(n, math.radians(house.azimuth_deg)),
    }
    for source, target in PASSTHROUGH.items():
        columns[target] = weather.values[source].to_numpy()
    columns["PV_Power_Generation_W"] = sim["pdc_w"].to_numpy()
    columns["PV_AC_Power_W"] = sim["pac_w"].to_numpy()
    columns["Panel_Count"] = np.full(n, house.panel_count, dtype="int64")
    columns["Panel_STC_Power_W"] = np.full(n, house.panel_stc_power_w)
    columns["Array_STC_Power_W"] = np.full(n, house.array_stc_power_w)
    columns["Inverter_Unit_Count"] = np.full(n, house.inverter_unit_count, dtype="int64")
    columns["Inverter_AC_Rated_Power_W"] = np.full(n, house.inverter_ac_rated_power_w)
    columns["Inverter_Nominal_Efficiency"] = np.full(n, float(house.inverter_nominal_efficiency))
    frame = pd.DataFrame(columns)
    return frame[FINAL_COLUMNS]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


# --------------------------------------------------------------------------- validation


def _check(results: dict[str, Any], name: str, passed: bool, **detail: Any) -> None:
    results[name] = {"passed": bool(passed), **detail}


def validate_output(path: Path, weather: ParsedWeather, house: House, sim: pd.DataFrame, physics: Physics,
                    out_format: str, reference_dir: Path | None = None) -> dict[str, Any]:
    r: dict[str, Any] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        header_line = fh.readline().rstrip("\n")
    _check(r, "header_exact", header_line == ",".join(FINAL_COLUMNS), header=header_line)
    out = pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_filter=False)
    names = list(out.columns)
    _check(r, "columns_exact_order", names == FINAL_COLUMNS, count=len(names))
    leaked = [c for c in names if c in REMOVED_COLUMNS or any(f in c.lower() for f in REMOVED_ALIAS_FRAGMENTS)]
    _check(r, "removed_columns_absent", not leaked, leaked=leaked)
    _check(r, "no_index_column", names[0] == "Timestamp" and len(names) == 20)

    expected_ts = weather.times.strftime(out_format)
    _check(r, "row_count_matches_source", len(out) == len(weather.values), rows=len(out), source_rows=len(weather.values))
    ts_equal = len(out) == len(expected_ts) and bool((out["Timestamp"].astype(str).to_numpy() == np.asarray(expected_ts)).all())
    reparsed = pd.to_datetime(out["Timestamp"], format=out_format, errors="coerce")
    _check(r, "timestamps_match_source", ts_equal and bool((pd.DatetimeIndex(reparsed) == weather.times).all()),
           first=str(out["Timestamp"].iloc[0]), last=str(out["Timestamp"].iloc[-1]))
    _check(r, "timestamp_format_without_offset", bool(out["Timestamp"].astype(str).str.fullmatch(OUTPUT_TIMESTAMP_PATTERN).all()))

    preserved = {}
    for source, target in PASSTHROUGH.items():
        a = out[target].to_numpy(dtype="float64")
        b = weather.values[source].to_numpy(dtype="float64")
        preserved[target] = bool(np.array_equal(a, b))
    _check(r, "source_values_preserved_exactly", all(preserved.values()), columns=preserved)

    const_ok = (
        (out["House_ID"] == house.house_id).all()
        and (out["Panel_Count"] == house.panel_count).all()
        and (out["Panel_STC_Power_W"] == house.panel_stc_power_w).all()
        and (out["Array_STC_Power_W"] == house.panel_count * house.panel_stc_power_w).all()
        and (out["Inverter_Unit_Count"] == house.inverter_unit_count).all()
        and (out["Inverter_AC_Rated_Power_W"] == house.inverter_unit_count * house.inverter_ac_power_w_per_unit).all()
        and (out["Inverter_Nominal_Efficiency"] == house.inverter_nominal_efficiency).all()
        and (out["Panel_Tilt_rad"] == math.radians(house.tilt_deg)).all()
        and (out["Panel_Azimuth_rad"] == math.radians(house.azimuth_deg)).all()
    )
    _check(r, "installation_constants", bool(const_ok),
           array_stc_power_w=house.array_stc_power_w, inverter_ac_rated_power_w=house.inverter_ac_rated_power_w)

    dc = out["PV_Power_Generation_W"].to_numpy(dtype="float64")
    ac = out["PV_AC_Power_W"].to_numpy(dtype="float64")
    _check(r, "power_roundtrip_exact", bool(np.array_equal(dc, sim["pdc_w"].to_numpy()) and np.array_equal(ac, sim["pac_w"].to_numpy())))
    _check(r, "power_finite_nonnegative", bool(np.isfinite(dc).all() and np.isfinite(ac).all() and (dc >= 0).all() and (ac >= 0).all()),
           min_dc=float(np.min(dc)), min_ac=float(np.min(ac)))
    below = weather.values["Sun Zenith (rad)"].to_numpy(dtype="float64") >= math.radians(physics.horizon_zenith_deg)
    _check(r, "zero_power_below_horizon", bool((dc[below] == 0).all() and (ac[below] == 0).all()), records_below_horizon=int(below.sum()))
    rating = house.inverter_ac_rated_power_w
    _check(r, "ac_not_above_aggregate_rating", bool((ac <= rating + RATING_TOLERANCE_W).all()),
           max_ac_w=float(ac.max()), rating_w=rating, max_excess_w=float(max(0.0, (ac - rating).max())),
           tolerance_w=RATING_TOLERANCE_W, records_at_clipping=int(np.isclose(ac, rating, rtol=0, atol=1e-6).sum()))
    _check(r, "ac_not_above_dc", bool((ac <= dc + RATING_TOLERANCE_W).all()), max_ac_minus_dc_w=float((ac - dc).max()))

    # Independent formula check (hand-written PVWatts DC and inverter equations vs pvlib).
    poa, tcell = sim["poa_global_wm2"].to_numpy(), sim["temp_cell_c"].to_numpy()
    manual_dc = np.maximum(house.array_stc_power_w * poa / 1000.0 * (1 + house.gamma_pdc_per_c * (tcell - physics.temp_ref_c)), 0.0)
    manual_dc *= (1.0 - physics.dc_loss_fraction)
    n, eta = house.inverter_unit_count, house.inverter_nominal_efficiency
    pdc0 = house.inverter_ac_power_w_per_unit / eta
    zeta = (manual_dc / n) / pdc0
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = eta / physics.inverter_eta_inv_ref * (-0.0162 * zeta - np.where(zeta > 0, 0.0059 / zeta, 0.0) + 0.9858)
    manual_ac = n * np.maximum(np.minimum(eta * pdc0, eff * manual_dc / n), 0.0)
    dc_diff, ac_diff = float(np.max(np.abs(manual_dc - dc))), float(np.max(np.abs(manual_ac - ac)))
    _check(r, "hand_formula_matches_pvlib", dc_diff <= POWER_TOLERANCE_W and ac_diff <= POWER_TOLERANCE_W,
           max_abs_diff_dc_w=dc_diff, max_abs_diff_ac_w=ac_diff, tolerance_w=POWER_TOLERANCE_W)

    # Schema reduction must not change the physics: recompute from the exported file itself
    # (plus the source albedo, which is not an exported column).
    rebuilt = out.rename(columns={v: k for k, v in PASSTHROUGH.items()}).copy()
    rebuilt["Surface Albedo"] = weather.values["Surface Albedo"].to_numpy()
    reparsed_times = pd.DatetimeIndex(pd.to_datetime(out["Timestamp"], format=out_format))
    again = simulate(rebuilt, reparsed_times, house, physics)
    re_dc = float(np.max(np.abs(again["pdc_w"].to_numpy() - dc)))
    re_ac = float(np.max(np.abs(again["pac_w"].to_numpy() - ac)))
    _check(r, "recompute_from_exported_file", re_dc <= POWER_TOLERANCE_W and re_ac <= POWER_TOLERANCE_W,
           max_abs_diff_dc_w=re_dc, max_abs_diff_ac_w=re_ac)

    if reference_dir is not None:
        r["reference_comparison"] = compare_reference(reference_dir, path.name, out)
    return r


def compare_reference(reference_dir: Path, name: str, out: pd.DataFrame) -> dict[str, Any]:
    ref_path = reference_dir / name
    if not ref_path.is_file():
        return {"passed": None, "detail": f"no reference file {ref_path}"}
    ref = pd.read_csv(ref_path, float_precision="round_trip")
    result: dict[str, Any] = {"reference_file": str(ref_path), "reference_sha256": sha256_file(ref_path)}
    if "Timestamp" not in ref.columns or len(ref) != len(out) or not (ref["Timestamp"].astype(str).to_numpy() == out["Timestamp"].astype(str).to_numpy()).all():
        result.update(passed=False, detail="reference timestamps/rows differ; values not compared")
        return result
    ok = True
    for column in ("PV_Power_Generation_W", "PV_AC_Power_W"):
        if column not in ref.columns:
            result[column] = "absent in reference"
            continue
        diff = np.abs(ref[column].to_numpy(dtype="float64") - out[column].to_numpy(dtype="float64"))
        result[column] = {"max_abs_diff_w": float(np.nanmax(diff)), "records_over_tolerance": int((diff > POWER_TOLERANCE_W).sum())}
        ok &= bool((diff <= POWER_TOLERANCE_W).all())
    result["passed"] = ok
    result["tolerance_w"] = POWER_TOLERANCE_W
    return result


# --------------------------------------------------------------------------- energy summary (optional)


def energy_summary(house: House, times: pd.DatetimeIndex, sim: pd.DataFrame) -> list[dict[str, Any]]:
    """Trapezoidal interval energy, treating source times as instantaneous samples (convention unconfirmed).

    interval_energy_kwh = ((p[:-1] + p[1:]) / 2) * dt_h / 1000, with dt_h = 0.25 for the audited 15-minute
    grid. N rows give N - 1 complete intervals; nothing is added after the final endpoint. An interval is
    assigned to the calendar year of its start.
    """
    dt_h = np.diff(times.as_unit("ns").asi8) / 3.6e12
    energy = {}
    for label, column in (("DC", "pdc_w"), ("AC", "pac_w")):
        p = sim[column].to_numpy()
        energy[label] = (p[:-1] + p[1:]) / 2 * dt_h / 1000.0
    starts, ends = times[:-1], times[1:]
    first, last = times[0], times[-1]
    kwp = house.array_stc_power_w / 1000.0
    rows = []
    periods = [(str(y), np.asarray(starts.year == y),
                "complete calendar year" if first <= pd.Timestamp(y, 1, 1) and last >= pd.Timestamp(y + 1, 1, 1) else "partial")
               for y in sorted(set(starts.year))]
    periods.append(("all", np.ones(len(starts), dtype=bool), "whole file"))
    for period, sel, coverage in periods:
        rows.append({
            "House_ID": house.house_id,
            "Period": period,
            "Coverage": coverage,
            "First_Interval_Start": str(starts[sel][0]),
            "Last_Interval_End": str(ends[sel][-1]),
            "Intervals": int(sel.sum()),
            "DC_Energy_kWh": float(energy["DC"][sel].sum()),
            "AC_Energy_kWh": float(energy["AC"][sel].sum()),
            "AC_Specific_Yield_kWh_per_kWp": float(energy["AC"][sel].sum()) / kwp,
        })
    return rows


# --------------------------------------------------------------------------- run


@dataclass
class HouseResult:
    audit: FileAudit
    checks: dict[str, Any] | None = None
    file_entry: dict[str, Any] | None = None
    energy_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[str]:
        return [k for k, v in (self.checks or {}).items() if isinstance(v, dict) and v.get("passed") is False]


def process_house(house: House, config: Config, input_dir: Path, output_dir: Path, *, reference_dir: Path | None = None,
                  audit_only: bool = False, with_energy: bool = False) -> HouseResult:
    """Audit one source file and, only if it is clean, simulate, write and validate that house."""
    src = input_dir / config.io["input_file_pattern"].format(house_id=house.house_id)
    audit, weather = audit_file(src, house.house_id, config)
    result = HouseResult(audit)
    status = "ok" if audit.ok else "BLOCKED"
    print(f"House {house.house_id:2d}: audit {status} — {src.name}"
          + (f" ({audit.data_rows} rows, {audit.first_timestamp} .. {audit.last_timestamp})" if audit.data_rows else ""))
    for finding in audit.defects:
        print(f"    DEFECT {finding.check} [{finding.column}] x{finding.count} {finding.detail} {finding.examples[:2]}")
    for finding in audit.warnings:
        print(f"    warning {finding.check} [{finding.column}] {finding.detail}")
    if not audit.ok or audit_only:
        return result

    out_format = config.io["output_timestamp_format"]
    sim = simulate(weather.values, weather.times, house, config.physics)
    frame = build_output(weather, house, sim, out_format)
    out_path = output_dir / "model_datasets" / config.io["output_file_pattern"].format(house_id=house.house_id)
    write_csv(frame, out_path)
    result.checks = validate_output(out_path, weather, house, sim, config.physics, out_format, reference_dir)
    result.file_entry = {
        "source": str(src), "source_sha256": audit.sha256, "source_rows": audit.data_rows,
        "output": str(out_path.relative_to(output_dir)), "output_sha256": sha256_file(out_path), "output_rows": len(frame),
    }
    failed = result.failed_checks
    print(f"           wrote {out_path.relative_to(output_dir)}; validation "
          + ("passed" if not failed else f"FAILED: {failed}")
          + f"; max AC {result.checks['ac_not_above_aggregate_rating']['max_ac_w']:.3f} W of {house.inverter_ac_rated_power_w:g} W")
    if with_energy:
        result.energy_rows = energy_summary(house, weather.times, sim)
    return result


def _house_record(house: House) -> dict[str, Any]:
    return {**asdict(house), "array_stc_power_w": house.array_stc_power_w,
            "inverter_ac_rated_power_w_aggregate": house.inverter_ac_rated_power_w}


def _common_metadata(config: Config, config_path: Path, env: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "command": sys.argv,
        "provenance": PROVENANCE_STATEMENT,
        "source_context": {
            "origin": "weather datasets collected in Spain (per the paper/user context)",
            "coordinates": "unconfirmed; not used (solar angles are taken from the source files)",
            "timezone_and_timestamp_semantics": "unconfirmed; timestamp labels are preserved without localisation or shifting",
            "units": config.raw.get("source_units"),
            "fill_flag": "audited only; not used as a predictor and not exported",
            "completeness_caveat": "a complete file does not prove the provider never filled earlier missing observations",
        },
        "environment": env,
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "physics": asdict(config.physics),
        "setting_provenance": config.raw.get("setting_provenance"),
        "final_columns": FINAL_COLUMNS,
        "removed_columns": list(REMOVED_COLUMNS),
    }


def _warn_versions(env: dict[str, Any]) -> None:
    if not env["all_match"]:
        print(f"WARNING: package versions differ from the reference environment: {env['actual']} vs {env['reference']}. "
              "Results are recorded with the actual versions; exact numerical equivalence is not claimed.")


def run_single_house(settings: Mapping[str, Any], input_dir: str | Path, output_dir: str | Path, *,
                     config_path: str | Path | None = None, energy_summary_csv: bool = True,
                     reference_dir: str | Path | None = None) -> int:
    """Run one house with the settings given in a house_XX.py file (used by the VS Code house scripts).

    Physics, audit rules and file layout come from simulation_config.json. Only this house's own
    earlier results in output_dir are replaced. Returns 0 when the audit and validation pass.
    """
    config_path = Path(config_path or HERE / "simulation_config.json").resolve()
    config = load_config(config_path)
    reference = config.houses.get(settings.get("house_id"))
    house = make_house(settings, default_notes=reference.notes if reference else ())
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        print(f"ERROR: INPUT_DIR {input_dir} does not exist; edit INPUT_DIR at the top of the house file", file=sys.stderr)
        return 2
    if output_dir == input_dir:
        print("ERROR: OUTPUT_DIR must differ from INPUT_DIR so the original files are never touched", file=sys.stderr)
        return 2
    tag = f"house_{house.house_id:02d}"
    meta_dir = output_dir / "metadata"
    stale = [output_dir / "model_datasets" / config.io["output_file_pattern"].format(house_id=house.house_id),
             output_dir / f"{tag}_energy_summary.csv", *meta_dir.glob(f"{tag}_*.json")]
    for path in stale:  # only this house's own earlier results
        if path.is_file():
            path.unlink()

    env = environment_versions()
    _warn_versions(env)
    differences = {}
    if reference is None:
        print(f"NOTE: house {house.house_id} is not in simulation_config.json; using the house file settings as given")
    else:
        for key in REQUIRED_HOUSE_KEYS:
            if getattr(house, key) != getattr(reference, key):
                differences[key] = {"house_file": getattr(house, key), "simulation_config": getattr(reference, key)}
        if differences:
            print(f"NOTE: edited scenario — settings differ from simulation_config.json: {differences}")

    result = process_house(house, config, input_dir, output_dir, reference_dir=Path(reference_dir).resolve() if reference_dir else None,
                           with_energy=energy_summary_csv)
    write_json(meta_dir / f"{tag}_audit.json", result.audit)
    if result.checks is not None:
        write_json(meta_dir / f"{tag}_validation.json", result.checks)
    if result.energy_rows:
        pd.DataFrame(result.energy_rows).to_csv(output_dir / f"{tag}_energy_summary.csv", index=False, lineterminator="\n")
    write_json(meta_dir / f"{tag}_run_metadata.json", {
        **_common_metadata(config, config_path, env),
        "run_mode": "single house (house file)",
        "house": _house_record(house),
        "settings_differ_from_simulation_config": differences,
        "file": result.file_entry,
        "energy_summary": f"{tag}_energy_summary.csv (trapezoidal; source times treated as instantaneous samples)"
        if result.energy_rows else None,
    })
    if not result.audit.ok:
        print(f"House {house.house_id} BLOCKED by audit defects; no dataset written. See {meta_dir / (tag + '_audit.json')}")
        return 1
    if result.failed_checks:
        print(f"House {house.house_id} validation FAILED: {result.failed_checks}. See {meta_dir / (tag + '_validation.json')}")
        return 1
    print(f"Done: {output_dir / result.file_entry['output']}")
    return 0



def environment_versions() -> dict[str, Any]:
    import scipy

    actual = {
        "python": platform.python_version(),
        "pvlib": pvlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    matches = {k: (actual[k].startswith(v + ".") or actual[k] == v) if k == "python" else actual[k] == v
               for k, v in REFERENCE_VERSIONS.items()}
    return {"actual": actual, "reference": REFERENCE_VERSIONS, "matches_reference": matches,
            "all_match": all(matches.values()), "platform": platform.platform()}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (Finding, FileAudit)):
        d = asdict(obj)
        if isinstance(obj, FileAudit):
            d["ok"] = obj.ok
        return d
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_jsonable, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        print(f"ERROR: --input-dir {input_dir} does not exist", file=sys.stderr)
        return 2
    if output_dir == input_dir:
        print("ERROR: --output-dir must differ from --input-dir so the original files are never touched", file=sys.stderr)
        return 2
    generated = [output_dir / "model_datasets", output_dir / "metadata", output_dir / "energy_summary.csv",
                 output_dir / "pv_model_datasets.zip"]
    if any(p.exists() for p in generated):
        if not args.overwrite:
            print(f"ERROR: {output_dir} already contains results; use a new --output-dir or pass --overwrite "
                  "(stale files from an earlier run must never be mistaken for current output)", file=sys.stderr)
            return 2
        for p in generated:  # only this script's own generated artifacts are removed
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    houses = [int(h) for h in args.houses.split(",")] if args.houses else sorted(config.houses)
    unknown = [h for h in houses if h not in config.houses]
    if unknown:
        print(f"ERROR: houses {unknown} are not in the configuration", file=sys.stderr)
        return 2

    env = environment_versions()
    _warn_versions(env)

    meta_dir = output_dir / "metadata"
    audits: dict[int, FileAudit] = {}
    validations: dict[int, Any] = {}
    files: dict[int, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None

    for house_id in houses:
        result = process_house(config.houses[house_id], config, input_dir, output_dir, reference_dir=reference_dir,
                               audit_only=args.audit_only, with_energy=args.energy_summary)
        audits[house_id] = result.audit
        if result.checks is not None:
            validations[house_id] = result.checks
            files[house_id] = result.file_entry
        summary_rows.extend(result.energy_rows)

    blocked = sorted(h for h, a in audits.items() if not a.ok)
    failed_validation = sorted(h for h, v in validations.items()
                               if any(isinstance(c, dict) and c.get("passed") is False for c in v.values()))

    write_json(meta_dir / "audit_report.json", {str(h): a for h, a in audits.items()})
    write_json(meta_dir / "validation_report.json", {str(h): v for h, v in validations.items()})
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "energy_summary.csv", index=False, lineterminator="\n")
    total_rows = sum(a.data_rows or 0 for a in audits.values())
    write_json(meta_dir / "run_metadata.json", {
        **_common_metadata(config, config_path, env),
        "run_mode": "all selected houses",
        "houses": {str(h): _house_record(config.houses[h]) for h in houses},
        "files": {str(h): f for h, f in files.items()},
        "records_audited": total_rows,
        "blocked_houses": blocked,
        "validation_failed_houses": failed_validation,
        "energy_summary": "energy_summary.csv (trapezoidal; source times treated as instantaneous samples)" if summary_rows else None,
    })

    for name in ("README.md", "schema.json", "requirements.txt"):
        if (HERE / name).is_file():
            shutil.copy2(HERE / name, output_dir / name)
    shutil.copy2(Path(__file__).resolve(), output_dir / "simulate_pv.py")
    if config_path != (output_dir / "simulation_config.json"):
        shutil.copy2(config_path, output_dir / "simulation_config.json")

    if args.zip and files:
        zip_path = output_dir / "pv_model_datasets.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file() and path != zip_path:
                    zf.write(path, path.relative_to(output_dir).as_posix())
        print(f"ZIP: {zip_path}")

    print(f"Audited {len(audits)} file(s), {total_rows} records; generated {len(files)} dataset(s).")
    if blocked:
        print(f"Blocked by audit defects (no output generated): houses {blocked}. See metadata/audit_report.json.")
    if failed_validation:
        print(f"Validation failures: houses {failed_validation}. See metadata/validation_report.json.")
    return 1 if (blocked or failed_validation) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Historical PV available-power simulation (pvlib) for the Weather House CSVs")
    parser.add_argument("--input-dir", required=True, help="folder containing 'Weather House N.csv' (read only)")
    parser.add_argument("--output-dir", default="pv_simulation_output", help="new folder for outputs (default: %(default)s)")
    parser.add_argument("--config", default=str(HERE / "simulation_config.json"), help="editable settings JSON")
    parser.add_argument("--houses", help="comma-separated subset, e.g. 1,4,8 (default: all 13)")
    parser.add_argument("--energy-summary", action="store_true", help="also write energy_summary.csv (separate from datasets)")
    parser.add_argument("--reference-dir", help="folder with earlier house_XX_model_dataset.csv files to compare DC/AC against")
    parser.add_argument("--audit-only", action="store_true", help="audit the source files without simulating")
    parser.add_argument("--zip", action="store_true", help="bundle datasets, code, config, schema and metadata into a ZIP")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace results from an earlier run in --output-dir (removes its model_datasets/, metadata/, "
                             "energy_summary.csv and ZIP first)")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
