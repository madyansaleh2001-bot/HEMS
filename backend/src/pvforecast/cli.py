"""Command-line interface for the forecast core.

    pvforecast grid     --config FILE [--date YYYY-MM-DD]
    pvforecast today    --config FILE WEATHER [--now TIME] [--csv FILE]
    pvforecast tomorrow --config FILE WEATHER [--now TIME]

WEATHER is one of:
    --clear-sky [--air-temp C] [--wind M/S]   simulated cloud-free scenario (not a forecast)
    --open-meteo [--save-weather FILE]        live Open-Meteo request
    --weather-json FILE                       a response saved with --save-weather
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import openmeteo
from .accuracy import AGREEMENT_EXPLANATION, FORECAST_SELECTION_RULE, ScoreStatus
from .config import Installation, Place, load_config
from .products import forecast_today, forecast_tomorrow
from .sun import daylight_grid
from .timegrid import DaylightGrid, DayType, IntervalStatus, local_day_bounds
from .view import TodayView, compose_today_view
from .weather import WeatherSnapshot, clear_sky_snapshot

DASH = "—"


def _parse_now(value: str | None, place: Place) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(place.timezone)
    return ts.tz_convert("UTC")


def _weather(args: argparse.Namespace, place: Place, now_utc: pd.Timestamp) -> WeatherSnapshot:
    if args.clear_sky:
        today = now_utc.tz_convert(place.timezone).date()
        start, _ = local_day_bounds(today - timedelta(days=1), place.timezone)
        _, end = local_day_bounds(today + timedelta(days=2), place.timezone)
        return clear_sky_snapshot(
            place, start.tz_convert("UTC"), end.tz_convert("UTC"), args.air_temp, args.wind, retrieved_at_utc=now_utc
        )
    if args.weather_json:
        saved = json.loads(Path(args.weather_json).read_text(encoding="utf-8"))
        return openmeteo.parse_forecast(
            saved["response"], pd.Timestamp(saved["retrieved_at_utc"]), place, saved.get("model")
        )
    snapshot = openmeteo.fetch_forecast(place)
    if args.save_weather:
        Path(args.save_weather).write_text(json.dumps({
            "retrieved_at_utc": snapshot.retrieved_at_utc.isoformat(),
            "model": snapshot.model,
            "response": snapshot.raw,
        }), encoding="utf-8")
    return snapshot


def _fmt(value: float | None, digits: int = 3) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _describe_grid(grid: DaylightGrid) -> str:
    if grid.day_type is DayType.POLAR_NIGHT:
        return "Polar night: the sun does not rise; no daylight rows."
    if grid.day_type is DayType.POLAR_DAY:
        return f"Polar day: the sun does not set; {len(grid.intervals)} rows cover the whole local day."
    return (
        f"Actual sunrise {grid.sunrise:%H:%M:%S}, sunset {grid.sunset:%H:%M:%S} ({grid.timezone}) -> "
        f"table {grid.table_start:%H:%M}–{grid.table_end:%H:%M}, {len(grid.intervals)} rows of 15 min, "
        "each labelled by its end time"
    )


def _header(place: Place, installation: Installation, snapshot_desc: dict, simulated: bool) -> list[str]:
    lines = []
    if simulated:
        lines.append("*** " + snapshot_desc["description"] + " ***")
    lines += [
        f"Place: {place.name} ({place.latitude:.4f}, {place.longitude:.4f}, {place.timezone})",
        f"Installation: {installation.name} [{installation.installation_id} v{installation.config_version}], "
        f"{installation.stc_capacity_w / 1000:.3f} kWp STC, inverter {installation.inverter.make} "
        f"{installation.inverter.model} ({installation.inverter.ac_rated_w / 1000:g} kW AC)",
        f"Weather: {snapshot_desc['provider']} / {snapshot_desc['model']}, retrieved {snapshot_desc['retrieved_at_utc']}, "
        f"upstream issue time {snapshot_desc['issued_at_utc'] or 'not supplied by provider'}, "
        f"native resolution {snapshot_desc['native_resolution_minutes']} min",
    ]
    return lines


def _print_today(view: TodayView, place: Place, installation: Installation) -> None:
    run = view.latest_run
    tz = place.timezone
    out = _header(place, installation, run.weather, run.is_simulation)
    out += [
        f"Target date: {run.target_date} | issued {run.issued_at_utc.tz_convert(tz):%Y-%m-%d %H:%M %Z}",
        _describe_grid(view.grid),
        f"Measurement boundary: {view.measurement_boundary.value} | Model: {run.model_status}",
        "",
        f"{'End':>5}  {'Interval':<11}  {'Status':<11}  {'Fcst kW':>8}  {'Fcst kWh':>8}  "
        f"{'Actual kW':>9}  {'Dev kW':>7}  {'Acc %':>6}  Notes",
    ]
    for r in view.rows:
        iv = r.interval
        acc = r.score.accuracy_percent
        notes = []
        if r.forecast_note not in ("latest run", FORECAST_SELECTION_RULE):
            notes.append(r.forecast_note)
        if r.status is IntervalStatus.COMPLETED:
            notes.append(r.actual_note)
        if r.score.status is ScoreStatus.LOW_OUTPUT:
            notes.append(ScoreStatus.LOW_OUTPUT.value)
        note = "; ".join(notes)
        out.append(
            f"{iv.label:>5}  {iv.start:%H:%M}–{iv.end:%H:%M}  {r.status.value:<11}  {_fmt(r.forecast_kw):>8}  "
            f"{_fmt(r.forecast_kwh):>8}  {_fmt(r.actual_kw):>9}  {_fmt(r.score.deviation_kw):>7}  "
            f"{_fmt(acc, 1):>6}  {note}"
        )
    pre = view.grid.pre_table_segment
    out.append("")
    if pre:
        out.append(
            f"Before the rounded start ({pre[0]:%H:%M:%S}–{pre[1]:%H:%M}): {_fmt(run.pre_table_energy_kwh)} kWh (not in table)"
        )
    out.append(
        f"Modelled energy with this run's weather — table: {_fmt(run.table_energy_kwh)} kWh, "
        f"full day incl. pre-table segment: {_fmt(run.full_day_energy_kwh)} kWh"
    )
    s = view.summary
    if s.agreement_percent is None:
        out.append(
            f"{s.label}: {s.paired_intervals}/{s.expected_intervals} completed intervals paired with measurements "
            "— accuracy unavailable (Awaiting actual data)."
        )
    else:
        out.append(
            f"{s.label}: agreement {s.agreement_percent:.1f} % (WAPE {s.wape_percent:.1f} %), "
            f"MAE {s.mae_kw:.3f} kW, RMSE {s.rmse_kw:.3f} kW, mean deviation {s.mean_deviation_kw:+.3f} kW, "
            f"coverage {s.paired_intervals}/{s.expected_intervals}. {AGREEMENT_EXPLANATION}"
        )
    out.append(f"Low-output threshold for row percentages: {view.threshold.kw:.3f} kW ({view.threshold.note})")
    out += ["", "Notes:"] + [f"  {n}" for n in run.notes]
    print("\n".join(out))


def _write_csv(view: TodayView, path: str) -> None:
    fields = [
        "interval_start", "interval_end", "label", "status", "measurement_boundary",
        "forecast_mean_kw", "forecast_energy_kwh", "forecast_issued_at_utc", "forecast_note",
        "actual_mean_kw", "actual_energy_kwh", "actual_note", "actual_source",
        "deviation_kw", "deviation_percent", "absolute_percentage_error", "accuracy_percent", "score_status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in view.rows:
            s = r.score
            w.writerow([
                r.interval.start.isoformat(), r.interval.end.isoformat(), r.interval.label, r.status.value,
                view.measurement_boundary.value, r.forecast_kw, r.forecast_kwh,
                r.forecast_issued_at_utc.isoformat() if r.forecast_issued_at_utc is not None else None,
                r.forecast_note, r.actual_kw, r.actual_kwh, r.actual_note, r.actual_source,
                s.deviation_kw, s.deviation_percent, s.absolute_percentage_error, s.accuracy_percent, s.status.value,
            ])


def _print_tomorrow(summary, place: Place, installation: Installation) -> None:
    tz = place.timezone
    out = _header(place, installation, summary.weather, summary.is_simulation)
    out += [
        f"Tomorrow: {summary.target_date} | {_describe_grid(summary.grid)}",
        f"Measurement boundary: {summary.measurement_boundary.value} (unconstrained potential) | Model: {summary.model_status}",
        f"Daylight duration (actual sunrise to sunset): {summary.daylight_hours:.2f} h",
        f"Daylight-average potential power: {_fmt(summary.average_kw)} kW",
        f"Expected daylight energy: {_fmt(summary.energy_kwh)} kWh",
    ]
    if summary.peak_interval is not None:
        out.append(
            f"Peak 15-minute mean: {summary.peak_kw:.3f} kW, {summary.peak_interval.start:%H:%M}–{summary.peak_interval.end:%H:%M}"
        )
    out += [
        f"Last successful update: {summary.issued_at_utc.tz_convert(tz):%Y-%m-%d %H:%M %Z}; "
        f"next update due: {summary.next_update_due_utc.tz_convert(tz):%Y-%m-%d %H:%M %Z} (every 2 h)",
        "", "Notes:",
    ] + [f"  {n}" for n in summary.notes]
    print("\n".join(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pvforecast", description="PV forecast core (physics baseline)")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("grid", help="show sunrise/sunset and the quarter-hour table grid")
    g.add_argument("--config", required=True)
    g.add_argument("--date", help="local date (default: today at the installation)")

    for name in ("today", "tomorrow"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--now", help="issue time, ISO 8601; without offset it is read in the installation time zone")
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--clear-sky", action="store_true", help="simulated cloud-free scenario (not a forecast)")
        src.add_argument("--open-meteo", action="store_true", help="fetch the Open-Meteo forecast")
        src.add_argument("--weather-json", help="saved Open-Meteo response (see --save-weather)")
        p.add_argument("--air-temp", type=float, default=25.0, help="clear-sky scenario air temperature, °C")
        p.add_argument("--wind", type=float, default=1.0, help="clear-sky scenario 10 m wind speed, m/s")
        p.add_argument("--save-weather", help="with --open-meteo: archive the raw response to this file")
        if name == "today":
            p.add_argument("--csv", help="also write the table to this CSV file")

    args = parser.parse_args(argv)
    place, installation = load_config(args.config)

    if args.command == "grid":
        local = date.fromisoformat(args.date) if args.date else pd.Timestamp.now(tz=place.timezone).date()
        grid = daylight_grid(place, local)
        print(f"{local} at {place.name}: {_describe_grid(grid)}")
        for iv in grid.intervals:
            print(f"  {iv.index:3d}  {iv.label}  ({iv.start:%H:%M}–{iv.end:%H:%M})")
        return 0

    now_utc = _parse_now(args.now, place)
    snapshot = _weather(args, place, now_utc)
    if args.command == "today":
        run = forecast_today(place, installation, snapshot, now_utc)
        view = compose_today_view(installation, [run], now_utc)
        _print_today(view, place, installation)
        if args.csv:
            _write_csv(view, args.csv)
    else:
        _print_tomorrow(forecast_tomorrow(place, installation, snapshot, now_utc), place, installation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
