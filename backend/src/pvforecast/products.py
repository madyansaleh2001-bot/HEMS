"""Forecast products: today's sunrise-to-sunset table and the tomorrow summary.

A ``DaylightForecastRun`` is an immutable record of one refresh. Rows that had
already finished when the run was issued get no forecast value: a past
prediction is never reconstructed from newer weather. A row that was in
progress at issue time keeps its value but is flagged, and it is not eligible
as the comparison forecast for that interval (see ``view.py``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import Installation, MeasurementBoundary, ModelNote, Place
from .physics import PHYSICAL_MODEL_VERSION, PowerSeries, integrate_energy_kwh, model_power
from .sun import daylight_grid
from .timegrid import DaylightGrid, Interval, IntervalStatus, local_day_bounds
from .weather import WeatherSnapshot

MODEL_STATUS_PHYSICS_ONLY = "Physics model — no trained AI correction yet"
TOMORROW_REFRESH = pd.Timedelta(hours=2)

_BOUNDARY_COLUMN = {
    MeasurementBoundary.PV_DC_INPUT: "pv_dc_w",
    MeasurementBoundary.SOLAR_AC_OUTPUT: "solar_ac_w_est",
}


@dataclass(frozen=True)
class ForecastRow:
    interval: Interval
    status_at_issue: IntervalStatus
    forecast_mean_kw: float | None  # at the run's measurement boundary; None = no forecast
    forecast_energy_kwh: float | None
    solar_ac_mean_kw_est: float | None  # modelled estimate, separate column
    ghi_wm2: float | None
    poa_wm2_est: float | None
    cell_temp_c_est: float | None
    air_temp_c: float | None
    wind_speed_ms: float | None
    cloud_cover_pct: float | None
    precipitation_mm: float | None
    flags: tuple[str, ...] = ()

    @property
    def eligible_as_advance_forecast(self) -> bool:
        """Issued at or before the interval start and carrying a value."""
        return self.status_at_issue is IntervalStatus.FUTURE and self.forecast_mean_kw is not None


@dataclass(frozen=True)
class DaylightForecastRun:
    forecast_id: str
    place_id: str
    installation_id: str
    config_version: int
    measurement_boundary: MeasurementBoundary
    issued_at_utc: pd.Timestamp
    grid: DaylightGrid
    rows: tuple[ForecastRow, ...]
    # Energies below come from this run's weather over the whole day, including
    # intervals that had already elapsed; they are model totals, not advance forecasts.
    pre_table_energy_kwh: float | None
    table_energy_kwh: float | None
    full_day_energy_kwh: float | None
    weather: dict[str, Any]
    physical_model_version: str = PHYSICAL_MODEL_VERSION
    correction_model_version: str | None = None
    model_status: str = MODEL_STATUS_PHYSICS_ONLY
    is_simulation: bool = False
    notes: tuple[ModelNote, ...] = field(default_factory=tuple)

    @property
    def target_date(self) -> date:
        return self.grid.local_date


@dataclass(frozen=True)
class TomorrowSummary:
    place_id: str
    installation_id: str
    config_version: int
    measurement_boundary: MeasurementBoundary
    target_date: date
    grid: DaylightGrid  # actual sunrise/sunset plus rounded display boundaries
    daylight_hours: float
    energy_kwh: float | None
    average_kw: float | None  # energy / actual daylight duration
    peak_kw: float | None  # highest 15-minute mean
    peak_interval: Interval | None
    issued_at_utc: pd.Timestamp
    next_update_due_utc: pd.Timestamp
    weather: dict[str, Any]
    physical_model_version: str = PHYSICAL_MODEL_VERSION
    model_status: str = MODEL_STATUS_PHYSICS_ONLY
    is_simulation: bool = False
    notes: tuple[ModelNote, ...] = field(default_factory=tuple)


def _local_date(place: Place, instant_utc: pd.Timestamp) -> date:
    return pd.Timestamp(instant_utc).tz_convert(place.timezone).date()


def _model_day(
    place: Place, installation: Installation, snapshot: WeatherSnapshot, local_date: date
) -> tuple[DaylightGrid, PowerSeries]:
    grid = daylight_grid(place, local_date)
    day_start, day_end = local_day_bounds(local_date, place.timezone)
    start = min(day_start, grid.table_start) if grid.table_start is not None else day_start
    end = max(day_end, grid.table_end) if grid.table_end is not None else day_end
    spans = [grid.daylight_span] if grid.daylight_span else []
    series = model_power(place, installation, snapshot, start.tz_convert("UTC"), end.tz_convert("UTC"), spans)
    return grid, series


def _interval_means(fine: pd.DataFrame, intervals: tuple[Interval, ...]) -> pd.DataFrame:
    """Mean of each column per 15-minute interval; NaN if any minute is missing."""
    if not intervals:
        return pd.DataFrame(columns=fine.columns)
    starts = np.array([iv.start.value for iv in intervals])
    ends = np.array([iv.end.value for iv in intervals])
    t = fine.index.as_unit("ns").asi8
    row = np.searchsorted(starts, t, side="right") - 1
    inside = (row >= 0) & (t < ends[np.clip(row, 0, None)])
    sub = fine.loc[inside].copy()
    sub["_row"] = row[inside]
    numeric = sub.drop(columns=["daylight"], errors="ignore")
    grouped = numeric.groupby("_row")
    incomplete = grouped.count().lt(grouped.size(), axis=0)
    means = grouped.mean()
    if "precipitation_mm" in means.columns:  # accumulated variable: sum, not mean
        means["precipitation_mm"] = grouped["precipitation_mm"].sum()
    return means.mask(incomplete).reindex(range(len(intervals)))


def _opt(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def forecast_today(
    place: Place,
    installation: Installation,
    snapshot: WeatherSnapshot,
    now_utc: pd.Timestamp,
    forecast_id: str | None = None,
) -> DaylightForecastRun:
    """Today's full daylight table (today in the installation's time zone)."""
    now_utc = pd.Timestamp(now_utc).tz_convert("UTC")
    target = _local_date(place, now_utc)
    grid, series = _model_day(place, installation, snapshot, target)
    fine = series.fine
    power_col = _BOUNDARY_COLUMN[installation.measurement_boundary]
    means = _interval_means(fine, grid.intervals)

    rows = []
    for i, interval in enumerate(grid.intervals):
        m = means.iloc[i]
        status = interval.status_at(now_utc)
        flags: list[str] = []
        mean_kw = None if pd.isna(m.get(power_col)) else float(m[power_col]) / 1000.0
        if mean_kw is None:
            flags.append("weather_unavailable")
        if status is IntervalStatus.COMPLETED:
            mean_kw = None
            flags = ["completed_before_issue"]
        elif status is IntervalStatus.IN_PROGRESS:
            flags.append("in_progress_at_issue")
        ac = m.get("solar_ac_w_est")
        rows.append(ForecastRow(
            interval=interval,
            status_at_issue=status,
            forecast_mean_kw=mean_kw,
            forecast_energy_kwh=None if mean_kw is None else mean_kw * interval.duration_hours,
            solar_ac_mean_kw_est=None if status is IntervalStatus.COMPLETED or pd.isna(ac) else float(ac) / 1000.0,
            ghi_wm2=_opt(m.get("ghi_wm2")),
            poa_wm2_est=_opt(m.get("poa_global_wm2_est")),
            cell_temp_c_est=_opt(m.get("cell_temp_c_est")),
            air_temp_c=_opt(m.get("air_temperature_c")),
            wind_speed_ms=_opt(m.get("wind_speed_ms")),
            cloud_cover_pct=_opt(m.get("cloud_cover_pct")),
            precipitation_mm=_opt(m.get("precipitation_mm")),
            flags=tuple(flags),
        ))

    pre = grid.pre_table_segment
    pre_kwh = integrate_energy_kwh(fine, power_col, *pre) if pre else 0.0
    table_kwh = integrate_energy_kwh(fine, power_col, grid.table_start, grid.table_end) if grid.intervals else 0.0
    full_kwh = None if pre_kwh is None or table_kwh is None else pre_kwh + table_kwh

    notes = list(series.notes)
    if pre:
        notes.append(ModelNote(
            "info", "table convention",
            f"rows start at the rounded sunrise {grid.table_start:%H:%M}; the {(pre[1] - pre[0]).total_seconds() / 60:.1f} "
            "minutes after actual sunrise are reported separately and included in the full-day total",
        ))
    if installation.measurement_boundary is MeasurementBoundary.PV_DC_INPUT:
        notes.append(ModelNote(
            "info", "measurement boundary",
            "forecast is unconstrained PV potential at the PV DC input; battery-full or low-load curtailment is not modelled",
        ))
    return DaylightForecastRun(
        forecast_id=forecast_id or str(uuid.uuid4()),
        place_id=place.place_id,
        installation_id=installation.installation_id,
        config_version=installation.config_version,
        measurement_boundary=installation.measurement_boundary,
        issued_at_utc=now_utc,
        grid=grid,
        rows=tuple(rows),
        pre_table_energy_kwh=pre_kwh,
        table_energy_kwh=table_kwh,
        full_day_energy_kwh=full_kwh,
        weather=snapshot.provenance(),
        is_simulation=snapshot.is_simulated,
        notes=tuple(dict.fromkeys(notes)),
    )


def forecast_tomorrow(
    place: Place, installation: Installation, snapshot: WeatherSnapshot, now_utc: pd.Timestamp
) -> TomorrowSummary:
    """Daylight-average potential power and expected energy for the next local date."""
    now_utc = pd.Timestamp(now_utc).tz_convert("UTC")
    target = _local_date(place, now_utc) + timedelta(days=1)
    grid, series = _model_day(place, installation, snapshot, target)
    power_col = _BOUNDARY_COLUMN[installation.measurement_boundary]
    span = grid.daylight_span
    energy = integrate_energy_kwh(series.fine, power_col, *span) if span else 0.0
    hours = grid.daylight_hours
    average = None if energy is None or hours == 0 else energy / hours

    means = _interval_means(series.fine, grid.intervals)
    peak_kw = peak_interval = None
    if grid.intervals and power_col in means and means[power_col].notna().all():
        i = int(means[power_col].to_numpy().argmax())
        peak_kw, peak_interval = float(means[power_col].iloc[i]) / 1000.0, grid.intervals[i]

    notes = list(series.notes)
    if energy is None:
        notes.append(ModelNote("warning", "weather", "weather data do not cover tomorrow's daylight; no summary value"))
    return TomorrowSummary(
        place_id=place.place_id,
        installation_id=installation.installation_id,
        config_version=installation.config_version,
        measurement_boundary=installation.measurement_boundary,
        target_date=target,
        grid=grid,
        daylight_hours=hours,
        energy_kwh=energy,
        average_kw=average,
        peak_kw=peak_kw,
        peak_interval=peak_interval,
        issued_at_utc=now_utc,
        next_update_due_utc=now_utc + TOMORROW_REFRESH,
        weather=snapshot.provenance(),
        is_simulation=snapshot.is_simulated,
        notes=tuple(dict.fromkeys(notes)),
    )
