"""Today's table as displayed: saved forecasts beside actual generation.

Composition rules:
  * the full daylight table stays visible after every refresh;
  * future rows show the latest run's prediction;
  * completed rows show the comparison forecast — the latest saved run issued at
    or before the interval start — and never a value recomputed afterwards;
  * actual values come only from genuine measurements (production-log uploads or
    verified inverter telemetry) at the same measurement boundary; until then the
    row reads "Awaiting actual data".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from .accuracy import (
    FORECAST_SELECTION_RULE,
    LowOutputThreshold,
    PairedInterval,
    RowScore,
    ScoreStatus,
    SummaryScore,
    low_output_threshold,
    score_interval,
    summarize,
)
from .config import Installation, MeasurementBoundary
from .products import DaylightForecastRun, ForecastRow
from .timegrid import DaylightGrid, Interval, IntervalStatus

# Measured coverage below this fraction of an interval is not treated as a complete actual value.
MIN_ACTUAL_COVERAGE = 0.999


@dataclass(frozen=True)
class ActualInterval:
    """Measured generation aggregated onto one grid interval (time-weighted)."""

    interval_start: pd.Timestamp
    interval_end: pd.Timestamp
    measurement_boundary: MeasurementBoundary
    energy_kwh: float
    coverage_fraction: float  # share of the interval covered by valid measurements
    source: str  # e.g. "upload:<file name>" or "inverter:<protocol profile>"
    quality_flags: tuple[str, ...] = ()

    @property
    def duration_hours(self) -> float:
        return (self.interval_end - self.interval_start) / pd.Timedelta(hours=1)

    @property
    def mean_kw(self) -> float:
        return self.energy_kwh / self.duration_hours


@dataclass(frozen=True)
class TodayViewRow:
    interval: Interval
    status: IntervalStatus
    forecast_kw: float | None
    forecast_kwh: float | None
    forecast_issued_at_utc: pd.Timestamp | None
    forecast_note: str
    actual_kw: float | None
    actual_kwh: float | None
    actual_note: str
    actual_source: str | None
    score: RowScore


@dataclass(frozen=True)
class TodayView:
    grid: DaylightGrid
    measurement_boundary: MeasurementBoundary
    latest_run: DaylightForecastRun
    rows: tuple[TodayViewRow, ...]
    summary: SummaryScore
    threshold: LowOutputThreshold


def _row_of(run: DaylightForecastRun, interval: Interval) -> ForecastRow | None:
    for row in run.rows:
        if row.interval.start == interval.start:
            return row
    return None


def comparison_forecast(runs: Sequence[DaylightForecastRun], interval: Interval) -> tuple[DaylightForecastRun, ForecastRow] | None:
    """The latest saved run issued at or before the interval start that has a value for it."""
    best = None
    for run in runs:
        row = _row_of(run, interval)
        if row is None or not row.eligible_as_advance_forecast or run.issued_at_utc > interval.start:
            continue
        if best is None or run.issued_at_utc > best[0].issued_at_utc:
            best = (run, row)
    return best


def compose_today_view(
    installation: Installation,
    runs: Iterable[DaylightForecastRun],
    now_utc: pd.Timestamp,
    actuals: Iterable[ActualInterval] = (),
    measurement_floor_kw: float | None = None,
) -> TodayView:
    now_utc = pd.Timestamp(now_utc).tz_convert("UTC")
    runs = [
        r for r in runs
        if r.installation_id == installation.installation_id
        and r.measurement_boundary is installation.measurement_boundary
        and r.issued_at_utc <= now_utc
    ]
    if not runs:
        raise ValueError("no forecast run for this installation issued before now")
    latest = max(runs, key=lambda r: r.issued_at_utc)
    runs = [r for r in runs if r.target_date == latest.target_date]
    threshold = low_output_threshold(installation, measurement_floor_kw)
    by_start = {pd.Timestamp(a.interval_start).value: a for a in actuals}

    rows: list[TodayViewRow] = []
    pairs: list[PairedInterval] = []
    exclusions: Counter[str] = Counter()
    expected = 0
    for interval in latest.grid.intervals:
        status = interval.status_at(now_utc)
        f_kw = f_kwh = issued = None
        a_kw = a_kwh = a_source = None
        a_note = ""
        if status is IntervalStatus.FUTURE:
            row = _row_of(latest, interval)
            f_kw, f_kwh, issued = row.forecast_mean_kw, row.forecast_energy_kwh, latest.issued_at_utc
            f_note = "latest run" if f_kw is not None else "weather unavailable"
            score = RowScore(ScoreStatus.NOT_COMPLETED)
        else:
            chosen = comparison_forecast(runs, interval)
            if chosen is not None:
                f_kw, f_kwh, issued = chosen[1].forecast_mean_kw, chosen[1].forecast_energy_kwh, chosen[0].issued_at_utc
                f_note = FORECAST_SELECTION_RULE
            elif status is IntervalStatus.IN_PROGRESS and _row_of(latest, interval).forecast_mean_kw is not None:
                row = _row_of(latest, interval)
                f_kw, f_kwh, issued = row.forecast_mean_kw, row.forecast_energy_kwh, latest.issued_at_utc
                f_note = "issued after interval start (in progress) — not an advance forecast"
            else:
                f_note = ScoreStatus.NO_FORECAST.value

            if status is IntervalStatus.IN_PROGRESS:
                a_note = "Interval in progress"
                score = RowScore(ScoreStatus.NOT_COMPLETED)
            else:
                expected += 1
                actual = by_start.get(interval.start.value)
                if actual is None:
                    a_note = ScoreStatus.AWAITING_DATA.value
                    score = RowScore(ScoreStatus.AWAITING_DATA)
                elif actual.measurement_boundary is not installation.measurement_boundary:
                    a_note = ScoreStatus.BOUNDARY_MISMATCH.value
                    score = RowScore(ScoreStatus.BOUNDARY_MISMATCH)
                elif actual.coverage_fraction < MIN_ACTUAL_COVERAGE:
                    a_note = f"{ScoreStatus.INSUFFICIENT_DATA.value} ({actual.coverage_fraction:.0%} coverage)"
                    a_source = actual.source
                    score = RowScore(ScoreStatus.INSUFFICIENT_DATA)
                else:
                    a_kw, a_kwh, a_source = actual.mean_kw, actual.energy_kwh, actual.source
                    a_note = "measured"
                    score = score_interval(f_kw, a_kw, threshold.kw, interval.duration_hours)
                if score.status in (ScoreStatus.SCORED, ScoreStatus.LOW_OUTPUT):
                    pairs.append(PairedInterval(f_kw, a_kw, interval.duration_hours))
                else:
                    exclusions[score.status.value] += 1
        rows.append(TodayViewRow(
            interval, status, f_kw, f_kwh, issued, f_note, a_kw, a_kwh, a_note, a_source, score,
        ))

    summary = summarize(pairs, expected, threshold.kw, exclusions=dict(exclusions), label="Today so far")
    return TodayView(latest.grid, installation.measurement_boundary, latest, tuple(rows), summary, threshold)
