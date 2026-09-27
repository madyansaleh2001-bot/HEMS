from datetime import date

import pandas as pd
import pytest

from pvforecast.accuracy import FORECAST_SELECTION_RULE, ScoreStatus
from pvforecast.config import MeasurementBoundary
from pvforecast.products import forecast_today
from pvforecast.timegrid import IntervalStatus
from pvforecast.view import ActualInterval, compose_today_view

from conftest import TEST_PLACE, hourly_snapshot, make_installation

DAY = date(2026, 9, 27)
TZ = TEST_PLACE.timezone


def at(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY} {hhmm}").tz_localize(TZ).tz_convert("UTC")


@pytest.fixture
def runs():
    inst = make_installation()
    early = forecast_today(TEST_PLACE, inst, hourly_snapshot(TEST_PLACE, DAY, clearness=0.8), at("05:00"))
    midday = forecast_today(TEST_PLACE, inst, hourly_snapshot(TEST_PLACE, DAY, clearness=0.5), at("13:05"))
    return inst, early, midday


def test_without_measurements_completed_rows_await_data(runs):
    inst, early, midday = runs
    view = compose_today_view(inst, [early, midday], at("13:10"))
    completed = [r for r in view.rows if r.status is IntervalStatus.COMPLETED]
    assert completed and all(r.actual_kw is None and r.actual_note == "Awaiting actual data" for r in completed)
    assert all(r.score.accuracy_percent is None for r in view.rows)
    assert view.summary.agreement_percent is None
    assert view.summary.paired_intervals == 0 and view.summary.expected_intervals == len(completed)


def test_midday_refresh_keeps_issued_forecasts_for_elapsed_rows(runs):
    inst, early, midday = runs
    view = compose_today_view(inst, [early, midday], at("13:10"))
    assert len(view.rows) == len(early.grid.intervals)
    for row in view.rows:
        if row.status is IntervalStatus.FUTURE:
            assert row.forecast_issued_at_utc == midday.issued_at_utc
        else:  # completed and in-progress rows keep the forecast issued before they started
            assert row.forecast_issued_at_utc == early.issued_at_utc
            assert row.forecast_note == FORECAST_SELECTION_RULE
    future = next(r for r in view.rows if r.status is IntervalStatus.FUTURE)
    early_value = next(r for r in early.rows if r.interval.start == future.interval.start).forecast_mean_kw
    assert future.forecast_kw != pytest.approx(early_value)  # future rows are updated


def test_first_run_after_sunrise_has_no_reconstructed_past_forecast():
    inst = make_installation()
    only = forecast_today(TEST_PLACE, inst, hourly_snapshot(TEST_PLACE, DAY), at("13:05"))
    view = compose_today_view(inst, [only], at("13:10"))
    past = [r for r in view.rows if r.status is IntervalStatus.COMPLETED]
    assert past and all(r.forecast_kw is None for r in past)
    assert all(r.forecast_note == ScoreStatus.NO_FORECAST.value for r in past)
    in_progress = next(r for r in view.rows if r.status is IntervalStatus.IN_PROGRESS)
    assert "not an advance forecast" in in_progress.forecast_note


def test_measurements_are_scored_at_the_same_boundary_only(runs):
    inst, early, midday = runs
    iv = [r.interval for r in early.rows]
    f = {r.interval.start: r.forecast_mean_kw for r in early.rows}
    full = iv[20]
    actuals = [
        ActualInterval(full.start, full.end, MeasurementBoundary.PV_DC_INPUT, f[full.start] * 0.25 * 1.05, 1.0, "test"),
        ActualInterval(iv[21].start, iv[21].end, MeasurementBoundary.PV_DC_INPUT, 0.3, 0.6, "test"),
        ActualInterval(iv[22].start, iv[22].end, MeasurementBoundary.SOLAR_AC_OUTPUT, 0.3, 1.0, "test"),
    ]
    view = compose_today_view(inst, [early, midday], at("13:10"), actuals)
    rows = {r.interval.start: r for r in view.rows}
    scored = rows[full.start]
    assert scored.score.status is ScoreStatus.SCORED
    assert scored.actual_kw == pytest.approx(f[full.start] * 1.05)
    assert scored.score.deviation_percent == pytest.approx(100 * 0.05 / 1.05)
    assert rows[iv[21].start].score.status is ScoreStatus.INSUFFICIENT_DATA
    assert rows[iv[21].start].actual_kw is None  # partial coverage is not extrapolated
    assert rows[iv[22].start].score.status is ScoreStatus.BOUNDARY_MISMATCH
    s = view.summary
    assert s.paired_intervals == 1
    assert s.exclusions[ScoreStatus.INSUFFICIENT_DATA.value] == 1
    assert s.exclusions[ScoreStatus.BOUNDARY_MISMATCH.value] == 1
    assert s.agreement_percent == pytest.approx(100 - 100 * 0.05 / 1.05)
