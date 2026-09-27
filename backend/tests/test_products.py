from datetime import date

import pandas as pd
import pytest

from pvforecast.config import Place
from pvforecast.physics import integrate_energy_kwh
from pvforecast.products import MODEL_STATUS_PHYSICS_ONLY, forecast_today, forecast_tomorrow
from pvforecast.timegrid import IntervalStatus

from conftest import TEST_PLACE, hourly_snapshot, make_installation

DAY = date(2026, 9, 27)
TZ = TEST_PLACE.timezone


def at(hhmm: str, day: date = DAY) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}").tz_localize(TZ).tz_convert("UTC")


@pytest.fixture
def snap():
    return hourly_snapshot(TEST_PLACE, DAY)


def test_row_energy_is_mean_power_times_quarter_hour(snap):
    run = forecast_today(TEST_PLACE, make_installation(), snap, at("04:00"))
    assert run.rows and all(r.status_at_issue is IntervalStatus.FUTURE for r in run.rows)
    for r in run.rows:
        assert r.forecast_energy_kwh == pytest.approx(r.forecast_mean_kw * 0.25)
    assert sum(r.forecast_energy_kwh for r in run.rows) == pytest.approx(run.table_energy_kwh, rel=1e-9)


def test_full_day_total_includes_segment_before_rounded_start(snap):
    run = forecast_today(TEST_PLACE, make_installation(), snap, at("04:00"))
    grid = run.grid
    assert grid.pre_table_segment is not None
    assert run.full_day_energy_kwh == pytest.approx(run.pre_table_energy_kwh + run.table_energy_kwh)
    assert run.pre_table_energy_kwh >= 0.0
    tomorrow_of_yesterday = forecast_tomorrow(TEST_PLACE, make_installation(), snap, at("12:00", date(2026, 9, 26)))
    # the tomorrow gadget integrates actual sunrise -> sunset; that must equal pre-table + table energy
    assert tomorrow_of_yesterday.target_date == DAY
    assert tomorrow_of_yesterday.energy_kwh == pytest.approx(run.full_day_energy_kwh, rel=1e-9)


def test_refresh_time_does_not_move_the_grid(snap):
    inst = make_installation()
    early = forecast_today(TEST_PLACE, inst, snap, at("07:00"))
    midday = forecast_today(TEST_PLACE, inst, snap, at("13:05"))
    assert [r.interval.start for r in early.rows] == [r.interval.start for r in midday.rows]
    assert all(r.interval.start.minute % 15 == 0 for r in midday.rows)


def test_midday_refresh_keeps_full_table_without_rewriting_the_past(snap):
    run = forecast_today(TEST_PLACE, make_installation(), snap, at("13:05"))
    assert run.rows[0].interval.start == run.grid.table_start  # whole day still present
    done = [r for r in run.rows if r.status_at_issue is IntervalStatus.COMPLETED]
    assert done and all(r.forecast_mean_kw is None and "completed_before_issue" in r.flags for r in done)
    current = [r for r in run.rows if r.status_at_issue is IntervalStatus.IN_PROGRESS]
    assert len(current) == 1 and "in_progress_at_issue" in current[0].flags
    assert not current[0].eligible_as_advance_forecast
    future = [r for r in run.rows if r.status_at_issue is IntervalStatus.FUTURE]
    assert future and all(r.eligible_as_advance_forecast for r in future)
    assert run.model_status == MODEL_STATUS_PHYSICS_ONLY and run.correction_model_version is None


def test_missing_weather_marks_rows_unavailable(snap):
    snap.data.loc[pd.Timestamp("2026-09-27 19:00", tz="UTC"), "ghi_wm2"] = float("nan")  # 12:00-13:00 MDT
    run = forecast_today(TEST_PLACE, make_installation(), snap, at("04:00"))
    gap = [r for r in run.rows if at("12:00") <= r.interval.start < at("13:00")]
    assert len(gap) == 4
    assert all(r.forecast_mean_kw is None and "weather_unavailable" in r.flags for r in gap)
    assert run.table_energy_kwh is None and run.full_day_energy_kwh is None


def test_tomorrow_average_uses_actual_daylight_duration(snap):
    summary = forecast_tomorrow(TEST_PLACE, make_installation(), snap, at("13:05"))
    grid = summary.grid
    assert summary.target_date == date(2026, 9, 28)
    hours = (grid.sunset - grid.sunrise) / pd.Timedelta(hours=1)
    assert summary.daylight_hours == pytest.approx(hours)
    assert summary.average_kw == pytest.approx(summary.energy_kwh / hours)
    assert summary.average_kw < summary.peak_kw
    assert summary.next_update_due_utc - summary.issued_at_utc == pd.Timedelta(hours=2)


def test_today_and_tomorrow_follow_the_installation_time_zone():
    amman = Place("amman", "Test", 31.95, 35.93, "Asia/Amman")
    inst = make_installation(place=amman)
    snap = hourly_snapshot(amman, date(2026, 9, 28))
    now = pd.Timestamp("2026-09-27 22:30", tz="UTC")  # already 01:30 on the 28th in Amman
    assert forecast_today(amman, inst, snap, now).target_date == date(2026, 9, 28)
    assert forecast_tomorrow(amman, inst, snap, now).target_date == date(2026, 9, 29)


def test_polar_night_tomorrow_has_zero_energy_and_no_average():
    svalbard = Place("svalbard", "Longyearbyen", 78.22, 15.65, "Arctic/Longyearbyen")
    inst = make_installation(place=svalbard)
    snap = hourly_snapshot(svalbard, date(2026, 12, 21))
    summary = forecast_tomorrow(svalbard, inst, snap, pd.Timestamp("2026-12-20 12:00", tz="UTC"))
    assert summary.energy_kwh == 0.0 and summary.average_kw is None and summary.peak_kw is None


def test_solar_ac_boundary_forecast_is_capped_at_inverter_rating():
    from pvforecast.config import MeasurementBoundary

    snap = hourly_snapshot(TEST_PLACE, date(2026, 6, 21), clearness=1.0, diffuse_fraction=0.12, air_temp_c=10.0)
    inst = make_installation(panel_count=12, boundary=MeasurementBoundary.SOLAR_AC_OUTPUT)
    run = forecast_today(TEST_PLACE, inst, snap, at("04:00", date(2026, 6, 21)))
    assert max(r.forecast_mean_kw for r in run.rows) <= 4.0 + 1e-9
    dc = forecast_today(TEST_PLACE, make_installation(panel_count=12), snap, at("04:00", date(2026, 6, 21)))
    assert max(r.forecast_mean_kw for r in dc.rows) > 4.0
