from datetime import date

import pandas as pd
import pytest

from pvforecast.config import Place
from pvforecast.sun import daylight_grid
from pvforecast.timegrid import DayType, IntervalStatus, build_daylight_grid, ceil_to_quarter_hour

TZ = "Asia/Amman"
DAY = date(2026, 9, 27)


def local(hhmmss: str, day: date = DAY, tz: str = TZ) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmmss}").tz_localize(tz)


def test_ceil_rounds_up_to_next_quarter_hour():
    assert ceil_to_quarter_hour(local("06:07")) == local("06:15")
    assert ceil_to_quarter_hour(local("18:02")) == local("18:15")
    assert ceil_to_quarter_hour(local("06:15:00.001")) == local("06:30")


def test_exact_quarter_hour_boundaries_are_unchanged():
    for t in ("06:00", "06:15", "18:30", "18:45"):
        assert ceil_to_quarter_hour(local(t)) == local(t)


def test_ceil_uses_local_quarter_hours_in_half_and_quarter_hour_offset_zones():
    assert ceil_to_quarter_hour(local("06:07", tz="Asia/Kolkata")) == local("06:15", tz="Asia/Kolkata")
    assert ceil_to_quarter_hour(local("06:07", tz="Asia/Kathmandu")) == local("06:15", tz="Asia/Kathmandu")


def test_spec_example_06_07_to_18_02_gives_48_rows():
    grid = build_daylight_grid(DAY, TZ, local("06:07"), local("18:02"))
    assert grid.table_start == local("06:15") and grid.table_end == local("18:15")
    assert len(grid.intervals) == 48
    first, last = grid.intervals[0], grid.intervals[-1]
    assert (first.start, first.end, first.label) == (local("06:15"), local("06:30"), "06:30")
    assert (last.start, last.end, last.label) == (local("18:00"), local("18:15"), "18:15")
    assert all(iv.duration_hours == 0.25 for iv in grid.intervals)
    # the physical sunrise/sunset are preserved alongside the rounded boundaries
    assert grid.sunrise == local("06:07") and grid.sunset == local("18:02")
    assert grid.pre_table_segment == (local("06:07"), local("06:15"))


def test_exactly_aligned_sunrise_has_no_pre_table_segment():
    grid = build_daylight_grid(DAY, TZ, local("06:15"), local("18:00"))
    assert grid.pre_table_segment is None
    assert grid.intervals[0].start == local("06:15")
    assert grid.table_end == local("18:00")


def test_row_count_follows_daylight_duration():
    short = build_daylight_grid(DAY, TZ, local("07:40"), local("16:20"))
    long = build_daylight_grid(DAY, TZ, local("05:01"), local("19:59"))
    assert len(short.intervals) == 35  # 07:45 -> 16:30
    assert len(long.intervals) == 59  # 05:15 -> 20:00 is 14 h 45 min
    assert len(short.intervals) != 24


def test_interval_status_boundaries():
    grid = build_daylight_grid(DAY, TZ, local("06:07"), local("18:02"))
    iv = grid.intervals[0]  # 06:15-06:30
    assert iv.status_at(local("06:15")) is IntervalStatus.FUTURE
    assert iv.status_at(local("06:20")) is IntervalStatus.IN_PROGRESS
    assert iv.status_at(local("06:30")) is IntervalStatus.COMPLETED


def test_spring_forward_day_uses_elapsed_time():
    day, tz = date(2026, 3, 29), "Europe/Berlin"  # 02:00 CET -> 03:00 CEST
    grid = build_daylight_grid(day, tz, local("01:10", day, tz), local("04:05", day, tz))
    assert len(grid.intervals) == 8  # 01:15 CET .. 04:15 CEST is two elapsed hours
    labels = [iv.label for iv in grid.intervals]
    assert labels == ["01:30", "01:45", "03:00", "03:15", "03:30", "03:45", "04:00", "04:15"]
    assert all(iv.duration_hours == 0.25 for iv in grid.intervals)


def test_fall_back_day_has_no_duplicate_identities():
    day, tz = date(2026, 10, 25), "Europe/Berlin"  # 03:00 CEST -> 02:00 CET
    sunrise = pd.Timestamp("2026-10-24 23:50", tz="UTC").tz_convert(tz)  # 01:50 CEST
    sunset = pd.Timestamp("2026-10-25 02:20", tz="UTC").tz_convert(tz)  # 03:20 CET
    grid = build_daylight_grid(day, tz, sunrise, sunset)
    assert len(grid.intervals) == 10
    starts = [iv.start for iv in grid.intervals]
    assert len(set(starts)) == len(starts)
    labels = [iv.label for iv in grid.intervals]
    assert len(set(labels)) == len(labels)
    assert "02:15 (UTC+0200)" in labels and "02:15 (UTC+0100)" in labels


def test_polar_day_and_night_policy():
    svalbard = Place("svalbard", "Longyearbyen", 78.22, 15.65, "Arctic/Longyearbyen")
    summer = daylight_grid(svalbard, date(2026, 6, 21))
    assert summer.day_type is DayType.POLAR_DAY
    assert summer.sunrise is None and summer.sunset is None
    assert len(summer.intervals) == 96 and summer.intervals[-1].label == "24:00"
    winter = daylight_grid(svalbard, date(2026, 12, 21))
    assert winter.day_type is DayType.POLAR_NIGHT
    assert winter.intervals == () and winter.daylight_hours == 0.0


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        ceil_to_quarter_hour(pd.Timestamp("2026-09-27 06:07"))
