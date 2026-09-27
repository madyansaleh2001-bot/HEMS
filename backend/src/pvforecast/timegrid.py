"""The sunrise-to-sunset quarter-hour grid.

Convention (proposed implementation of the user's t+15 rule):

    t        = ceil_to_quarter_hour(actual_sunrise)
    last_end = ceil_to_quarter_hour(actual_sunset)
    row i    = [t + (i-1)·15 min, t + i·15 min), labelled by its end time

The grid depends only on the target date and the installation's location, never
on when a refresh runs. The short segment from actual sunrise to the rounded
start is outside the table; ``DaylightGrid.pre_table_segment`` exposes it so its
energy can be reported separately rather than silently lost.

All interval arithmetic uses elapsed time on UTC instants, so days with a
daylight-saving change have neither duplicate nor missing rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum

import pandas as pd

QUARTER_HOUR = pd.Timedelta(minutes=15)
_QUARTER_HOUR_TD = timedelta(minutes=15)


class DayType(str, Enum):
    NORMAL = "normal"
    POLAR_DAY = "polar_day"  # sun never sets: table covers the whole local day
    POLAR_NIGHT = "polar_night"  # sun never rises: empty table, zero energy


class IntervalStatus(str, Enum):
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    FUTURE = "future"


def _require_aware(ts: pd.Timestamp, what: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"{what} must be timezone-aware")
    return ts


def ceil_to_quarter_hour(ts: pd.Timestamp) -> pd.Timestamp:
    """Round up to the next local :00/:15/:30/:45; exact boundaries are unchanged."""
    ts = _require_aware(ts, "timestamp")
    offset = ts.utcoffset()
    if offset % _QUARTER_HOUR_TD:
        raise ValueError(f"UTC offset {offset} is not a whole number of quarter hours")
    # With a quarter-hour-multiple offset, UTC and local quarter-hour boundaries coincide.
    return ts.tz_convert("UTC").ceil("15min").tz_convert(ts.tz)


def local_midnight(local_date: date, timezone: str) -> pd.Timestamp:
    """First instant of a local calendar date (shifted forward if midnight does not exist)."""
    naive = pd.Timestamp(datetime.combine(local_date, time()))
    return naive.tz_localize(timezone, nonexistent="shift_forward", ambiguous=True)


def local_day_bounds(local_date: date, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return local_midnight(local_date, timezone), local_midnight(local_date + timedelta(days=1), timezone)


@dataclass(frozen=True)
class Interval:
    index: int  # 1-based row number
    start: pd.Timestamp  # local, tz-aware; the UTC instant is the row's identity
    end: pd.Timestamp
    local_date: date
    show_offset: bool = False  # set when wall-clock labels repeat (daylight-saving fall-back)

    @property
    def label(self) -> str:
        """Row label = interval end; a midnight end on the next date reads 24:00."""
        if self.end.date() > self.local_date and self.end.time() == time():
            text = "24:00"
        else:
            text = self.end.strftime("%H:%M")
        if self.show_offset:
            text += self.end.strftime(" (UTC%z)")
        return text

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start) / pd.Timedelta(hours=1)

    def status_at(self, now: pd.Timestamp) -> IntervalStatus:
        """An interval is FUTURE until any of it has elapsed (now <= start)."""
        now = _require_aware(now, "now")
        if now >= self.end:
            return IntervalStatus.COMPLETED
        if now > self.start:
            return IntervalStatus.IN_PROGRESS
        return IntervalStatus.FUTURE


@dataclass(frozen=True)
class DaylightGrid:
    local_date: date
    timezone: str
    day_type: DayType
    sunrise: pd.Timestamp | None  # actual (physical) times; None on polar days/nights
    sunset: pd.Timestamp | None
    table_start: pd.Timestamp | None  # rounded display boundaries
    table_end: pd.Timestamp | None
    intervals: tuple[Interval, ...]

    @property
    def pre_table_segment(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Daylight between actual sunrise and the rounded table start, if any."""
        if self.day_type is DayType.NORMAL and self.table_start > self.sunrise:
            return self.sunrise, self.table_start
        return None

    @property
    def daylight_span(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Physical daylight period used for full-day energy and daylight averages."""
        if self.day_type is DayType.NORMAL:
            return self.sunrise, self.sunset
        if self.day_type is DayType.POLAR_DAY:
            return local_day_bounds(self.local_date, self.timezone)
        return None

    @property
    def daylight_hours(self) -> float:
        span = self.daylight_span
        if span is None:
            return 0.0
        return (span[1] - span[0]) / pd.Timedelta(hours=1)


def build_daylight_grid(
    local_date: date,
    timezone: str,
    sunrise: pd.Timestamp | None,
    sunset: pd.Timestamp | None,
    day_type: DayType = DayType.NORMAL,
) -> DaylightGrid:
    if day_type is DayType.POLAR_NIGHT:
        return DaylightGrid(local_date, timezone, day_type, None, None, None, None, ())

    if day_type is DayType.POLAR_DAY:
        start, end = local_day_bounds(local_date, timezone)
        sunrise = sunset = None
    else:
        sunrise = _require_aware(sunrise, "sunrise").tz_convert(timezone)
        sunset = _require_aware(sunset, "sunset").tz_convert(timezone)
        if sunset <= sunrise:
            raise ValueError("sunset must be after sunrise")
        start, end = ceil_to_quarter_hour(sunrise), ceil_to_quarter_hour(sunset)

    start_utc, end_utc = start.tz_convert("UTC"), end.tz_convert("UTC")
    n_rows = int((end_utc - start_utc) / QUARTER_HOUR)
    ends = [(start_utc + (i + 1) * QUARTER_HOUR).tz_convert(timezone) for i in range(n_rows)]
    wall = [e.strftime("%Y-%m-%d %H:%M") for e in ends]
    repeated = {w for w in wall if wall.count(w) > 1}
    intervals = tuple(
        Interval(
            index=i + 1,
            start=(start_utc + i * QUARTER_HOUR).tz_convert(timezone),
            end=ends[i],
            local_date=local_date,
            show_offset=wall[i] in repeated,
        )
        for i in range(n_rows)
    )
    return DaylightGrid(local_date, timezone, day_type, sunrise, sunset, start, end, intervals)
