"""Sunrise/sunset and solar geometry for a place (pvlib SPA)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pvlib

from .config import Place
from .timegrid import DayType, DaylightGrid, build_daylight_grid, local_midnight


class UnsupportedDayError(ValueError):
    """The sun rises without setting (or vice versa) within the local date."""


def sun_times(place: Place, local_date: date) -> tuple[DayType, pd.Timestamp | None, pd.Timestamp | None]:
    """Actual sunrise and sunset for the local date at the place (SPA, apparent horizon)."""
    midnight = local_midnight(local_date, place.timezone)
    spa = pvlib.solarposition.sun_rise_set_transit_spa(
        pd.DatetimeIndex([midnight]), place.latitude, place.longitude
    )
    sunrise, sunset, transit = (spa[c].iloc[0] for c in ("sunrise", "sunset", "transit"))
    if pd.notna(sunrise) and pd.notna(sunset):
        return DayType.NORMAL, pd.Timestamp(sunrise), pd.Timestamp(sunset)
    if pd.isna(sunrise) and pd.isna(sunset):
        elevation = pvlib.solarposition.get_solarposition(
            pd.DatetimeIndex([transit]), place.latitude, place.longitude
        )["apparent_elevation"].iloc[0]
        return (DayType.POLAR_DAY if elevation > 0 else DayType.POLAR_NIGHT), None, None
    raise UnsupportedDayError(
        f"{local_date} at ({place.latitude}, {place.longitude}) has a sunrise or a sunset but not both; "
        "polar transition days are not supported"
    )


def daylight_grid(place: Place, local_date: date) -> DaylightGrid:
    day_type, sunrise, sunset = sun_times(place, local_date)
    return build_daylight_grid(local_date, place.timezone, sunrise, sunset, day_type)


def altitude_m(place: Place) -> tuple[float, bool]:
    """Site altitude; returns (value, looked_up) where looked_up means it was not entered."""
    if place.elevation_m is not None:
        return float(place.elevation_m), False
    return float(pvlib.location.lookup_altitude(place.latitude, place.longitude)), True


def solar_geometry(place: Place, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Solar position, extraterrestrial DNI, air mass and clear-sky irradiance at the given instants.

    Clear-sky irradiance is used only as an intra-interval *shape* when spreading
    weather-provider interval means over finer time steps; it never replaces the
    provider's irradiance totals.
    """
    alt, _ = altitude_m(place)
    location = pvlib.location.Location(place.latitude, place.longitude, tz="UTC", altitude=alt)
    solpos = location.get_solarposition(times_utc)
    clearsky = location.get_clearsky(times_utc, model="ineichen", solar_position=solpos)
    out = pd.DataFrame(index=times_utc)
    out["apparent_zenith"] = solpos["apparent_zenith"]
    out["azimuth"] = solpos["azimuth"]
    out["apparent_elevation"] = solpos["apparent_elevation"]
    out["dni_extra"] = pvlib.irradiance.get_extra_radiation(times_utc)
    out["airmass_relative"] = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    out["clearsky_ghi"] = clearsky["ghi"]
    return out
