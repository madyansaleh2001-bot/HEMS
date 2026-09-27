"""Shared test fixtures. All sites, panels and weather here are synthetic test inputs."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pvlib
import pytest

from pvforecast.config import (
    Installation,
    Losses,
    MeasurementBoundary,
    Mounting,
    PanelSpec,
    Place,
    SubArray,
    felicity_ivem4024_ii,
)
from pvforecast.timegrid import local_day_bounds
from pvforecast.weather import Aggregation, WeatherSnapshot

# A generic mid-latitude test location with an explicit time zone and elevation.
TEST_PLACE = Place("test-place", "Test place", 39.7407, -105.1686, "America/Denver", elevation_m=1800.0)


def make_installation(
    panel_count: int = 6,
    rated_w: float = 500.0,
    efficiency: float | None = None,
    boundary: MeasurementBoundary = MeasurementBoundary.PV_DC_INPUT,
    losses: Losses | None = None,
    place: Place = TEST_PLACE,
) -> Installation:
    panel = PanelSpec(source="manual", rated_power_w=rated_w, efficiency=efficiency, gamma_pmp_per_c=-0.0030,
                      length_m=2.0 if efficiency else None, width_m=1.134 if efficiency else None)
    return Installation(
        installation_id="test-installation",
        place_id=place.place_id,
        name="Test installation",
        config_version=1,
        sub_arrays=(SubArray("south", panel, panel_count, 25.0, 180.0, Mounting.CLOSE_ROOF_MOUNT),),
        inverter=felicity_ivem4024_ii(),
        losses=losses or Losses(),
        measurement_boundary=boundary,
    )


def hourly_snapshot(
    place: Place,
    first_day: date,
    days: int = 3,
    clearness: float = 0.7,
    diffuse_fraction: float = 0.35,
    air_temp_c: float = 20.0,
    wind_ms: float = 2.0,
    retrieved_at: pd.Timestamp | None = None,
) -> WeatherSnapshot:
    """Synthetic hourly weather in provider layout: irradiance = mean over the preceding hour."""
    start, _ = local_day_bounds(first_day - timedelta(days=1), place.timezone)
    _, end = local_day_bounds(first_day + timedelta(days=days), place.timezone)
    start, end = start.tz_convert("UTC").floor("1h"), end.tz_convert("UTC").ceil("1h")
    minutes = pd.date_range(start, end, freq="1min", inclusive="left") + pd.Timedelta(seconds=30)
    loc = pvlib.location.Location(place.latitude, place.longitude, altitude=place.elevation_m or 0.0)
    ghi_cs = loc.get_clearsky(minutes)["ghi"]
    hourly_ghi = (ghi_cs * clearness).groupby(minutes.ceil("1h")).mean()
    index = pd.DatetimeIndex(hourly_ghi.index)
    data = pd.DataFrame(
        {
            "ghi_wm2": hourly_ghi.to_numpy(),
            "dhi_wm2": hourly_ghi.to_numpy() * diffuse_fraction,
            "air_temperature_c": air_temp_c,
            "wind_speed_ms": wind_ms,
        },
        index=index,
    )
    return WeatherSnapshot(
        snapshot_id="synthetic-hourly",
        provider="synthetic test data",
        model=None,
        latitude=place.latitude,
        longitude=place.longitude,
        retrieved_at_utc=retrieved_at or start,
        issued_at_utc=None,
        native_resolution_minutes=60,
        data=data,
        aggregation={
            "ghi_wm2": Aggregation.PRECEDING_MEAN,
            "dhi_wm2": Aggregation.PRECEDING_MEAN,
            "air_temperature_c": Aggregation.INSTANTANEOUS,
            "wind_speed_ms": Aggregation.INSTANTANEOUS,
        },
        is_simulated=True,
        description="synthetic test weather",
    )


@pytest.fixture
def place() -> Place:
    return TEST_PLACE


@pytest.fixture
def installation() -> Installation:
    return make_installation()
