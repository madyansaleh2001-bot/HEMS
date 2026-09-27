"""Open-Meteo forecast adapter.

Requests UTC unix timestamps and SI units, then converts the response to
canonical column names. Per Open-Meteo's documentation, radiation values are
means over the preceding interval, precipitation is a sum over the preceding
interval, and the other variables are instantaneous.

The forecast endpoint does not report the upstream model run time, so
``issued_at_utc`` stays empty; only the retrieval time is recorded.

Status: the parser is tested against a fixture that follows the documented
response layout. A live request has not been tested from the development
environment (its network policy blocks api.open-meteo.com).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import pandas as pd

from .config import Place
from .weather import Aggregation, WeatherSnapshot

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# provider variable -> (canonical column, expected unit, aggregation)
VARIABLES: dict[str, tuple[str, str, Aggregation]] = {
    "shortwave_radiation": ("ghi_wm2", "W/m²", Aggregation.PRECEDING_MEAN),
    "diffuse_radiation": ("dhi_wm2", "W/m²", Aggregation.PRECEDING_MEAN),
    "temperature_2m": ("air_temperature_c", "°C", Aggregation.INSTANTANEOUS),
    "wind_speed_10m": ("wind_speed_ms", "m/s", Aggregation.INSTANTANEOUS),
    "cloud_cover": ("cloud_cover_pct", "%", Aggregation.INSTANTANEOUS),
    "cloud_cover_low": ("cloud_cover_low_pct", "%", Aggregation.INSTANTANEOUS),
    "cloud_cover_mid": ("cloud_cover_mid_pct", "%", Aggregation.INSTANTANEOUS),
    "cloud_cover_high": ("cloud_cover_high_pct", "%", Aggregation.INSTANTANEOUS),
    "relative_humidity_2m": ("relative_humidity_pct", "%", Aggregation.INSTANTANEOUS),
    "precipitation": ("precipitation_mm", "mm", Aggregation.PRECEDING_SUM),
    "surface_pressure": ("surface_pressure_hpa", "hPa", Aggregation.INSTANTANEOUS),
}

REQUIRED = ("shortwave_radiation", "temperature_2m", "wind_speed_10m")

UNIT_ALIASES = {"W/m²": ("W/m²", "W/m^2", "W m-2")}


class OpenMeteoError(RuntimeError):
    pass


def request_params(place: Place, forecast_days: int = 3, model: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "latitude": place.latitude,
        "longitude": place.longitude,
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "forecast_days": forecast_days,
        "past_days": 1,  # keeps today's early intervals covered for zones ahead of UTC
    }
    if model:
        params["models"] = model
    return params


def parse_forecast(
    payload: Mapping[str, Any], retrieved_at_utc: pd.Timestamp, place: Place, model: str | None = None
) -> WeatherSnapshot:
    if payload.get("error"):
        raise OpenMeteoError(str(payload.get("reason", "Open-Meteo returned an error")))
    hourly, units = payload.get("hourly"), payload.get("hourly_units", {})
    if not hourly or "time" not in hourly:
        raise OpenMeteoError("response has no hourly block")
    if units.get("time") != "unixtime":
        raise OpenMeteoError(f"expected unixtime timestamps, got {units.get('time')!r}")

    index = pd.to_datetime(pd.Series(hourly["time"], dtype="int64"), unit="s", utc=True)
    frame, aggregation = {}, {}
    for variable, (column, unit, agg) in VARIABLES.items():
        if variable not in hourly:
            if variable in REQUIRED:
                raise OpenMeteoError(f"required variable {variable} missing from response")
            continue
        if units.get(variable) not in UNIT_ALIASES.get(unit, (unit,)):
            raise OpenMeteoError(f"{variable}: expected unit {unit!r}, got {units.get(variable)!r}")
        frame[column] = pd.to_numeric(pd.Series(hourly[variable]), errors="coerce").to_numpy(dtype="float64")
        aggregation[column] = agg
    data = pd.DataFrame(frame, index=pd.DatetimeIndex(index))

    steps = data.index.to_series().diff().dropna().unique()
    if len(steps) != 1:
        raise OpenMeteoError("hourly timestamps are not evenly spaced")
    resolution = int(pd.Timedelta(steps[0]) / pd.Timedelta(minutes=1))

    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return WeatherSnapshot(
        snapshot_id=f"open-meteo-{retrieved_at_utc:%Y%m%dT%H%M%SZ}-{digest}",
        provider="Open-Meteo forecast API",
        model=model or "best_match",
        latitude=float(payload.get("latitude", place.latitude)),
        longitude=float(payload.get("longitude", place.longitude)),
        retrieved_at_utc=retrieved_at_utc,
        issued_at_utc=None,
        native_resolution_minutes=resolution,
        data=data,
        aggregation=aggregation,
        description=f"Open-Meteo hourly forecast ({model or 'best_match'}), retrieved {retrieved_at_utc:%Y-%m-%d %H:%M} UTC",
        raw=dict(payload),
    )


def fetch_forecast(place: Place, forecast_days: int = 3, model: str | None = None, timeout_s: float = 20.0) -> WeatherSnapshot:
    import requests

    response = requests.get(FORECAST_URL, params=request_params(place, forecast_days, model), timeout=timeout_s)
    retrieved = pd.Timestamp.now(tz="UTC")
    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenMeteoError(f"HTTP {response.status_code}: response is not JSON") from exc
    if response.status_code != 200:
        raise OpenMeteoError(f"HTTP {response.status_code}: {payload.get('reason', payload)}")
    return parse_forecast(payload, retrieved, place, model)
