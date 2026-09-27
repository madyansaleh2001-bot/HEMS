"""Weather snapshots and their conversion to the model's fine time steps.

A ``WeatherSnapshot`` keeps provider data as delivered (UTC timestamps as the
provider labels them), together with provenance: provider, model, retrieval
time, upstream issue time when supplied, native resolution and how each
variable is aggregated. A retrieval time is not claimed as an issue time.

Irradiance conversion (interval means -> 1-minute steps) preserves energy:
within each native interval the provider's mean GHI is distributed in
proportion to clear-sky GHI, so the mean of the fine values equals the provider
mean exactly. The provider's diffuse fraction is held constant within the
interval and beam irradiance follows from closure (GHI = DNI·cosZ + DHI). Only
one irradiance workflow is used: horizontal components are later transposed
once to each panel plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import ModelNote, Place
from .sun import solar_geometry

FINE_STEP = pd.Timedelta(minutes=1)

# Beam irradiance is not resolved closer to the horizon than this; any remainder is treated as diffuse.
MAX_BEAM_ZENITH_DEG = 87.0

IRRADIANCE_COLUMNS = ("ghi_wm2", "dhi_wm2", "dni_wm2")


class Aggregation(str, Enum):
    INSTANTANEOUS = "instantaneous"
    PRECEDING_MEAN = "mean_over_preceding_interval"
    PRECEDING_SUM = "sum_over_preceding_interval"


@dataclass(frozen=True)
class WeatherSnapshot:
    snapshot_id: str
    provider: str
    model: str | None
    latitude: float
    longitude: float
    retrieved_at_utc: pd.Timestamp
    issued_at_utc: pd.Timestamp | None
    native_resolution_minutes: int
    data: pd.DataFrame  # UTC index; canonical column names (see openmeteo.VARIABLES)
    aggregation: Mapping[str, Aggregation]
    is_simulated: bool = False
    description: str = ""
    raw: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.data.index.tz is None or str(self.data.index.tz) != "UTC":
            raise ValueError("weather data must be indexed by UTC timestamps")
        missing = set(self.data.columns) - set(self.aggregation)
        if missing:
            raise ValueError(f"aggregation convention missing for columns {sorted(missing)}")
        if "ghi_wm2" not in self.data.columns:
            raise ValueError("weather snapshot needs global horizontal irradiance (ghi_wm2)")
        if self.aggregation["ghi_wm2"] is not Aggregation.PRECEDING_MEAN:
            raise NotImplementedError("only interval-mean irradiance (mean over preceding interval) is supported")

    @property
    def native_step(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.native_resolution_minutes)

    def provenance(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "provider": self.provider,
            "model": self.model,
            "retrieved_at_utc": self.retrieved_at_utc.isoformat(),
            "issued_at_utc": self.issued_at_utc.isoformat() if self.issued_at_utc is not None else None,
            "native_resolution_minutes": self.native_resolution_minutes,
            "is_simulated": self.is_simulated,
            "description": self.description,
        }


def fine_bins(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DatetimeIndex:
    """Start instants of the 1-minute bins covering [start, end)."""
    return pd.date_range(start_utc, end_utc, freq=FINE_STEP, inclusive="left", tz="UTC")


def _native_labels(midpoints: pd.DatetimeIndex, step: pd.Timedelta) -> pd.DatetimeIndex:
    # A provider row labelled L covers (L - step, L]; midpoints never fall on a boundary.
    return midpoints.ceil(step)


def _interp_instantaneous(series: pd.Series, targets: pd.DatetimeIndex) -> np.ndarray:
    """Linear interpolation in time; NaN outside the data range or next to a missing value."""
    src_t = series.index.as_unit("ns").asi8.astype("float64")
    src_v = series.to_numpy(dtype="float64")
    t = targets.as_unit("ns").asi8.astype("float64")
    j = np.searchsorted(src_t, t, side="right")
    ok = (j > 0) & (j < len(src_t))
    out = np.full(len(t), np.nan)
    i0, i1 = j[ok] - 1, j[ok]
    w = (t[ok] - src_t[i0]) / (src_t[i1] - src_t[i0])
    out[ok] = src_v[i0] * (1.0 - w) + src_v[i1] * w
    return out


def to_fine_resolution(
    place: Place, snapshot: WeatherSnapshot, start_utc: pd.Timestamp, end_utc: pd.Timestamp
) -> tuple[pd.DataFrame, tuple[ModelNote, ...]]:
    """Weather and solar geometry on 1-minute bins over [start, end).

    The returned frame is indexed by bin start; values are evaluated at bin
    midpoints. Missing provider data remain NaN — they are never replaced by zero.
    """
    step = snapshot.native_step
    notes: list[ModelNote] = []

    # Work on whole native intervals so each interval's clear-sky mean is complete.
    ext_start, ext_end = start_utc.floor(step), end_utc.ceil(step)
    bins = fine_bins(ext_start, ext_end)
    mid = bins + FINE_STEP / 2
    geo = solar_geometry(place, mid)
    labels = _native_labels(mid, step)
    data = snapshot.data

    def native(column: str) -> np.ndarray:
        return data[column].reindex(labels).to_numpy(dtype="float64")

    # --- global horizontal: clear-sky-shaped, energy-preserving
    cs = geo["clearsky_ghi"].to_numpy()
    cs_mean = pd.Series(cs).groupby(labels.asi8).transform("mean").to_numpy()
    ghi_native = native("ghi_wm2")
    negative = np.nansum(ghi_native < 0)
    ghi_native = np.where(ghi_native < 0, 0.0, ghi_native)
    with np.errstate(invalid="ignore", divide="ignore"):
        ghi = np.where(cs_mean > 0, ghi_native / cs_mean * cs, 0.0)
    ghi = np.where(np.isnan(ghi_native), np.nan, ghi)
    unplaced = (cs_mean <= 0) & (ghi_native > 1.0)
    if unplaced.any():
        notes.append(ModelNote(
            "warning", "weather",
            f"{int(unplaced.sum())} minute(s) where the provider reports irradiance but the modelled sun is "
            "below the horizon for the whole native interval; that irradiance was not used",
        ))
    if negative:
        notes.append(ModelNote("warning", "weather", "negative provider irradiance values were set to zero"))

    # --- diffuse / beam split
    zen = geo["apparent_zenith"].to_numpy()
    cosz = np.cos(np.radians(zen))
    if "dhi_wm2" in data.columns:
        dhi_native = native("dhi_wm2")
        with np.errstate(invalid="ignore", divide="ignore"):
            kd = np.where(ghi_native > 0, np.clip(dhi_native / ghi_native, 0.0, 1.0), 1.0)
        kd = np.where(np.isnan(dhi_native), np.nan, kd)
        dhi = kd * ghi
    else:
        import pvlib

        erbs = pvlib.irradiance.erbs(pd.Series(ghi, index=mid), pd.Series(zen, index=mid), mid)
        dhi = erbs["dhi"].to_numpy()
        notes.append(ModelNote("assumption", "weather", "provider gave no diffuse irradiance; Erbs decomposition used"))
    beam_ok = zen < MAX_BEAM_ZENITH_DEG
    with np.errstate(invalid="ignore", divide="ignore"):
        dni = np.where(beam_ok, (ghi - dhi) / cosz, 0.0)
    dni = np.clip(dni, 0.0, geo["dni_extra"].to_numpy())
    dni = np.where(np.isnan(ghi) | np.isnan(dhi), np.nan, dni)
    dhi = ghi - dni * np.where(beam_ok, cosz, 0.0)  # closure keeps GHI (and its energy) unchanged

    out = pd.DataFrame(index=bins)
    out["ghi_wm2"], out["dhi_wm2"], out["dni_wm2"] = ghi, dhi, dni
    for col in ("apparent_zenith", "azimuth", "apparent_elevation", "dni_extra", "airmass_relative"):
        out[col] = geo[col].to_numpy()

    # --- other variables
    for column in data.columns:
        if column in IRRADIANCE_COLUMNS:
            continue
        agg = snapshot.aggregation[column]
        if agg is Aggregation.INSTANTANEOUS:
            out[column] = _interp_instantaneous(data[column], mid)
        elif agg is Aggregation.PRECEDING_MEAN:
            out[column] = native(column)
        else:  # accumulated totals are spread evenly over their interval
            out[column] = native(column) * (FINE_STEP / step)

    return out.loc[(out.index >= start_utc) & (out.index < end_utc)], tuple(notes)


def clear_sky_snapshot(
    place: Place,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    air_temperature_c: float,
    wind_speed_ms: float,
    retrieved_at_utc: pd.Timestamp,
) -> WeatherSnapshot:
    """A *simulated* cloud-free scenario (Ineichen clear sky) at 15-minute resolution.

    Air temperature and wind are constant scenario inputs. This is a simulation
    for testing and what-if comparison — it is not a weather forecast.
    """
    step = pd.Timedelta(minutes=15)
    ext_start, ext_end = start_utc.floor(step), end_utc.ceil(step)
    bins = fine_bins(ext_start, ext_end)
    mid = bins + FINE_STEP / 2
    import pvlib

    alt = place.elevation_m if place.elevation_m is not None else pvlib.location.lookup_altitude(place.latitude, place.longitude)
    loc = pvlib.location.Location(place.latitude, place.longitude, tz="UTC", altitude=alt)
    cs = loc.get_clearsky(mid, model="ineichen")
    labels = _native_labels(mid, step)
    means = cs[["ghi", "dhi"]].groupby(labels).mean()
    data = pd.DataFrame(
        {
            "ghi_wm2": means["ghi"].to_numpy(),
            "dhi_wm2": means["dhi"].to_numpy(),
            "air_temperature_c": float(air_temperature_c),
            "wind_speed_ms": float(wind_speed_ms),
        },
        index=pd.DatetimeIndex(means.index, tz="UTC"),
    )
    return WeatherSnapshot(
        snapshot_id=f"clearsky-{place.place_id}-{ext_start:%Y%m%dT%H%MZ}",
        provider="simulation: pvlib Ineichen clear sky",
        model="ineichen",
        latitude=place.latitude,
        longitude=place.longitude,
        retrieved_at_utc=retrieved_at_utc,
        issued_at_utc=None,
        native_resolution_minutes=15,
        data=data,
        aggregation={
            "ghi_wm2": Aggregation.PRECEDING_MEAN,
            "dhi_wm2": Aggregation.PRECEDING_MEAN,
            "air_temperature_c": Aggregation.INSTANTANEOUS,
            "wind_speed_ms": Aggregation.INSTANTANEOUS,
        },
        is_simulated=True,
        description=(
            f"SIMULATION — cloud-free sky, constant {air_temperature_c:g} °C air and {wind_speed_ms:g} m/s wind; "
            "not a weather forecast"
        ),
    )
