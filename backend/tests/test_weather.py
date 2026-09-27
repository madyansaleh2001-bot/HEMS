from datetime import date

import numpy as np
import pandas as pd
import pytest

from pvforecast import openmeteo
from pvforecast.weather import Aggregation, to_fine_resolution

from conftest import TEST_PLACE, hourly_snapshot

DAY = date(2026, 9, 27)


def test_disaggregation_preserves_interval_mean_irradiance():
    snap = hourly_snapshot(TEST_PLACE, DAY)
    start, end = pd.Timestamp("2026-09-27 00:00", tz="UTC"), pd.Timestamp("2026-09-28 00:00", tz="UTC")
    fine, _ = to_fine_resolution(TEST_PLACE, snap, start, end)
    hourly_from_fine = fine["ghi_wm2"].groupby((fine.index + pd.Timedelta(seconds=30)).ceil("1h")).mean()
    provider = snap.data["ghi_wm2"].reindex(hourly_from_fine.index)
    daytime = provider > 1.0
    assert daytime.sum() >= 10
    np.testing.assert_allclose(hourly_from_fine[daytime], provider[daytime], rtol=1e-9)
    # closure: GHI = DNI·cos(Z) + DHI at every step
    cosz = np.cos(np.radians(fine["apparent_zenith"]))
    beam = np.where(fine["apparent_zenith"] < 87, fine["dni_wm2"] * cosz, 0.0)
    np.testing.assert_allclose(beam + fine["dhi_wm2"], fine["ghi_wm2"], atol=1e-6)


def test_missing_provider_values_stay_missing():
    snap = hourly_snapshot(TEST_PLACE, DAY)
    snap.data.loc[pd.Timestamp("2026-09-27 19:00", tz="UTC"), "ghi_wm2"] = np.nan
    start, end = pd.Timestamp("2026-09-27 17:00", tz="UTC"), pd.Timestamp("2026-09-27 21:00", tz="UTC")
    fine, _ = to_fine_resolution(TEST_PLACE, snap, start, end)
    gap = fine.loc["2026-09-27 18:00":"2026-09-27 18:59", "ghi_wm2"]
    assert gap.isna().all()
    assert fine.loc["2026-09-27 19:00":, "ghi_wm2"].notna().all()


def _payload():
    """Synthetic response in the documented Open-Meteo layout (not real forecast data)."""
    t0 = int(pd.Timestamp("2026-09-27 00:00", tz="UTC").timestamp())
    times = [t0 + 3600 * i for i in range(48)]
    # daylight at the test place (UTC-6) falls roughly between 13:00 and 01:00 UTC
    ghi = [700 * np.sin(np.pi * ((i % 24 - 13) % 24) / 12) if 0 < (i % 24 - 13) % 24 < 12 else 0.0 for i in range(48)]
    return {
        "latitude": 39.74, "longitude": -105.17, "utc_offset_seconds": 0, "timezone": "GMT",
        "hourly_units": {
            "time": "unixtime", "shortwave_radiation": "W/m²", "diffuse_radiation": "W/m²",
            "temperature_2m": "°C", "wind_speed_10m": "m/s", "cloud_cover": "%", "precipitation": "mm",
        },
        "hourly": {
            "time": times, "shortwave_radiation": ghi, "diffuse_radiation": [g * 0.3 for g in ghi],
            "temperature_2m": [18.0] * 48, "wind_speed_10m": [3.0] * 48, "cloud_cover": [20] * 48,
            "precipitation": [0.0] * 47 + [None],
        },
    }


def test_open_meteo_parsing_and_provenance():
    retrieved = pd.Timestamp("2026-09-27 12:00", tz="UTC")
    snap = openmeteo.parse_forecast(_payload(), retrieved, TEST_PLACE)
    assert snap.native_resolution_minutes == 60
    assert snap.issued_at_utc is None  # the endpoint does not report a model run time
    assert snap.retrieved_at_utc == retrieved
    assert str(snap.data.index.tz) == "UTC"
    assert snap.aggregation["ghi_wm2"] is Aggregation.PRECEDING_MEAN
    assert snap.aggregation["precipitation_mm"] is Aggregation.PRECEDING_SUM
    assert snap.aggregation["air_temperature_c"] is Aggregation.INSTANTANEOUS
    assert np.isnan(snap.data["precipitation_mm"].iloc[-1])  # missing stays missing
    assert not snap.is_simulated


def test_open_meteo_unit_mismatch_is_rejected():
    payload = _payload()
    payload["hourly_units"]["wind_speed_10m"] = "km/h"
    with pytest.raises(openmeteo.OpenMeteoError):
        openmeteo.parse_forecast(payload, pd.Timestamp.now(tz="UTC"), TEST_PLACE)


def test_open_meteo_request_uses_utc_and_si_units():
    params = openmeteo.request_params(TEST_PLACE)
    assert params["timezone"] == "GMT" and params["timeformat"] == "unixtime"
    assert params["wind_speed_unit"] == "ms"
    assert "shortwave_radiation" in params["hourly"]


def test_parsed_open_meteo_snapshot_drives_a_forecast():
    from pvforecast.products import forecast_today

    from conftest import make_installation

    snap = openmeteo.parse_forecast(_payload(), pd.Timestamp("2026-09-27 09:00", tz="UTC"), TEST_PLACE)
    run = forecast_today(TEST_PLACE, make_installation(), snap, pd.Timestamp("2026-09-27 10:00", tz="UTC"))
    assert run.rows and all(r.forecast_mean_kw is not None for r in run.rows)
    assert max(r.forecast_mean_kw for r in run.rows) > 0.5
    assert not run.is_simulation and run.weather["issued_at_utc"] is None
