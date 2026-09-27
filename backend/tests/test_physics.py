from datetime import date

import numpy as np
import pandas as pd
import pytest

from pvforecast.config import Losses
from pvforecast.physics import integrate_energy_kwh, model_power
from pvforecast.sun import daylight_grid
from pvforecast.timegrid import local_day_bounds

from conftest import TEST_PLACE, hourly_snapshot, make_installation

DAY = date(2026, 6, 21)


def _run(installation, snapshot=None):
    grid = daylight_grid(TEST_PLACE, DAY)
    start, end = (t.tz_convert("UTC") for t in local_day_bounds(DAY, TEST_PLACE.timezone))
    snapshot = snapshot or hourly_snapshot(TEST_PLACE, DAY, clearness=1.0, diffuse_fraction=0.12, air_temp_c=10.0)
    return grid, model_power(TEST_PLACE, installation, snapshot, start, end, [grid.daylight_span]).fine


def test_module_efficiency_does_not_change_power():
    _, a = _run(make_installation(efficiency=None))
    _, b = _run(make_installation(efficiency=0.22))
    np.testing.assert_allclose(a["pv_dc_w"], b["pv_dc_w"])


def test_ac_estimate_is_clipped_but_dc_harvest_is_not():
    _, fine = _run(make_installation(panel_count=12, rated_w=500))  # 6 kWp against a 4 kW AC rating
    assert fine["pv_dc_w"].max() > 4000.0
    assert fine["solar_ac_w_est"].max() <= 4000.0 + 1e-9
    assert np.isclose(fine["solar_ac_w_est"].max(), 4000.0)


def test_power_scales_with_panel_count_and_losses():
    _, one = _run(make_installation(panel_count=1, losses=Losses(**{n: 0.0 for n in Losses.NAMES})))
    _, two = _run(make_installation(panel_count=2, losses=Losses(**{n: 0.0 for n in Losses.NAMES})))
    np.testing.assert_allclose(two["pv_dc_w"], 2 * one["pv_dc_w"])
    _, lossy = _run(make_installation(panel_count=1))
    ratio = lossy["pv_dc_w"].sum() / one["pv_dc_w"].sum()
    assert 0.85 < ratio < 0.9  # PVWatts defaults (availability 0 %) ≈ 11.4 % loss


def test_night_is_zero_even_when_weather_is_missing():
    snap = hourly_snapshot(TEST_PLACE, DAY)
    snap.data.loc[:, "ghi_wm2"] = np.where(snap.data["ghi_wm2"] < 1.0, np.nan, snap.data["ghi_wm2"])
    grid, fine = _run(make_installation(), snap)
    night = ~fine["daylight"]
    assert (fine.loc[night, "pv_dc_w"] == 0.0).all()


def test_missing_daytime_weather_is_nan_not_zero():
    snap = hourly_snapshot(TEST_PLACE, DAY)
    snap.data.loc[pd.Timestamp("2026-06-21 19:00", tz="UTC"), "ghi_wm2"] = np.nan  # 12:00-13:00 MDT
    _, fine = _run(make_installation(), snap)
    gap = fine.loc["2026-06-21 18:00":"2026-06-21 18:59", "pv_dc_w"]
    assert gap.isna().all()
    assert integrate_energy_kwh(fine, "pv_dc_w", pd.Timestamp("2026-06-21 18:00", tz="UTC"),
                                pd.Timestamp("2026-06-21 18:15", tz="UTC")) is None


def test_integration_weights_partial_minutes():
    _, fine = _run(make_installation())
    t0 = pd.Timestamp("2026-06-21 18:00", tz="UTC")
    whole = integrate_energy_kwh(fine, "pv_dc_w", t0, t0 + pd.Timedelta(minutes=1))
    half = integrate_energy_kwh(fine, "pv_dc_w", t0, t0 + pd.Timedelta(seconds=30))
    assert half == pytest.approx(whole / 2, rel=1e-12)


def test_zero_daytime_irradiance_gives_zero_power_not_missing():
    snap = hourly_snapshot(TEST_PLACE, DAY)
    snap.data.loc[:, ["ghi_wm2", "dhi_wm2"]] = 0.0
    _, fine = _run(make_installation(), snap)
    assert fine["pv_dc_w"].notna().all()
    assert (fine["pv_dc_w"] == 0.0).all()
