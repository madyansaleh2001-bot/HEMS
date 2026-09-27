"""SYNTHETIC weather files in the 'Weather House N.csv' layout — for testing the code only.

The values are generated (pvlib clear sky with random cloudiness at an arbitrary test
location); they are not the user's data and not any house's location or weather.

    python tests/synthetic_weather.py OUT_DIR [--days 3] [--houses 13]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate_pv import SOURCE_COLUMNS  # noqa: E402

TEST_LATITUDE, TEST_LONGITUDE = 40.0, -3.0  # arbitrary synthetic test location


def synthetic_frame(start: str = "2021-01-01 00:00:00", periods: int = 3 * 96 + 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=periods, freq="15min")
    loc = pvlib.location.Location(TEST_LATITUDE, TEST_LONGITUDE, tz="UTC", altitude=600)
    utc = times.tz_localize("UTC")
    sp = loc.get_solarposition(utc)
    cs = loc.get_clearsky(utc, solar_position=sp)
    cloud = rng.uniform(0.3, 1.0, periods)
    ghi = np.round(cs["ghi"].to_numpy() * cloud)
    dni = np.round(cs["dni"].to_numpy() * cloud ** 2)
    dhi = np.round(np.maximum(ghi - dni * np.cos(np.radians(sp["zenith"].to_numpy())).clip(0), 0))
    temp = np.round(8 + 10 * np.sin(np.linspace(0, 6 * np.pi, periods)) + rng.normal(0, 1, periods), 1)
    temp[min(5, periods - 1)] = 0.0  # 0 degC is a valid value
    return pd.DataFrame({
        "Timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature": temp,
        "Clearsky DHI": np.round(cs["dhi"].to_numpy()),
        "Clearsky DNI": np.round(cs["dni"].to_numpy()),
        "Clearsky GHI": np.round(cs["ghi"].to_numpy()),
        "Cloud Type": rng.integers(0, 8, periods),
        "Dew Point": np.round(temp - rng.uniform(2, 8, periods), 1),
        "DHI": dhi,
        "DNI": dni,
        "Fill Flag": np.zeros(periods, dtype=int),
        "GHI": ghi,
        "Ozone": np.round(rng.uniform(0.25, 0.35, periods), 3),
        "Relative Humidity": np.round(rng.uniform(20, 95, periods), 2),
        "Surface Albedo": np.round(rng.uniform(0.12, 0.2, periods), 3),
        "Pressure": np.round(rng.uniform(930, 960, periods)),
        "Precipitable Water": np.round(rng.uniform(0.5, 2.5, periods), 1),
        "Wind Direction": np.round(rng.uniform(0, 360, periods)),
        "Wind Speed": np.round(rng.uniform(0, 8, periods), 1),
        "Sun Azimuth (rad)": np.radians(sp["azimuth"].to_numpy()),
        "Sun Zenith (rad)": np.radians(sp["zenith"].to_numpy()),
    })[SOURCE_COLUMNS]


def write_houses(out_dir: Path, houses: int = 13, days: int = 3, start: str = "2021-01-01 00:00:00") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for h in range(1, houses + 1):
        path = out_dir / f"Weather House {h}.csv"
        synthetic_frame(start, days * 96 + 1, seed=h).to_csv(path, index=False)
        paths.append(path)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--houses", type=int, default=13)
    ap.add_argument("--start", default="2021-01-01 00:00:00")
    a = ap.parse_args()
    for p in write_houses(Path(a.out_dir), a.houses, a.days, a.start):
        print(p)
