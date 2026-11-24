"""
hems_forecast_feed.py - loads the PV forecasts, builds the 3-model ensemble and the forecast features.

What it does
  * Finds the forecast files below RESULTS_DIR:
        train  2024-03-15..2025-02-28 : cv_<model>_*/run_*_cv1..cv4_*/forecasts/validation_{same_day,next_day}_forecasts.csv.gz
        select 2025-03-01..2025-05-26 : cv_<model>_*/run_*_cv5_*/forecasts/validation_*_forecasts.csv.gz
        test   2025-05-27..2025-12-31 : final_cv_<model>/run_<model>_*/forecasts/test_evaluation_*_forecasts.csv.gz
    (honest out-of-sample forecasts only; the fold models are never used for the test or on the Pi).
  * Keeps Valid_Target rows, inner-joins LSTM, GRU and CNN-LSTM on Issue_Timestamp + Target_Timestamp,
    Ensemble = mean of Predicted_W, Spread = std (ddof = 0). Night (Daylight = False) = 0 W, 0 spread.
  * For every issue hour it builds the "best available" forecast for the next 50 hours from all issues
    made at or before that hour (Same-Day t+1..t+8, Next-Day t+24..t+31; the gap t+9..t+23 is covered by the
    Next-Day issues of the previous hours), then the 15-minute state features:
        Same-Day ensemble/spread for the next 8 h, Next-Day ensemble/spread (tomorrow's horizon),
        PV energy left today and tomorrow (kWh) with summed spread, last-hour forecast error,
        cloudiness now (dataset GHI / clear-sky GHI) and forecast cloudiness.
  * Perfect-forecast features (forecast = actual, spread 0) are built next to them; the no-forecast
    agent gets zeros. Stress episodes scale the real error up (ratio measured on winter folds).
  * Also loads the dataset (hourly PV truth PV_DC_Power_W, Temperature, Relative Humidity, GHI) and checks
    that its timestamps are UTC by comparing Solar_Zenith_rad with the computed sun position.

Run in Colab:
    !python hems_forecast_feed.py --selftest            # synthetic fixtures, no Drive data needed
    !python hems_forecast_feed.py --build               # real data: builds and caches train/select/test feeds, prints checks
    !python hems_forecast_feed.py --make-synthetic DIR  # writes stand-in inputs for a full dry run
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hems_config import Config, SelfTest, add_common_args, config_from_args, make_config, new_run_dir, synthetic_overrides

HOUR = np.timedelta64(1, "h")
QUARTER = np.timedelta64(15, "m")
D_BEST = 50  # hours of best-available forecast kept per issue hour
FC_NAMES = ([f"sd_ens_{k}" for k in range(1, 9)] + [f"sd_spr_{k}" for k in range(1, 9)]
            + [f"nd_ens_{k}" for k in range(1, 9)] + [f"nd_spr_{k}" for k in range(1, 9)]
            + ["e_today_kwh", "e_today_spr_kwh", "e_tom_kwh", "e_tom_spr_kwh", "last_hour_err_w", "cloud_fc_8h", "cloud_fc_tom"])
FC_SPREAD = np.array([("_spr" in n) for n in FC_NAMES])
FORECAST_COLUMNS = ["House", "Issue_Timestamp", "Is_First_Issue", "Slot", "Lead_Minutes", "Target_Timestamp", "Daylight",
                    "Valid_Target", "Actual_W", "Predicted_W", "Raw_Model_Output_pu", "Raw_Model_Output_W",
                    "Array_Rated_Power_W", "Weather_Run_Init", "Weather_Run_Age_At_Issue_h", "Weather_Lead_From_Run_Start_h"]


# =====================================================================================
# Solar geometry (NOAA equations) - day/night flag, sunrise/sunset, clear-sky GHI
# =====================================================================================
def _noaa(doy, hour_utc):
    g = 2.0 * np.pi / 365.0 * (doy - 1 + (hour_utc - 12.0) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    return eqt, decl


def cos_zenith(t_utc, lat, lon):
    t = pd.DatetimeIndex(np.asarray(t_utc, dtype="datetime64[ns]"))
    doy = t.dayofyear.values.astype(float)
    hour = t.hour.values + t.minute.values / 60.0
    eqt, decl = _noaa(doy, hour)
    tst = hour * 60.0 + eqt + 4.0 * lon
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)


def sunrise_sunset_utc_min(dates, lat, lon):
    """Sunrise and sunset in minutes after UTC midnight for each UTC date."""
    d = pd.DatetimeIndex(np.asarray(dates, dtype="datetime64[ns]"))
    eqt, decl = _noaa(d.dayofyear.values.astype(float), 12.0)
    la = np.radians(lat)
    x = np.cos(np.radians(90.833)) / (np.cos(la) * np.cos(decl)) - np.tan(la) * np.tan(decl)
    ha = np.degrees(np.arccos(np.clip(x, -1, 1)))
    return 720.0 - 4.0 * (lon + ha) - eqt, 720.0 - 4.0 * (lon - ha) - eqt


def clearsky_ghi(cosz):
    c = np.clip(cosz, 1e-6, None)
    return np.where(cosz > 0, 1098.0 * c * np.exp(-0.057 / c), 0.0)


# =====================================================================================
# Time helpers
# =====================================================================================
def to_utc_naive(values, assume_tz="UTC") -> np.ndarray:
    """Parse timestamps (tz-aware or naive strings) to naive UTC datetime64[ns]."""
    s = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
    if s.dt.tz is None:
        if assume_tz and assume_tz.upper() != "UTC":
            s = s.dt.tz_localize(assume_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        else:
            return s.values.astype("datetime64[ns]")
    else:
        s = s.dt.tz_convert("UTC")
    return s.dt.tz_localize(None).values.astype("datetime64[ns]")


def _bool(series) -> np.ndarray:
    if series.dtype == bool:
        return series.values
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "t"]).values


def split_bounds(split):
    s0 = np.datetime64(split[0], "ns")
    s1 = np.datetime64(split[1], "ns") + np.timedelta64(1, "D")  # exclusive end
    return s0, s1


# =====================================================================================
# Dataset
# =====================================================================================
def dataset_path(cfg: Config) -> str:
    return cfg.DATASET_FILE if os.path.isabs(cfg.DATASET_FILE) else os.path.join(cfg.DATA_ROOT, cfg.DATASET_FILE)


def load_dataset(cfg: Config):
    """Hourly dataset on a complete UTC grid: columns pv, temp, rh, ghi, cosz_data, cloud_obs. Returns (df, info)."""
    path = dataset_path(cfg)
    raw = pd.read_csv(path)
    need = [cfg.DATASET_TIME_COL, cfg.DATASET_PV_COL, cfg.DATASET_TEMP_COL, cfg.DATASET_RH_COL]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise KeyError(f"{path}: missing columns {miss}; available: {list(raw.columns)}")
    t = to_utc_naive(raw[cfg.DATASET_TIME_COL], cfg.DATA_TZ)
    df = pd.DataFrame({"pv": pd.to_numeric(raw[cfg.DATASET_PV_COL], errors="coerce").values,
                       "temp": pd.to_numeric(raw[cfg.DATASET_TEMP_COL], errors="coerce").values,
                       "rh": pd.to_numeric(raw[cfg.DATASET_RH_COL], errors="coerce").values}, index=pd.DatetimeIndex(t))
    if cfg.DATASET_GHI_COL in raw.columns:
        df["ghi"] = pd.to_numeric(raw[cfg.DATASET_GHI_COL], errors="coerce").values
    if cfg.DATASET_ZENITH_COL in raw.columns:
        df["zen"] = pd.to_numeric(raw[cfg.DATASET_ZENITH_COL], errors="coerce").values
    if cfg.DATASET_CLOUD_COL and cfg.DATASET_CLOUD_COL in raw.columns:
        df["cloud_col"] = pd.to_numeric(raw[cfg.DATASET_CLOUD_COL], errors="coerce").values / 100.0
    df = df[~df.index.isna()]
    n_dup = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(df.index.min().floor("h"), df.index.max().ceil("h"), freq="h")
    n_missing = int(len(full) - len(df.index.intersection(full)))
    df = df.reindex(full)
    df = df.interpolate(limit=3, limit_direction="both")
    df["pv"] = df["pv"].fillna(0.0).clip(lower=0.0)
    df["temp"] = df["temp"].ffill().bfill()
    df["rh"] = df["rh"].ffill().bfill().clip(1, 100)
    info = {"path": path, "rows": int(len(raw)), "start": str(df.index.min()), "end": str(df.index.max()),
            "duplicates_dropped": n_dup, "missing_hours_filled": n_missing}
    # UTC check: the dataset zenith must match the computed sun position at the timestamp
    if "zen" in df.columns:
        sel = df["zen"].notna().values & (np.cos(df["zen"].values) > 0.2)
        idx = np.flatnonzero(sel)[:: max(1, int(sel.sum() // 3000))]
        errs = {}
        for sh in range(-3, 4):
            cz = cos_zenith(df.index.values[idx] + np.timedelta64(sh, "h"), cfg.HOUSE_LAT, cfg.HOUSE_LON)
            errs[sh] = float(np.degrees(np.nanmean(np.abs(np.arccos(np.clip(cz, -1, 1)) - df["zen"].values[idx]))))
        best = min(errs, key=errs.get)
        info["zenith_check_deg_by_shift_h"] = errs
        info["timestamps_utc_ok"] = bool(best == 0 and errs[0] < 1.5)
        if not info["timestamps_utc_ok"]:
            print(f"WARNING: dataset zenith fits a {best:+d} h shift best ({errs}); check DATA_TZ")
    cz = cos_zenith(df.index.values + np.timedelta64(30, "m"), cfg.HOUSE_LAT, cfg.HOUSE_LON)
    df["cosz"] = cz
    if "cloud_col" in df.columns:
        cloud = df["cloud_col"]
    elif "ghi" in df.columns:
        cz0 = cos_zenith(df.index.values, cfg.HOUSE_LAT, cfg.HOUSE_LON)
        csg = clearsky_ghi(cz0)
        cloud = pd.Series(np.where(cz0 > 0.15, np.clip(1.0 - df["ghi"].values / np.maximum(csg, 1.0), 0.0, 1.0), np.nan), index=df.index)
    else:
        cloud = pd.Series(np.nan, index=df.index)
    df["cloud_obs"] = cloud.ffill().fillna(0.5).values
    return df, info


def clearsky_pv_table(ds: pd.DataFrame, cutoff, q: float, window: int) -> np.ndarray:
    """Empirical clear-sky PV [doy 0..366, utc hour 0..23] from past PV (timestamps <= cutoff)."""
    past = ds.loc[ds.index <= cutoff, "pv"]
    if len(past) < 24 * 30:
        past = ds["pv"]
    days = past.index.normalize()
    mat = pd.DataFrame({"d": days, "h": past.index.hour, "v": past.values}).pivot_table(index="d", columns="h", values="v", aggfunc="mean")
    mat = mat.reindex(columns=range(24))
    doys = pd.DatetimeIndex(mat.index).dayofyear.values
    vals = mat.values
    table = np.zeros((367, 24))
    for d in range(1, 367):
        dist = np.minimum(np.abs(doys - d), 366 - np.abs(doys - d))
        w = window
        sel = dist <= w
        while sel.sum() < 5 and w < 183:
            w *= 2
            sel = dist <= w
        table[d] = np.nan_to_num(np.nanquantile(vals[sel], q, axis=0)) if sel.any() else 0.0
    table[0] = table[1]
    return table


# =====================================================================================
# Forecast file discovery
# =====================================================================================
def _walk_dirs(root, max_depth):
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath.count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        dirnames.sort()
        yield dirpath, dirnames


_DIR_INDEX = {}


def _dir_index(root, max_depth):
    """name -> [paths] of all folders below root (one walk per root; Drive folders are slow to list)."""
    key = (os.path.abspath(root), max_depth)
    if key not in _DIR_INDEX:
        idx = {}
        for dirpath, dirnames in _walk_dirs(root, max_depth):
            for d in dirnames:
                idx.setdefault(d, []).append(os.path.join(dirpath, d))
        _DIR_INDEX[key] = idx
    return _DIR_INDEX[key]


def find_dirs(root, name, max_depth):
    out = list(_dir_index(root, max_depth).get(name, []))
    if os.path.basename(os.path.abspath(root)) == name:
        out.insert(0, os.path.abspath(root))
    return out


def _find_files(root, filename):
    out = []
    for dirpath, _, files in os.walk(root):
        if filename in files:
            out.append(os.path.join(dirpath, filename))
    return sorted(out)


_FOLD_RE = re.compile(r"(?:^|_)cv(\d+)(?:_|$)")


def discover_forecast_files(cfg: Config) -> dict:
    """{split: {model: {horizon: [paths sorted by fold]}}} plus a 'messages' list."""
    _DIR_INDEX.clear()  # always look at the current folder tree
    res = {"train": {}, "select": {}, "test": {}, "messages": []}
    for m in cfg.FORECAST_MODELS:
        for split in ("train", "select", "test"):
            res[split][m] = {}
        cvdirs = find_dirs(cfg.RESULTS_DIR, cfg.CV_FOLDERS[m], cfg.SEARCH_MAX_DEPTH)
        if not cvdirs:
            res["messages"].append(f"{m}: CV folder {cfg.CV_FOLDERS[m]} not found below {cfg.RESULTS_DIR}")
        parent, run = cfg.BASELINE_RUNS[m]
        bdirs = [d for p in find_dirs(cfg.RESULTS_DIR, parent, cfg.SEARCH_MAX_DEPTH) for d in find_dirs(p, run, 4)]
        if not bdirs:
            res["messages"].append(f"{m}: baseline run {parent}/{run} not found below {cfg.RESULTS_DIR}")
        for h in ("same_day", "next_day"):
            fname = cfg.FORECAST_FILE_PATTERN.format(kind="validation", horizon=h)
            by_fold = {}
            for cv in cvdirs[:1]:
                for p in _find_files(cv, fname):
                    rel = os.path.relpath(p, cv).split(os.sep)
                    folds = [int(mm.group(1)) for part in rel for mm in [_FOLD_RE.search(part)] if mm]
                    if folds:
                        by_fold.setdefault(folds[0], []).append(p)
            res["train"][m][h] = [by_fold[k][0] for k in sorted(by_fold) if k in cfg.CV_TRAIN_FOLDS]
            res["select"][m][h] = [by_fold[k][0] for k in sorted(by_fold) if k == cfg.CV_SELECT_FOLD]
            for k in list(cfg.CV_TRAIN_FOLDS) + [cfg.CV_SELECT_FOLD]:
                if k not in by_fold and cvdirs:
                    res["messages"].append(f"{m} {h}: fold {k} validation file not found")
            tname = cfg.FORECAST_FILE_PATTERN.format(kind="test_evaluation", horizon=h)
            tf = [p for b in bdirs[:1] for p in _find_files(b, tname)]
            res["test"][m][h] = tf[:1]
            if bdirs and not tf:
                res["messages"].append(f"{m} {h}: {tname} not found in {bdirs[0]}")
    return res


# =====================================================================================
# Reading and joining forecasts
# =====================================================================================
def read_forecast_file(path: str, cfg: Config) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    need = ["Issue_Timestamp", "Target_Timestamp", "Predicted_W", "Actual_W", "Daylight", "Valid_Target"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{path}: missing columns {miss}")
    info = {"path": path, "rows": int(len(df))}
    if "House" in df.columns and df["House"].nunique() > 1:
        keep = df["House"].value_counts().index[0]
        info["house_filter"] = str(keep)
        df = df[df["House"] == keep]
    valid = _bool(df["Valid_Target"])
    info["valid_rows"] = int(valid.sum())
    df = df[valid]
    out = pd.DataFrame({"issue": to_utc_naive(df["Issue_Timestamp"].values, cfg.DATA_TZ),
                        "target": to_utc_naive(df["Target_Timestamp"].values, cfg.DATA_TZ),
                        "pred": pd.to_numeric(df["Predicted_W"], errors="coerce").values,
                        "actual": pd.to_numeric(df["Actual_W"], errors="coerce").values,
                        "daylight": _bool(df["Daylight"])})
    if "Lead_Minutes" in df.columns:
        out["lead_min_file"] = pd.to_numeric(df["Lead_Minutes"], errors="coerce").values
    out = out.dropna(subset=["issue", "target", "pred"])
    if "lead_min_file" in out.columns:
        lead = (out["target"] - out["issue"]).dt.total_seconds().values / 60.0
        info["lead_minutes_mismatch"] = int(np.nansum(np.abs(lead - out["lead_min_file"].values) > 1))
        out = out.drop(columns=["lead_min_file"])
    return out.reset_index(drop=True), info


def load_model_horizon(paths, cfg: Config):
    frames, infos = [], []
    for p in paths:
        f, i = read_forecast_file(p, cfg)
        frames.append(f)
        infos.append(i)
    if not frames:
        return pd.DataFrame(columns=["issue", "target", "pred", "actual", "daylight"]), {"files": [], "duplicates_dropped": 0}
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(subset=["issue", "target"], keep="last")  # later folds win on overlaps
    return df, {"files": infos, "duplicates_dropped": int(n0 - len(df))}


def join_ensemble(per_model: dict) -> tuple[pd.DataFrame, dict]:
    """Inner join on issue + target; ensemble = mean, spread = std (ddof=0); night = 0 W, 0 spread."""
    models = list(per_model)
    j = None
    for m in models:
        d = per_model[m][["issue", "target", "pred", "actual", "daylight"]].rename(
            columns={"pred": f"pred_{m}", "actual": f"actual_{m}", "daylight": f"day_{m}"})
        j = d if j is None else j.merge(d, on=["issue", "target"], how="inner")
    sizes = {m: int(len(per_model[m])) for m in models}
    if j is None or len(j) == 0:
        return pd.DataFrame(columns=["issue", "target", "lead_h", "ens", "spread", "actual", "daylight"]), {"rows_per_model": sizes, "joined": 0}
    P = j[[f"pred_{m}" for m in models]].values.astype(float)
    ens = P.mean(axis=1)
    spr = P.std(axis=1, ddof=0)
    day = j[f"day_{models[0]}"].values.astype(bool)
    ens = np.where(day, ens, 0.0)
    spr = np.where(day, spr, 0.0)
    A = j[[f"actual_{m}" for m in models]].values.astype(float)
    out = pd.DataFrame({"issue": j["issue"].values, "target": j["target"].values,
                        "lead_h": ((j["target"] - j["issue"]).dt.total_seconds().values / 3600.0).round().astype(int),
                        "ens": ens, "spread": spr, "actual": A[:, 0], "daylight": day})
    info = {"rows_per_model": sizes, "joined": int(len(out)),
            "dropped_by_inner_join": int(max(sizes.values()) - len(out)),
            "actual_max_disagreement_w": float(np.nanmax(np.abs(A - A[:, :1]))) if A.size else 0.0}
    return out, info


# =====================================================================================
# Split data: everything the environment needs, at 15-minute resolution
# =====================================================================================
@dataclass
class SplitData:
    name: str
    t_utc: np.ndarray  # datetime64[ns], step starts
    hour_local: np.ndarray  # local hour incl. fraction
    doy_local: np.ndarray
    month_local: np.ndarray
    day_id_local: np.ndarray  # integer local date (days since epoch)
    weekday_local: np.ndarray
    slots_to_midnight: np.ndarray
    pv_w: np.ndarray  # true PV (DC), 15 min
    t_out: np.ndarray
    rh_out: np.ndarray
    cloud_now: np.ndarray
    is_day: np.ndarray
    min_to_sunrise: np.ndarray
    min_to_sunset: np.ndarray
    fc_ens: np.ndarray  # n x F forecast features from the real ensemble
    fc_perf: np.ndarray  # n x F forecast features with forecast = actual
    issue_covered: np.ndarray
    valid_start: np.ndarray
    pv_hourly_cs: np.ndarray = field(default=None)  # clear-sky PV at the step's hour (for scenarios)
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.t_utc)

    def save(self, path):
        arrs = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        arrs["t_utc"] = self.t_utc.astype("int64")
        np.savez_compressed(path, name=self.name, meta=json.dumps(self.meta, default=str), **arrs)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files if k not in ("name", "meta")}
        kw["t_utc"] = kw["t_utc"].astype("datetime64[ns]")
        return SplitData(name=str(z["name"]), meta=json.loads(str(z["meta"])), **kw)

    def window(self, s0: int, s1: int) -> "SplitData":
        kw = {k: (v[s0:s1] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return SplitData(**kw)


def _local_time_arrays(t_utc, tz):
    """Local wall-clock features. Works on naive wall time, so DST days (Lebanon switches at midnight) are safe."""
    idx = pd.DatetimeIndex(t_utc).tz_localize("UTC").tz_convert(tz)
    wall = idx.tz_localize(None)
    hour = wall.hour.values + wall.minute.values / 60.0
    day = wall.normalize()
    slots = np.rint((day + pd.Timedelta(days=1) - wall).total_seconds().values / 900.0).astype(int)
    day_id = day.values.astype("datetime64[D]").astype(np.int64)
    return hour, wall.dayofyear.values, wall.month.values, day_id, wall.weekday.values, np.clip(slots, 1, 100)


def _upsample_slots(v, mode):
    """(n, D) hourly -> (n, 4D) 15-min values."""
    if mode == "linear":
        nxt = np.concatenate([v[:, 1:], v[:, -1:]], axis=1)
        f = np.array([0.0, 0.25, 0.5, 0.75])
        return (v[:, :, None] + (nxt - v)[:, :, None] * f[None, None, :]).reshape(v.shape[0], -1)
    return np.repeat(v, 4, axis=1)


def _best_available(sd_ens, sd_spr, nd_ens, nd_spr, D=D_BEST):
    """Best available forecast for target hour i+d at issue hour i (latest issue <= i wins)."""
    nh = sd_ens.shape[0]
    be = np.full((nh, D), np.nan)
    bs = np.full((nh, D), np.nan)
    for d in range(0, 32):
        for m in range(0, 32 - d):
            lead = d + m
            if 1 <= lead <= 8:
                src_e, src_s, col = sd_ens, sd_spr, lead - 1
            elif 24 <= lead <= 31:
                src_e, src_s, col = nd_ens, nd_spr, lead - 24
            else:
                continue
            if m == 0:
                ve, vs = src_e[:, col], src_s[:, col]
            else:
                ve = np.concatenate([np.full(m, np.nan), src_e[:-m, col]])
                vs = np.concatenate([np.full(m, np.nan), src_s[:-m, col]])
            fill = np.isnan(be[:, d]) & ~np.isnan(ve)
            be[fill, d] = ve[fill]
            bs[fill, d] = vs[fill]
    for d in range(32, D):  # beyond any issue: day-ahead persistence of the best forecast
        be[:, d] = be[:, d - 24]
        bs[:, d] = bs[:, d - 24]
    return be, bs


def build_split_from_joined(cfg: Config, name: str, split, ds: pd.DataFrame, sd: pd.DataFrame, nd: pd.DataFrame,
                            join_info: dict | None = None) -> SplitData:
    s0, s1 = split_bounds(split)
    g0 = s0 - np.timedelta64(2, "D")
    g1 = s1 + np.timedelta64(3, "D")
    hours = np.arange(g0, g1, HOUR)
    nh = len(hours)
    dsr = ds.reindex(pd.DatetimeIndex(hours))
    pv_h = dsr["pv"].fillna(0.0).values
    cosz_mid = cos_zenith(hours + np.timedelta64(30, "m"), cfg.HOUSE_LAT, cfg.HOUSE_LON)
    day_h = (cosz_mid > 0) | (pv_h > 0)

    def place(df, leads):
        ens = np.full((nh, len(leads)), np.nan)
        spr = np.full((nh, len(leads)), np.nan)
        if len(df):
            ii = ((df["issue"].values - g0) // HOUR).astype(np.int64)
            ll = df["lead_h"].values.astype(int)
            ok = (ii >= 0) & (ii < nh) & (ll >= leads[0]) & (ll <= leads[-1])
            ens[ii[ok], ll[ok] - leads[0]] = df["ens"].values[ok]
            spr[ii[ok], ll[ok] - leads[0]] = df["spread"].values[ok]
            tt = ((df["target"].values - g0) // HOUR).astype(np.int64)
            okt = (tt >= 0) & (tt < nh)
            day_h[tt[okt]] = df["daylight"].values[okt].astype(bool)  # the forecast files' Daylight flag wins
        return ens, spr

    sd_e, sd_s = place(sd, cfg.SD_LEADS_H)
    nd_e, nd_s = place(nd, cfg.ND_LEADS_H)
    covered = ~np.all(np.isnan(sd_e), axis=1)
    be, bs = _best_available(sd_e, sd_s, nd_e, nd_s)
    # night = 0 W / 0 spread; remaining daylight gaps -> past actual (same hour, most recent day before the issue)
    tidx = np.arange(nh)[:, None] + np.arange(D_BEST)[None, :]
    tidx_c = np.clip(tidx, 0, nh - 1)
    night = ~day_h[tidx_c]
    gap = np.isnan(be) & ~night
    k = (np.arange(D_BEST)[None, :] + 1 + 23) // 24
    past = np.clip(tidx - 24 * k, 0, nh - 1)
    n_gap = int(gap.sum())
    be[gap] = pv_h[past][gap]
    bs[gap] = 0.0
    be[night] = 0.0
    bs[night] = 0.0
    be = np.nan_to_num(be)
    bs = np.nan_to_num(bs)
    bp = np.where(night, 0.0, pv_h[tidx_c])  # perfect forecast
    covered |= np.all(night[:, 1:9], axis=1)

    # clear-sky PV (empirical, from the past only)
    cutoff = pd.Timestamp(cfg.SPLIT_TRAIN[1]) + pd.Timedelta(days=1)
    table = clearsky_pv_table(ds, cutoff, cfg.CLEARSKY_QUANTILE, cfg.CLEARSKY_DOY_WINDOW)
    hidx = pd.DatetimeIndex(hours)
    pvcs_h = table[hidx.dayofyear.values, hidx.hour.values]

    # 15-minute steps
    t = np.arange(s0, s1, QUARTER)
    n = len(t)
    ih = ((t - g0) // HOUR).astype(np.int64)
    q = ((t - (g0 + ih * HOUR)) // QUARTER).astype(np.int64)
    hour_l, doy_l, mon_l, day_id, wd, nm = _local_time_arrays(t, cfg.LOCAL_TZ)

    def feats(B, S):
        Bs = _upsample_slots(B, cfg.FORECAST_UPSAMPLE)
        Ss = _upsample_slots(S, cfg.FORECAST_UPSAMPLE)
        cb = np.concatenate([np.zeros((nh, 1)), np.cumsum(Bs, axis=1)], axis=1)
        cs = np.concatenate([np.zeros((nh, 1)), np.cumsum(Ss, axis=1)], axis=1)
        a, b, c = q, np.minimum(q + nm, 4 * D_BEST), np.minimum(q + nm + 96, 4 * D_BEST)
        e_today = (cb[ih, b] - cb[ih, a]) * 0.25 / 1000.0
        e_today_s = (cs[ih, b] - cs[ih, a]) * 0.25 / 1000.0
        e_tom = (cb[ih, c] - cb[ih, b]) * 0.25 / 1000.0
        e_tom_s = (cs[ih, c] - cs[ih, b]) * 0.25 / 1000.0
        prev = np.clip(ih - 1, 0, nh - 1)
        err = pv_h[prev] - B[prev, 0]
        # forecast cloudiness = 1 - forecast PV / clear-sky PV
        cs_rows = pvcs_h[np.clip(ih[:, None] + np.arange(D_BEST)[None, :], 0, nh - 1)]
        num8 = B[ih, 1:9].sum(axis=1)
        den8 = cs_rows[:, 1:9].sum(axis=1)
        cslot = np.concatenate([np.zeros((n, 1)), np.cumsum(np.repeat(cs_rows, 4, axis=1), axis=1)], axis=1)
        rows = np.arange(n)
        den_t = (cslot[rows, c] - cslot[rows, b]) * 0.25 / 1000.0
        cl_t = np.clip(1.0 - e_tom / np.maximum(den_t, 1e-3), 0.0, 1.0)
        cl_8 = np.where(den8 > 50.0, np.clip(1.0 - num8 / np.maximum(den8, 1e-3), 0.0, 1.0), cl_t)
        return np.column_stack([B[ih, 1:9], S[ih, 1:9], B[ih, 24:32], S[ih, 24:32],
                                e_today, e_today_s, e_tom, e_tom_s, err, cl_8, cl_t]).astype(np.float32)

    fc_ens = feats(be, bs)
    fc_perf = feats(bp, np.zeros_like(bp))
    if cfg.CLOUD_FORECAST_SOURCE == "dataset_future":
        cl = dsr["cloud_obs"].ffill().fillna(0.5).values
        ci = np.clip(ih[:, None] + np.arange(1, 9)[None, :], 0, nh - 1)
        dd = day_h[ci]
        c8 = np.where(dd.any(axis=1), (cl[ci] * dd).sum(axis=1) / np.maximum(dd.sum(axis=1), 1), 0.5)
        for F in (fc_ens, fc_perf):
            F[:, FC_NAMES.index("cloud_fc_8h")] = c8
    # weather, PV truth
    xh = (t - g0) / HOUR
    if cfg.FORECAST_UPSAMPLE == "linear":
        pv = np.interp(xh, np.arange(nh), pv_h)
    else:
        pv = pv_h[ih]
    t_out = np.interp(xh, np.arange(nh), dsr["temp"].ffill().bfill().values)
    rh_out = np.interp(xh, np.arange(nh), dsr["rh"].ffill().bfill().values)
    cloud_now = dsr["cloud_obs"].ffill().fillna(0.5).values[ih]
    cz = cos_zenith(t + np.timedelta64(450, "s"), cfg.HOUSE_LAT, cfg.HOUSE_LON)
    dates = t.astype("datetime64[D]")
    ud = np.unique(dates)
    rise, sset = sunrise_sunset_utc_min(np.concatenate([ud, ud[-1:] + np.timedelta64(1, "D")]), cfg.HOUSE_LAT, cfg.HOUSE_LON)
    di = np.searchsorted(ud, dates)
    mins = (t - dates.astype("datetime64[ns]")) / np.timedelta64(1, "m")
    to_rise = np.where(mins < rise[di], rise[di] - mins, rise[di + 1] + 1440.0 - mins)
    to_set = np.where(mins < sset[di], sset[di] - mins, sset[di + 1] + 1440.0 - mins)
    cov_steps = covered[ih]
    L = cfg.episode_steps + 1
    cum = np.concatenate([[0], np.cumsum(cov_steps)])
    vs = np.zeros(n, dtype=bool)
    if n > L:
        frac = (cum[L:] - cum[:-L]) / L
        vs[: n - L + 1] = frac >= cfg.FORECAST_MIN_COVERAGE
    meta = {"split": name, "dates": list(split), "steps": int(n), "issue_hours_covered_frac": float(cov_steps.mean()) if n else 0.0,
            "valid_episode_starts": int(vs.sum()), "daylight_gaps_filled_with_past_actual": n_gap,
            "join": join_info or {}, "upsample": cfg.FORECAST_UPSAMPLE, "cloud_forecast_source": cfg.CLOUD_FORECAST_SOURCE}
    # forecast quality on this split (Same-Day slot 1 and Next-Day slot 1, daylight steps)
    dmask = fc_perf[:, 0] > 0
    if dmask.any():
        meta["sd1_mae_w"] = float(np.mean(np.abs(fc_ens[dmask, 0] - fc_perf[dmask, 0])))
        meta["nd1_mae_w"] = float(np.mean(np.abs(fc_ens[dmask, 16] - fc_perf[dmask, 16])))
    if len(sd):
        dd = sd[sd["daylight"].astype(bool)]
        if len(dd):
            mon = pd.DatetimeIndex(dd["target"].values).tz_localize("UTC").tz_convert(cfg.LOCAL_TZ).month
            err = np.abs(dd["ens"].values - dd["actual"].values)
            w = np.isin(mon, [12, 1, 2])
            meta["mae_all_w"] = float(err.mean())
            meta["mae_winter_w"] = float(err[w].mean()) if w.any() else None
            meta["winter_error_ratio"] = float(err[w].mean() / max(err.mean(), 1e-6)) if w.any() else 1.0
    return SplitData(name=name, t_utc=t, hour_local=hour_l.astype(np.float32), doy_local=doy_l.astype(np.int16),
                     month_local=mon_l.astype(np.int8), day_id_local=day_id.astype(np.int32), weekday_local=wd.astype(np.int8),
                     slots_to_midnight=nm.astype(np.int16), pv_w=pv.astype(np.float32), t_out=t_out.astype(np.float32),
                     rh_out=rh_out.astype(np.float32), cloud_now=cloud_now.astype(np.float32), is_day=(cz > 0),
                     min_to_sunrise=to_rise.astype(np.float32), min_to_sunset=to_set.astype(np.float32),
                     fc_ens=fc_ens, fc_perf=fc_perf, issue_covered=cov_steps, valid_start=vs,
                     pv_hourly_cs=pvcs_h[ih].astype(np.float32), meta=meta)


def _split_dates(cfg, split):
    return {"train": cfg.SPLIT_TRAIN, "select": cfg.SPLIT_SELECT, "test": cfg.SPLIT_TEST}[split]


def build_split(cfg: Config, split: str, ds=None, files=None) -> SplitData:
    if ds is None:
        ds, dinfo = load_dataset(cfg)
    else:
        dinfo = {}
    files = files or discover_forecast_files(cfg)
    joins, jinfo = {}, {"dataset": dinfo, "messages": files.get("messages", [])}
    for h in ("same_day", "next_day"):
        per_model = {}
        for m in cfg.FORECAST_MODELS:
            paths = files[split][m][h]
            if not paths:
                raise FileNotFoundError(f"no {h} forecast files for model {m}, split {split}. Messages: {files.get('messages')}")
            per_model[m], info = load_model_horizon(paths, cfg)
            jinfo[f"{m}_{h}"] = info
        joins[h], jinfo[f"join_{h}"] = join_ensemble(per_model)
    return build_split_from_joined(cfg, split, _split_dates(cfg, split), ds, joins["same_day"], joins["next_day"], jinfo)


def _cache_key(cfg: Config, split: str) -> str:
    keys = ["DATA_ROOT", "DATASET_FILE", "RESULTS_DIR", "CV_FOLDERS", "BASELINE_RUNS", "FORECAST_MODELS", "SPLIT_TRAIN",
            "SPLIT_SELECT", "SPLIT_TEST", "CV_TRAIN_FOLDS", "CV_SELECT_FOLD", "FORECAST_UPSAMPLE", "CLOUD_FORECAST_SOURCE",
            "CLEARSKY_QUANTILE", "CLEARSKY_DOY_WINDOW", "FORECAST_MIN_COVERAGE", "EPISODE_DAYS", "LOCAL_TZ", "DATA_TZ",
            "DATASET_CLOUD_COL", "HOUSE_LAT", "HOUSE_LON"]
    d = {k: getattr(cfg, k) for k in keys}
    d["split"] = split
    d["v"] = 3
    try:
        d["mtime"] = os.path.getmtime(dataset_path(cfg))
    except OSError:
        pass
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def load_or_build(cfg: Config, split: str, rebuild: bool = False, verbose: bool = True) -> SplitData:
    folder = os.path.join(cfg.OUTPUT_DIR, "feeds")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{split}_{_cache_key(cfg, split)}.npz")
    if os.path.exists(path) and not rebuild:
        return SplitData.load(path)
    if verbose:
        print(f"building {split} feed ...")
    sd = build_split(cfg, split)
    sd.save(path)
    with open(path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(sd.meta, f, indent=1, default=str)
    return sd


# =====================================================================================
# Synthetic stand-in inputs (same folder layout and columns as the real files)
# =====================================================================================
def make_synthetic_inputs(cfg: Config, seed: int = 0, start="2023-06-01", end="2024-08-05", overlap_days: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    root = cfg.DATA_ROOT
    os.makedirs(root, exist_ok=True)
    hours = np.arange(np.datetime64(start, "ns"), np.datetime64(end, "ns"), HOUR)
    n = len(hours)
    idx = pd.DatetimeIndex(hours)
    cz0 = cos_zenith(hours, cfg.HOUSE_LAT, cfg.HOUSE_LON)
    ghi_cs = clearsky_ghi(cz0)
    ndays = n // 24 + 2
    kd = np.empty(ndays)
    kd[0] = 0.9
    for i in range(1, ndays):
        kd[i] = np.clip(0.75 * kd[i - 1] + 0.25 * rng.choice([0.95, 0.9, 0.6, 0.3], p=[0.55, 0.25, 0.12, 0.08]) + rng.normal(0, 0.05), 0.15, 1.0)
    day_i = (hours - hours[0]) // np.timedelta64(1, "D")
    k = np.clip(kd[day_i] + rng.normal(0, 0.05, n), 0.05, 1.05)
    ghi = ghi_cs * k
    doy = idx.dayofyear.values
    lh = (idx.hour.values + 2.5) % 24
    tmean = 18.0 - 8.0 * np.cos(2 * np.pi * (doy - 20) / 365.0)
    amp = 4.0 + 2.5 * (1 - np.cos(2 * np.pi * (doy - 20) / 365.0)) / 2
    temp = tmean + amp * np.sin(2 * np.pi * (lh - 9.0) / 24.0) + rng.normal(0, 0.6, n)
    rh = np.clip(62 + 12 * np.cos(2 * np.pi * (doy - 20) / 365.0) - 1.8 * (temp - tmean) + rng.normal(0, 4, n), 15, 98)
    pv = np.clip(cfg.ARRAY_RATED_W * 0.82 * ghi / 1000.0 * (1 - 0.0034 * (temp + 20 * ghi / 800 - 25)), 0, cfg.ARRAY_RATED_W)
    ds = pd.DataFrame({"Timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"), "Pmax_Per_Panel_W": 450.0, "Number_of_Panels": 6,
                       "Array_Rated_Power_W": cfg.ARRAY_RATED_W, "Gamma_Pmax_per_C": -0.0034333, "Panel_Tilt_rad": 0.5236,
                       "Panel_Azimuth_rad": 3.1416, "Temperature": temp.round(1), "Relative Humidity": rh.round(0),
                       "GHI": ghi.round(1), "DNI": (ghi * 1.2).round(1), "DHI": (ghi * 0.25).round(1), "Wind Speed": 1.5,
                       "Solar_Zenith_rad": np.arccos(np.clip(cz0, -1, 1)), "Solar_Azimuth_rad": 2.0, "PV_DC_Power_W": pv})
    ds.to_csv(dataset_path(cfg), index=False)
    pv_s = pd.Series(pv, index=idx)
    cz_mid = cos_zenith(hours + np.timedelta64(30, "m"), cfg.HOUSE_LAT, cfg.HOUSE_LON)
    day_s = pd.Series(cz_mid > 0, index=idx)

    def periods(split, nfold):
        s0, s1 = split_bounds(split)
        edges = [s0 + (s1 - s0) * i // nfold for i in range(nfold + 1)]
        return [(np.datetime64(edges[i], "h"), np.datetime64(edges[i + 1], "h")) for i in range(nfold)]

    def write(path, p0, p1, mi, naive_ts, drop_issue=None):
        issues = np.arange(np.datetime64(p0, "ns"), np.datetime64(p1, "ns"), HOUR)
        rows = []
        for h, leads in (("same_day", cfg.SD_LEADS_H), ("next_day", cfg.ND_LEADS_H)):
            I = np.repeat(issues, len(leads))
            L = np.tile(np.array(leads), len(issues))
            Tt = I + L * HOUR
            ok = (Tt >= hours[0]) & (Tt <= hours[-1])
            I, L, Tt = I[ok], L[ok], Tt[ok]
            act = pv_s.reindex(pd.DatetimeIndex(Tt)).values
            dl = day_s.reindex(pd.DatetimeIndex(Tt)).values.astype(bool)
            dayerr = np.random.default_rng([seed, mi, int(p0.astype(np.int64)) % 99991]).normal(0, 0.12, len(I))
            bias = [0.02, -0.03, 0.0][mi]
            pred = np.clip(act * (1 + bias + dayerr * (1 + L / 40.0)) + rng.normal(0, 30, len(I)), 0, cfg.ARRAY_RATED_W)
            pred = np.where(dl, pred, 0.0)
            valid = rng.random(len(I)) > 0.01
            if drop_issue is not None:
                valid &= I != drop_issue
            fmt = (lambda a: pd.DatetimeIndex(a).strftime("%Y-%m-%d %H:%M:%S")) if naive_ts else (lambda a: pd.DatetimeIndex(a).strftime("%Y-%m-%d %H:%M:%S+00:00"))
            df = pd.DataFrame({"House": "my_house", "Issue_Timestamp": fmt(I), "Is_First_Issue": False, "Slot": L - L.min() + 1,
                               "Lead_Minutes": L * 60, "Target_Timestamp": fmt(Tt), "Daylight": dl, "Valid_Target": valid,
                               "Actual_W": act, "Predicted_W": pred, "Raw_Model_Output_pu": pred / cfg.ARRAY_RATED_W,
                               "Raw_Model_Output_W": pred, "Array_Rated_Power_W": cfg.ARRAY_RATED_W,
                               "Weather_Run_Init": fmt(I - 6 * HOUR), "Weather_Run_Age_At_Issue_h": 6.0,
                               "Weather_Lead_From_Run_Start_h": 6.0 + L})
            rows.append((h, df))
        return rows

    written = []
    for mi, m in enumerate(cfg.FORECAST_MODELS):
        cv = os.path.join(root, "results_synthetic", cfg.CV_FOLDERS[m])
        stem = cfg.CV_FOLDERS[m].replace("cv_", "run_", 1)
        tr = periods(cfg.SPLIT_TRAIN, len(cfg.CV_TRAIN_FOLDS))
        folds = list(zip(cfg.CV_TRAIN_FOLDS, tr)) + [(cfg.CV_SELECT_FOLD, periods(cfg.SPLIT_SELECT, 1)[0])]
        for fold, (p0, p1) in folds:
            if fold == 2:
                p0 = p0 - np.timedelta64(24 * overlap_days, "h")  # overlap with fold 1 (dedupe test)
            fdir = os.path.join(cv, f"{stem}_cv{fold}_synthetic", "forecasts")
            os.makedirs(fdir, exist_ok=True)
            drop = np.datetime64(p0, "ns") + 30 * HOUR if (mi == 1 and fold == 1) else None
            for h, df in write(fdir, p0, p1, mi, naive_ts=True, drop_issue=drop):
                fp = os.path.join(fdir, cfg.FORECAST_FILE_PATTERN.format(kind="validation", horizon=h))
                df.to_csv(fp, index=False, compression="gzip")
                written.append(fp)
        parent, run = cfg.BASELINE_RUNS[m]
        bdir = os.path.join(root, "results_synthetic", parent, run, "forecasts")
        os.makedirs(bdir, exist_ok=True)
        p0, p1 = periods(cfg.SPLIT_TEST, 1)[0]
        for h, df in write(bdir, p0, p1, mi, naive_ts=False):
            fp = os.path.join(bdir, cfg.FORECAST_FILE_PATTERN.format(kind="test_evaluation", horizon=h))
            df.to_csv(fp, index=False, compression="gzip")
            written.append(fp)
    return {"dataset": dataset_path(cfg), "forecast_files": written}


def ensure_inputs(cfg: Config, synthetic: bool):
    """For --synthetic runs: generate the stand-in inputs once."""
    if synthetic and not os.path.exists(dataset_path(cfg)):
        print(f"generating synthetic inputs in {cfg.DATA_ROOT} ...")
        make_synthetic_inputs(cfg)


# =====================================================================================
# Self-test
# =====================================================================================
def selftest(out_root: str | None = None) -> bool:
    T = SelfTest("hems_forecast_feed")
    tmp = tempfile.mkdtemp(prefix="hems_ff_")
    try:
        cfg = make_config(**synthetic_overrides(tmp))
        info = make_synthetic_inputs(cfg, seed=1)
        T.check(len(info["forecast_files"]) == 3 * (5 + 1) * 2, "synthetic inputs written with the real folder layout", str(len(info["forecast_files"])))
        # solar geometry vs known values (Beirut-like longitude, June 21 sunrise ~02:30 UTC, sunset ~16:50 UTC)
        r, s = sunrise_sunset_utc_min(np.array(["2024-06-21"], dtype="datetime64[D]"), cfg.HOUSE_LAT, cfg.HOUSE_LON)
        T.check(abs(r[0] - 148) < 8 and abs(s[0] - 1010) < 8, "sunrise/sunset at the house on June 21 (UTC)", f"{r[0]:.0f} {s[0]:.0f} min")
        ds, dinfo = load_dataset(cfg)
        T.check(dinfo.get("timestamps_utc_ok", False), "dataset timestamps are UTC (zenith check)", str(dinfo.get("zenith_check_deg_by_shift_h")))
        files = discover_forecast_files(cfg)
        T.check(all(len(files["train"][m][h]) == 4 and len(files["select"][m][h]) == 1 and len(files["test"][m][h]) == 1
                    for m in cfg.FORECAST_MODELS for h in ("same_day", "next_day")) and not files["messages"],
                "discovery: folds 1-4 for train, fold 5 for select, baseline test files", str(files["messages"]))
        # join
        per = {m: load_model_horizon(files["train"][m]["same_day"], cfg) for m in cfg.FORECAST_MODELS}
        T.check(all(v[1]["duplicates_dropped"] > 0 for v in per.values()), "overlapping folds deduplicated")
        raw = {m: v[0] for m, v in per.items()}
        j, jinfo = join_ensemble(raw)
        T.check(jinfo["dropped_by_inner_join"] > 0, "inner join drops rows missing in one model", str(jinfo["dropped_by_inner_join"]))
        row = j[j["daylight"] & (j["ens"] > 300)].iloc[5]
        vals = [raw[m].set_index(["issue", "target"]).loc[(row["issue"], row["target"]), "pred"] for m in cfg.FORECAST_MODELS]
        T.check(abs(row["ens"] - np.mean(vals)) < 1e-6 and abs(row["spread"] - np.std(vals, ddof=0)) < 1e-6, "ensemble = mean, spread = std ddof=0")
        T.check(bool((j.loc[~j["daylight"], ["ens", "spread"]].values == 0).all()), "night rows are 0 W with 0 spread")
        f0 = pd.read_csv(files["train"]["lstm"]["same_day"][0])
        T.check(len(raw["lstm"]) < len(f0) * 4, "Valid_Target filter applied")
        # build splits
        sd_train = build_split(cfg, "train", ds=ds, files=files)
        T.check(sd_train.fc_ens.shape[1] == len(FC_NAMES) and sd_train.n == 30 * 96, "train feed: 43 forecast features per 15-min step")
        T.check(sd_train.valid_start.sum() > 0 and sd_train.meta["issue_hours_covered_frac"] > 0.95, "coverage and valid episode starts",
                f"{sd_train.meta['issue_hours_covered_frac']:.3f}")
        # perfect mode = actual
        i = int(np.flatnonzero(sd_train.fc_perf[:, 0] > 500)[10])
        tgt = sd_train.t_utc[i].astype("datetime64[h]") + np.timedelta64(1, "h")
        T.check(abs(sd_train.fc_perf[i, 0] - ds.loc[pd.Timestamp(tgt), "pv"]) < 1e-3, "perfect forecast = actual PV of the target hour")
        T.check(np.all(sd_train.fc_perf[:, FC_SPREAD] == 0), "perfect forecast has zero spread")
        # causality: forecasts issued after X must not change features before X
        sdj = {}
        for h in ("same_day", "next_day"):
            pm = {m: load_model_horizon(files["train"][m][h], cfg)[0] for m in cfg.FORECAST_MODELS}
            sdj[h] = join_ensemble(pm)[0]
        base = build_split_from_joined(cfg, "train", cfg.SPLIT_TRAIN, ds, sdj["same_day"], sdj["next_day"])
        X = np.datetime64("2024-06-15T10:00", "ns")
        mod = {h: d.assign(ens=np.where(d["issue"].values > X, d["ens"].values * 2 + 100 * d["daylight"].values, d["ens"].values))
               for h, d in sdj.items()}
        pert = build_split_from_joined(cfg, "train", cfg.SPLIT_TRAIN, ds, mod["same_day"], mod["next_day"])
        before = base.t_utc < X + np.timedelta64(1, "h")
        same_before = np.allclose(base.fc_ens[before], pert.fc_ens[before])
        T.check(same_before and not np.allclose(base.fc_ens[~before], pert.fc_ens[~before]), "no look-ahead: features use only issues <= now")
        # Next-Day slot k at issue i is the forecast for i+24+k-1 issued at i
        jn = sdj["next_day"].set_index(["issue", "target"])
        k = int(np.flatnonzero((base.fc_ens[:, 16] > 300) & (base.t_utc.astype("datetime64[m]").astype(int) % 60 == 0))[3])
        iss = base.t_utc[k]
        T.check(abs(base.fc_ens[k, 16] - jn.loc[(iss, iss + 24 * HOUR), "ens"]) < 1e-3, "Next-Day slot 1 = t+24 h forecast of the current issue")
        # energies: hold mode = sum of hourly values; linear mode changes only the shape
        cfg_lin = cfg.replace(FORECAST_UPSAMPLE="linear")
        lin = build_split_from_joined(cfg_lin, "train", cfg.SPLIT_TRAIN, ds, sdj["same_day"], sdj["next_day"])
        e = FC_NAMES.index("e_tom_kwh")
        T.check(np.all(base.fc_ens[:, e] >= 0) and np.corrcoef(base.fc_ens[:, e], lin.fc_ens[:, e])[0, 1] > 0.98, "energy tomorrow: hold and linear consistent")
        et = FC_NAMES.index("e_today_kwh")
        ev = base.fc_ens[:, et]
        T.check(ev.max() < 30 and ev.min() >= 0, "energy left today in a plausible range (kWh)", f"max {ev.max():.1f}")
        nt = base.t_utc.astype("datetime64[m]")
        lh = base.hour_local
        T.check(np.all(ev[lh >= 21] < 1e-6) and ev[lh < 1].max() > 1.0, "no PV energy left today after sunset; a full day after midnight")
        T.check(base.meta.get("winter_error_ratio") is not None, "winter error ratio measured for stress episodes")
        # save / load round trip and cache
        p = os.path.join(tmp, "x.npz")
        base.save(p)
        back = SplitData.load(p)
        T.check(np.array_equal(back.fc_ens, base.fc_ens) and np.array_equal(back.t_utc, base.t_utc), "SplitData save/load round trip")
        sel = load_or_build(cfg, "select", verbose=False)
        tst = load_or_build(cfg, "test", verbose=False)
        T.check(sel.n == 9 * 96 and tst.n == 15 * 96 and tst.meta["issue_hours_covered_frac"] > 0.95, "select and test feeds build (baseline test forecasts)")
        T.check(np.all(np.isfinite(tst.fc_ens)) and np.all(np.isfinite(sd_train.pv_w)), "all features finite")
        del nt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = new_run_dir(make_config(), "selftest", "hems_forecast_feed", base=out_root) if out_root else None
    return T.summary(out)


def build_real(cfg: Config) -> bool:
    run = new_run_dir(cfg, "feeds", "build")
    ds, dinfo = load_dataset(cfg)
    print(json.dumps(dinfo, indent=1, default=str))
    files = discover_forecast_files(cfg)
    for msg in files["messages"]:
        print("WARNING:", msg)
    ok = True
    summary = {"dataset": dinfo, "files": {s: files[s] for s in ("train", "select", "test")}}
    for split in ("train", "select", "test"):
        try:
            sd = load_or_build(cfg, split, rebuild=True)
            summary[split] = sd.meta
            print(f"{split}: {sd.n} steps, covered {sd.meta['issue_hours_covered_frac']:.3f}, valid starts {sd.meta['valid_episode_starts']}, "
                  f"SD1 MAE {sd.meta.get('sd1_mae_w', float('nan')):.0f} W, ND1 MAE {sd.meta.get('nd1_mae_w', float('nan')):.0f} W")
        except Exception as ex:  # report and continue with the other splits
            ok = False
            print(f"ERROR building {split}: {ex}")
            summary[split] = {"error": str(ex)}
    with open(os.path.join(run, "feed_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("summary saved in", run)
    return ok


if __name__ == "__main__":
    from hems_config import _default_selftest_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true", help="build and cache the train/select/test feeds from the real data")
    ap.add_argument("--make-synthetic", default=None, metavar="DIR")
    ap.add_argument("--out", default=None)
    add_common_args(ap)
    a = ap.parse_args()
    if a.make_synthetic:
        c = make_config(**synthetic_overrides(a.make_synthetic))
        print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in make_synthetic_inputs(c).items()}, indent=1))
    elif a.build:
        c = config_from_args(a)
        ensure_inputs(c, bool(a.synthetic))
        sys.exit(0 if build_real(c) else 1)
    else:
        sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
