# =============================================================================
# combined_xgboost.py
# COMBINED SAME-DAY + NEXT-DAY PV FORECASTING - XGBoost - ONE COMPLETE EXPERIMENT
# =============================================================================
# Every execution is ONE complete, independent experiment:
#   load data -> preprocess -> train on Houses 01-09 -> validate on Houses 10-11
#   -> choose/checkpoint the best boosting rounds (per slot) on the GLOBAL validation monitor
#   -> save model -> reload saved model -> verify saved/reloaded predictions
#   -> Test evaluation on Houses 12-13 -> all metrics -> all plots
#   -> everything saved into a NEW unique run folder (an earlier run is never overwritten).
#
# Manual experimentation: edit the hyperparameters in SECTION 1 and rerun the complete
# file. There is NO automatic hyperparameter search, model ranking or architecture
# selection. The PRIMARY selection result is the best validation val_monitor (LOWER IS
# BETTER); validation macro nRMSE / WAPE / R2 and Houses 10 and 11 individually are
# secondary diagnostics. Never select hyperparameters from Houses 12-13.
#
# Split: training Houses 01-09 | validation Houses 10-11 (official hyperparameter
#        selection) | "Test evaluation" Houses 12-13 (visible during manual
#        experimentation; never used for training, scaler fitting, early stopping,
#        checkpoint selection or hyperparameter selection; opened only after the
#        model has been saved, reloaded and verified).
#
# ORACLE-WEATHER PV forecasting: the future weather inputs are REALIZED weather values
# at the target times, so weather-forecast uncertainty is NOT represented in these results.
#
# Inputs   : 8 target-time weather/solar features + 4 static PV-system features
#            (Gamma_Pmax_per_C, Number_of_Panels, Pmax_Per_Panel_W, Array_Rated_Power_W).
#            Same-Day ALSO receives measured PV history rows [PV_kW, daylight_flag]
#            at t-(LOOKBACK-1)*15 min ... t. Next-Day receives NO PV of any kind.
# Target   : PV_DC_Power_W (kW inside the model; all forecasts and metrics in W).
# Daylight : Solar_Zenith_rad < pi/2 is the only rule. Night forecasts are exactly 0 W;
#            negative forecasts are clipped to 0 W.
# Schedule : S = first daylight 15-min grid point of the day, H = floor(S to the hour);
#            issues S, H+2h, H+4h, ... while the issue time is daylight and both tasks
#            have at least one valid daylight target. Same-Day: t+15 ... t+360 (24).
#            Next-Day: t+24h ... t+30h (25); the first issue S has 24 real targets
#            (S ... S+5h45 tomorrow) + 1 structural padding slot at the END.
# Selection: val_monitor = 0.5*GlobalMaskedMSE_SD/Var_SD + 0.5*GlobalMaskedMSE_ND/Var_ND
#            on Houses 10-11 with GlobalMaskedMSE = SUM(mask*error^2)/SUM(mask) over ALL
#            valid target points; variances from valid training targets (Houses 01-09).
# Model    : task/slot XGBoost regressors (24 Same-Day + 25 Next-Day) under one combined interface.
# =============================================================================

# =============================================================================
# SECTION 1 - MANUAL HYPERPARAMETERS (edit here, then rerun the complete file)
# =============================================================================
ARCH_KEY = "xgboost"
ARCH_LABEL = "XGBoost"

LOOKBACK = 16                  # Same-Day PV history length in 15-min steps (1 ... 96)
N_ESTIMATORS = 1000            # maximum boosting rounds per slot regressor
EARLY_STOPPING_ROUNDS = 50     # early stopping on Houses 10-11 only (None = use all N_ESTIMATORS)
LEARNING_RATE = 0.05
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 1.0
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
REG_ALPHA = 0.0
GAMMA = 0.0
MAX_BIN = 256
XGB_DEVICE = "auto"            # "auto" (GPU when available), "cuda" or "cpu"
SEED = 42

SHOW_PLOTS_IN_COLAB = True     # False: every plot is still saved, but not displayed

DATA_DIR = "/content/drive/MyDrive/Mutli-House"
RESULTS_DIR = "/content/drive/MyDrive/Mutli-House/results_combined_xgboost"

MANUAL_HYPERPARAMETERS = {
    "LOOKBACK": LOOKBACK,
    "N_ESTIMATORS": N_ESTIMATORS,
    "EARLY_STOPPING_ROUNDS": EARLY_STOPPING_ROUNDS,
    "LEARNING_RATE": LEARNING_RATE,
    "MAX_DEPTH": MAX_DEPTH,
    "MIN_CHILD_WEIGHT": MIN_CHILD_WEIGHT,
    "SUBSAMPLE": SUBSAMPLE,
    "COLSAMPLE_BYTREE": COLSAMPLE_BYTREE,
    "REG_LAMBDA": REG_LAMBDA,
    "REG_ALPHA": REG_ALPHA,
    "GAMMA": GAMMA,
    "MAX_BIN": MAX_BIN,
    "SEED": SEED,
}

# =============================================================================
# SECTION 2 - IMPORTS AND RUNTIME
# =============================================================================
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PYTHONHASHSEED"] = str(SEED)   # recorded for child processes; seeds are set explicitly below

import hashlib
import json
import math
import platform
import random
import sys
import textwrap
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # figures are saved to Drive; display uses the saved PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# =============================================================================
# SECTION 3 - FIXED EXPERIMENT DEFINITIONS (identical in all five programs)
# =============================================================================
HOUSE_FILE_PATTERN = "house_{house_id:02d}_model_dataset.csv"

TRAIN_HOUSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
VALIDATION_HOUSES = [10, 11]
TEST_HOUSES = [12, 13]

SPLIT_HOUSES = {"validation": VALIDATION_HOUSES, "test_evaluation": TEST_HOUSES}
SPLIT_LABELS = {
    "validation": "Validation (Houses 10-11)",
    "test_evaluation": "Test evaluation (Houses 12-13)",
}

TIMESTAMP_COL = "Timestamp"
TARGET_COL = "PV_DC_Power_W"
ZENITH_COL = "Solar_Zenith_rad"
HOUSE_ID_COL = "House_ID"

# Weather / solar inputs: EXACTLY these 8 (target-time values).
WEATHER_FEATURES = [
    "Temperature",
    "Relative Humidity",
    "GHI",
    "DNI",
    "DHI",
    "Wind Speed",
    "Solar_Zenith_rad",
    "Solar_Azimuth_rad",
]

# Static PV-system inputs: EXACTLY these 4.
# Array_Rated_Power_W = Number_of_Panels * Pmax_Per_Panel_W is mathematically
# redundant; the redundancy is intentional and identical in all five architectures.
STATIC_FEATURES = [
    "Gamma_Pmax_per_C",
    "Number_of_Panels",
    "Pmax_Per_Panel_W",
    "Array_Rated_Power_W",
]
RATED_POWER_COL = "Array_Rated_Power_W"
PANELS_COL = "Number_of_Panels"
PMAX_COL = "Pmax_Per_Panel_W"
RATED_POWER_RTOL = 1e-9    # tight check: Array_Rated_Power_W ~= Number_of_Panels * Pmax_Per_Panel_W
RATED_POWER_ATOL_W = 1e-6

# Columns that may exist in the files but must NEVER be model inputs.
NON_INPUT_COLUMNS = [
    "Panel_Tilt_rad",
    "Panel_Azimuth_rad",
    "Is_Bifacial",
    "Surface_Albedo",
    "House_ID",
]

STEP_MINUTES = 15
STEP = pd.Timedelta(minutes=STEP_MINUTES)
STEP_NS = STEP_MINUTES * 60 * 10**9          # one 15-min step in nanoseconds
STEP_HOURS = STEP_MINUTES / 60.0
STEPS_PER_DAY = 96
COMMON_MAX_LOOKBACK_STEPS = 96          # 96 samples = t-23h45 ... t (not t-24h ... t)

SD_SLOTS = 24                            # Same-Day targets t+15 min ... t+360 min
ND_SLOTS = 25                            # Next-Day slots   t+24 h   ... t+30 h
ND_FIRST_ISSUE_REAL_SLOTS = 24           # first issue S: S tomorrow ... S+5h45 tomorrow (+1 padding slot)
ND_PADDING_SLOT = ND_SLOTS - 1           # the structural padding slot is always the LAST slot
ND_LEAD_STEPS = 96                       # Next-Day slot 0 = t + 24 h
ISSUE_CADENCE = pd.Timedelta(hours=2)    # S, then H+2h, H+4h, ...

DAYLIGHT_ZENITH_LIMIT_RAD = math.pi / 2.0   # the ONLY daylight rule: Solar_Zenith_rad < pi/2
W_PER_KW = 1000.0
MAPE_MIN_FRACTION_OF_RATED = 0.01        # MAPE_1pct: actual PV >= 1 % of that house's rated power

TASKS = ("same_day", "next_day")
TASK_PREFIX = {"same_day": "sd", "next_day": "nd"}
TASK_SLOTS = {"same_day": SD_SLOTS, "next_day": ND_SLOTS}
TASK_LABELS = {
    "same_day": "Same-Day (t+15 min ... t+6 h)",
    "next_day": "Next-Day (t+24 h ... t+30 h)",
}

EVAL_BATCH_SIZE = 1024                   # inference batch size (does not affect training)
TITLE_WRAP_WIDTH = 80                    # ~8-inch-wide figures
FOOTER_WRAP_WIDTH = 120
PLOT_DPI = 150

POINT_METRIC_KEYS = ["RMSE_W", "MAE_W", "nRMSE_pct", "WAPE_pct", "R2", "sMAPE_pct", "MAPE_1pct_pct"]
HEMS_METRIC_KEYS = [
    "Energy_MAE_kWh", "Energy_RMSE_kWh", "Energy_Bias_kWh", "Energy_WAPE_pct",
    "Peak_Power_MAE_W", "Peak_Power_Bias_W", "Peak_Time_MAE_min", "Peak_Time_Bias_min",
]

ORACLE_WEATHER_STATEMENT = (
    "ORACLE-WEATHER PV forecasting: the future weather inputs are REALIZED weather values "
    "at the target times, so weather-forecast uncertainty is NOT represented in these results."
)

VAL_MONITOR_DEFINITION = (
    "val_monitor = 0.5 * GlobalMaskedMSE_SameDay_validation / Variance_SameDay_training "
    "+ 0.5 * GlobalMaskedMSE_NextDay_validation / Variance_NextDay_training, where "
    "GlobalMaskedMSE = SUM(mask * error^2) / SUM(mask) over ALL valid target points "
    "(daylight, real, non-padding) of Houses 10-11, error in kW on the raw model output. "
    "Variances are population variances (kW^2) of all valid TRAINING target points of the Houses 01-09 "
    "windows (a target time contained in several windows is counted once per window, exactly as in the loss). "
    "LOWER IS BETTER. Official criterion for early stopping, checkpoint/best-epoch selection "
    "and manual hyperparameter comparison."
)

RUN_STATE = {"run_id": None, "run_dir": None}
HOUSE_ACCESS = {"test_unlocked": False, "log": []}


# =============================================================================
# SECTION 4 - SMALL UTILITIES
# =============================================================================
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, indent=2, default=json_default)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def section(title):
    print("\n" + "=" * 79)
    print(title)
    print("=" * 79)


def hyperparameter_text():
    return " | ".join(f"{k}={v}" for k, v in MANUAL_HYPERPARAMETERS.items())


def ns_index(values):
    """DatetimeIndex with nanosecond resolution (identical behaviour in pandas 2.x and 3.x)."""
    return pd.DatetimeIndex(pd.to_datetime(values)).as_unit("ns")


def is_daylight(zenith_rad):
    """The ONLY daylight rule of the experiment: Solar_Zenith_rad < pi/2."""
    return np.asarray(zenith_rad, dtype=np.float64) < DAYLIGHT_ZENITH_LIMIT_RAD


def lead_minutes(task):
    if task == "same_day":
        return (np.arange(SD_SLOTS) + 1) * STEP_MINUTES               # 15 ... 360
    return ND_LEAD_STEPS * STEP_MINUTES + np.arange(ND_SLOTS) * STEP_MINUTES  # 1440 ... 1800


def mount_google_drive_if_colab():
    """Mount Google Drive when running inside Colab (no-op elsewhere)."""
    if Path("/content/drive/MyDrive").exists():
        return "already_mounted"
    try:
        from google.colab import drive  # noqa: WPS433
    except ImportError:
        return "not_colab"
    try:
        drive.mount("/content/drive")
        return "mounted"
    except Exception as exc:  # e.g. executed with !python (no notebook kernel)
        print(f"WARNING: Google Drive mount failed ({exc!r}).")
        return "mount_failed"


def validate_static_configuration():
    """Fail fast if the fixed feature/split definitions were edited inconsistently."""
    assert len(WEATHER_FEATURES) == 8 and len(set(WEATHER_FEATURES)) == 8
    assert len(STATIC_FEATURES) == 4 and len(set(STATIC_FEATURES)) == 4
    assert PANELS_COL in STATIC_FEATURES, "Number_of_Panels must remain a static input."
    assert "Solar_Azimuth_rad" in WEATHER_FEATURES, "Solar_Azimuth_rad must remain an input."
    assert "Panel_Azimuth_rad" not in WEATHER_FEATURES + STATIC_FEATURES
    for col in NON_INPUT_COLUMNS:
        assert col not in WEATHER_FEATURES + STATIC_FEATURES, f"{col} must not be an input."
    assert TARGET_COL not in WEATHER_FEATURES + STATIC_FEATURES
    assert ZENITH_COL in WEATHER_FEATURES
    assert not set(TRAIN_HOUSES) & set(VALIDATION_HOUSES)
    assert not set(TRAIN_HOUSES) & set(TEST_HOUSES)
    assert not set(VALIDATION_HOUSES) & set(TEST_HOUSES)
    assert TRAIN_HOUSES == list(range(1, 10)) and VALIDATION_HOUSES == [10, 11] and TEST_HOUSES == [12, 13]
    if not (isinstance(LOOKBACK, int) and 1 <= LOOKBACK <= COMMON_MAX_LOOKBACK_STEPS):
        raise ValueError(f"LOOKBACK must be an integer in [1, {COMMON_MAX_LOOKBACK_STEPS}], got {LOOKBACK!r}.")


# =============================================================================
# SECTION 5 - DATA LOADING AND DATA-INTEGRITY CHECKS
# =============================================================================
def house_file_path(house_id):
    return Path(DATA_DIR) / HOUSE_FILE_PATTERN.format(house_id=house_id)


def check_rated_power_consistency(rated_w, n_panels, pmax_w):
    """Array_Rated_Power_W must equal Number_of_Panels * Pmax_Per_Panel_W (tight tolerance)."""
    expected = float(n_panels) * float(pmax_w)
    abs_diff = abs(float(rated_w) - expected)
    ok = abs_diff <= RATED_POWER_ATOL_W + RATED_POWER_RTOL * abs(expected)
    return bool(ok), expected, abs_diff


def load_house(house_id, phase):
    """Load and strictly validate one house file.

    Houses 12-13 can only be read after the Test-evaluation phase has been unlocked,
    i.e. after training, checkpoint selection, saving, reloading and verification.
    """
    if house_id in TEST_HOUSES and not HOUSE_ACCESS["test_unlocked"]:
        raise RuntimeError(
            f"House {house_id:02d} is a Test-evaluation house and is locked until the model has been "
            "trained, selected on Houses 10-11, saved, reloaded and verified."
        )
    if house_id not in TRAIN_HOUSES + VALIDATION_HOUSES + TEST_HOUSES:
        raise ValueError(f"Unknown house {house_id}.")
    path = house_file_path(house_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    HOUSE_ACCESS["log"].append({"house": house_id, "phase": phase, "utc": utc_now_iso(), "path": str(path)})

    df = pd.read_csv(path)
    required = [TIMESTAMP_COL, TARGET_COL] + WEATHER_FEATURES + STATIC_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"House {house_id:02d}: missing required columns {missing}")

    report = {
        "house": house_id,
        "file": str(path),
        "sha256": file_sha256(path),
        "n_rows": int(len(df)),
        "columns_present": list(map(str, df.columns)),
        "non_input_columns_present": [c for c in NON_INPUT_COLUMNS if c in df.columns],
        "non_input_columns_used_as_inputs": [],
    }

    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="raise").dt.as_unit("ns")
    if ts.isna().any():
        raise ValueError(f"House {house_id:02d}: unparsable timestamps.")
    if ts.duplicated().any():
        raise ValueError(f"House {house_id:02d}: duplicated timestamps.")
    if not ts.is_monotonic_increasing:
        raise ValueError(f"House {house_id:02d}: timestamps are not increasing.")
    diffs = ts.diff().iloc[1:]
    if not (diffs == STEP).all():
        bad = ts.iloc[1:][diffs != STEP].head(5).tolist()
        raise ValueError(f"House {house_id:02d}: not continuous {STEP_MINUTES}-minute data (first gaps at {bad}).")
    aligned = (ts.dt.minute % STEP_MINUTES == 0) & (ts.dt.second == 0) & (ts.dt.microsecond == 0)
    if not aligned.all():
        raise ValueError(f"House {house_id:02d}: timestamps are not on the 15-minute clock grid.")

    numeric_cols = [TARGET_COL] + WEATHER_FEATURES + STATIC_FEATURES
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="raise").astype(np.float64)
    values = numeric.to_numpy()
    n_nonfinite = int((~np.isfinite(values)).sum())
    if n_nonfinite:
        raise ValueError(f"House {house_id:02d}: {n_nonfinite} NaN/Inf values in required columns.")

    if HOUSE_ID_COL in df.columns:
        ids = pd.to_numeric(df[HOUSE_ID_COL], errors="coerce").unique()
        report["house_id_column_values"] = [float(v) for v in ids]
        if len(ids) != 1 or float(ids[0]) != float(house_id):
            raise ValueError(f"House {house_id:02d}: House_ID column {ids} does not match the file name.")

    static_raw = []
    for col in STATIC_FEATURES:
        uniq = numeric[col].unique()
        if len(uniq) != 1:
            raise ValueError(f"House {house_id:02d}: static feature {col} is not constant inside the file.")
        static_raw.append(float(uniq[0]))
    static_raw = np.asarray(static_raw, dtype=np.float64)
    static_map = dict(zip(STATIC_FEATURES, static_raw.tolist()))

    rated_w = static_map[RATED_POWER_COL]
    n_panels = static_map[PANELS_COL]
    pmax_w = static_map[PMAX_COL]
    if rated_w <= 0 or n_panels <= 0 or pmax_w <= 0:
        raise ValueError(f"House {house_id:02d}: rated power, panel count and Pmax must be positive.")
    if abs(n_panels - round(n_panels)) > 1e-9:
        raise ValueError(f"House {house_id:02d}: Number_of_Panels={n_panels} is not an integer.")
    ok, expected, abs_diff = check_rated_power_consistency(rated_w, n_panels, pmax_w)
    if not ok:
        raise ValueError(
            f"House {house_id:02d}: Array_Rated_Power_W={rated_w} differs from Number_of_Panels*Pmax_Per_Panel_W="
            f"{expected} by {abs_diff} W (tolerance rtol={RATED_POWER_RTOL}, atol={RATED_POWER_ATOL_W} W)."
        )

    pv_w = numeric[TARGET_COL].to_numpy(np.float64)
    zenith = numeric[ZENITH_COL].to_numpy(np.float64)
    daylight = is_daylight(zenith)
    n_negative = int((pv_w < 0).sum())
    if n_negative:
        raise ValueError(f"House {house_id:02d}: {n_negative} negative {TARGET_COL} values.")
    n_night_nonzero = int(((~daylight) & (pv_w != 0)).sum())
    if n_night_nonzero:
        raise ValueError(
            f"House {house_id:02d}: {n_night_nonzero} night rows (Solar_Zenith_rad >= pi/2) have non-zero PV. "
            "The Same-Day history encoding requires night rows to be exactly [0, 0]."
        )

    report.update({
        "first_timestamp": ts.iloc[0],
        "last_timestamp": ts.iloc[-1],
        "n_calendar_days": int(ts.dt.normalize().nunique()),
        "continuous_15min": True,
        "on_15min_clock_grid": True,
        "n_nonfinite_required_values": 0,
        "n_negative_pv": 0,
        "n_night_rows": int((~daylight).sum()),
        "n_daylight_rows": int(daylight.sum()),
        "n_night_rows_with_nonzero_pv": 0,
        "n_daylight_rows_with_zero_pv": int((daylight & (pv_w == 0)).sum()),
        "pv_max_w": float(pv_w.max()),
        "pv_max_fraction_of_rated": float(pv_w.max() / rated_w),
        "static_values": static_map,
        "rated_power_check": {
            "Array_Rated_Power_W": rated_w,
            "Number_of_Panels_x_Pmax_Per_Panel_W": expected,
            "abs_difference_W": abs_diff,
            "passed": ok,
        },
    })

    return {
        "house_id": house_id,
        "timestamps": ns_index(ts),
        "weather_raw": numeric[WEATHER_FEATURES].to_numpy(np.float64),
        "zenith": zenith,
        "daylight": daylight,
        "pv_w": pv_w,
        "pv_kw": (pv_w / W_PER_KW).astype(np.float32),
        "static_raw": static_raw,
        "rated_w": rated_w,
        "integrity": report,
    }


# =============================================================================
# SECTION 6 - SCALERS (fitted on Houses 01-09 ONLY)
# =============================================================================
def fit_scalers(houses):
    loaded_train = sorted(h for h in houses if h in TRAIN_HOUSES)
    if loaded_train != TRAIN_HOUSES:
        raise RuntimeError(f"Scalers must be fitted on Houses 01-09 exactly; got {loaded_train}.")
    weather_rows = np.vstack([houses[h]["weather_raw"] for h in TRAIN_HOUSES])
    static_rows = np.vstack([houses[h]["static_raw"][None, :] for h in TRAIN_HOUSES])
    weather_scaler = StandardScaler().fit(weather_rows)
    static_scaler = StandardScaler().fit(static_rows)
    metadata = {
        "fitted_on_houses": TRAIN_HOUSES,
        "validation_houses_used_for_fitting": False,
        "test_houses_used_for_fitting": False,
        "weather_scaler": {
            "type": "sklearn.preprocessing.StandardScaler",
            "fitted_on_houses": TRAIN_HOUSES,
            "features": WEATHER_FEATURES,
            "n_rows": int(weather_rows.shape[0]),
            "mean": weather_scaler.mean_.tolist(),
            "scale": weather_scaler.scale_.tolist(),
        },
        "static_scaler": {
            "type": "sklearn.preprocessing.StandardScaler",
            "fitted_on_houses": TRAIN_HOUSES,
            "features": STATIC_FEATURES,
            "n_rows": int(static_rows.shape[0]),
            "mean": static_scaler.mean_.tolist(),
            "scale": static_scaler.scale_.tolist(),
        },
    }
    metadata["fingerprint_sha256"] = scaler_fingerprint(weather_scaler, static_scaler)
    return weather_scaler, static_scaler, metadata


def scaler_fingerprint(weather_scaler, static_scaler):
    return array_sha256(weather_scaler.mean_, weather_scaler.scale_, static_scaler.mean_, static_scaler.scale_)


def apply_scalers(house, weather_scaler, static_scaler):
    house["weather_scaled"] = weather_scaler.transform(house["weather_raw"]).astype(np.float32)
    house["static_scaled"] = static_scaler.transform(house["static_raw"][None, :])[0].astype(np.float32)


# =============================================================================
# SECTION 7 - JOINT ISSUE SCHEDULE (one rule for training, evaluation and deployment)
# =============================================================================
def build_joint_issue_schedule(timestamps, daylight, history_steps_required):
    """Joint Same-Day / Next-Day issue schedule.

    For every calendar day present from 00:00:
      S = first 15-minute grid point of the day with Solar_Zenith_rad < pi/2
      H = floor(S to the clock hour)
      candidate issues: S, H+2h, H+4h, ... (same day, after S)
      the candidate sequence stops at the first issue time that is not daylight;
      an issue is kept only if
        - its Same-Day history (history_steps_required samples ending at t) is inside the data,
        - all real Same-Day and Next-Day targets are inside the data,
        - Same-Day has >= 1 daylight target AND Next-Day has >= 1 real daylight target.
    The first issue S has 24 real Next-Day targets (S ... S+5h45 tomorrow) plus one
    structural padding slot at the END; every later issue has 25 real Next-Day slots.

    Returns (accepted DataFrame, rejected DataFrame).
    """
    ts = ns_index(timestamps)
    daylight = np.asarray(daylight, dtype=bool)
    n = len(ts)
    if n == 0:
        return _empty_schedule(), pd.DataFrame(columns=["Candidate_Timestamp", "Reason"])
    if n > 1 and not (np.diff(ts.asi8) == STEP_NS).all():
        raise ValueError("Schedule construction requires continuous 15-minute timestamps.")

    day_values = ts.normalize().asi8
    change = np.flatnonzero(np.diff(day_values)) + 1
    day_starts = np.concatenate([[0], change])
    day_stops = np.concatenate([change, [n]])

    accepted, rejected = [], []
    for start, stop in zip(day_starts, day_stops):
        day = ts[start].normalize()
        if ts[start] != day:
            rejected.append({"Candidate_Timestamp": ts[start], "Reason": "day_not_complete_from_midnight"})
            continue
        dl_pos = np.flatnonzero(daylight[start:stop])
        if len(dl_pos) == 0:
            continue
        s_idx = int(start + dl_pos[0])
        s_ts = ts[s_idx]
        h_ts = s_ts.floor("h")

        candidates = [(s_ts, True)]
        k = 1
        while True:
            c_ts = h_ts + k * ISSUE_CADENCE
            if c_ts.normalize() != day:
                break
            candidates.append((c_ts, False))
            k += 1

        for c_ts, is_first in candidates:
            if not is_first and c_ts <= s_ts:
                rejected.append({"Candidate_Timestamp": c_ts, "Reason": "not_after_S"})
                continue
            c_idx = s_idx + int((c_ts - s_ts) // STEP)
            if c_idx >= stop:
                rejected.append({"Candidate_Timestamp": c_ts, "Reason": "issue_outside_data"})
                break
            if ts[c_idx] != c_ts:
                raise RuntimeError(f"Schedule index mismatch at {c_ts}.")
            if not daylight[c_idx]:
                rejected.append({"Candidate_Timestamp": c_ts, "Reason": "issue_time_not_daylight"})
                break
            n_nd_real = ND_FIRST_ISSUE_REAL_SLOTS if is_first else ND_SLOTS
            if c_idx - (history_steps_required - 1) < 0:
                rejected.append({"Candidate_Timestamp": c_ts, "Reason": "insufficient_history"})
                continue
            last_needed = max(c_idx + SD_SLOTS, c_idx + ND_LEAD_STEPS + n_nd_real - 1)
            if last_needed >= n:
                rejected.append({"Candidate_Timestamp": c_ts, "Reason": "targets_outside_data"})
                continue
            sd_valid = int(daylight[c_idx + 1: c_idx + 1 + SD_SLOTS].sum())
            nd_valid = int(daylight[c_idx + ND_LEAD_STEPS: c_idx + ND_LEAD_STEPS + n_nd_real].sum())
            if sd_valid == 0:
                rejected.append({"Candidate_Timestamp": c_ts, "Reason": "same_day_all_targets_masked"})
                continue
            if nd_valid == 0:
                rejected.append({"Candidate_Timestamp": c_ts, "Reason": "next_day_all_targets_masked"})
                continue
            accepted.append({
                "issue_idx": c_idx,
                "Issue_Timestamp": c_ts,
                "Is_First_Issue_S": bool(is_first),
                "S_Timestamp": s_ts,
                "H_Timestamp": h_ts,
                "N_SameDay_Valid_Targets": sd_valid,
                "N_NextDay_Real_Slots": n_nd_real,
                "N_NextDay_Valid_Targets": nd_valid,
            })

    acc = pd.DataFrame(accepted) if accepted else _empty_schedule()
    rej = pd.DataFrame(rejected) if rejected else pd.DataFrame(columns=["Candidate_Timestamp", "Reason"])
    return acc, rej


def _empty_schedule():
    return pd.DataFrame(columns=[
        "issue_idx", "Issue_Timestamp", "Is_First_Issue_S", "S_Timestamp", "H_Timestamp",
        "N_SameDay_Valid_Targets", "N_NextDay_Real_Slots", "N_NextDay_Valid_Targets",
    ])


def validate_solar_frame(frame, name):
    if TIMESTAMP_COL not in frame.columns or ZENITH_COL not in frame.columns:
        raise ValueError(f"{name} must contain {TIMESTAMP_COL} and {ZENITH_COL}.")
    ts = ns_index(pd.to_datetime(frame[TIMESTAMP_COL], errors="raise"))
    if ts.has_duplicates or not ts.is_monotonic_increasing:
        raise ValueError(f"{name}: timestamps must be unique and increasing.")
    if len(ts) > 1 and not (np.diff(ts.asi8) == STEP_NS).all():
        raise ValueError(f"{name}: timestamps must be continuous {STEP_MINUTES}-minute data.")
    if ((ts.minute % STEP_MINUTES) != 0).any() or (ts.second != 0).any():
        raise ValueError(f"{name}: timestamps must lie on the 15-minute clock grid.")
    zen = pd.to_numeric(frame[ZENITH_COL], errors="raise").to_numpy(np.float64)
    if not np.isfinite(zen).all():
        raise ValueError(f"{name}: non-finite {ZENITH_COL}.")
    return ts, zen


def validate_deployment_issue_time(issue_timestamp, solar_frame):
    """Reconstruct the joint schedule from Solar_Zenith_rad and validate one issue time.

    solar_frame must cover the issue day from 00:00 (to identify S) through issue + 30 h
    (to check the Next-Day targets). Raises ValueError for any issue time that is not on
    the schedule (e.g. 09:15, 07:30 when it is not S, 10:00 when H is odd, 05:00 at night).
    """
    issue_ts = pd.Timestamp(issue_timestamp)
    ts, zen = validate_solar_frame(solar_frame, "solar_frame")
    day = issue_ts.normalize()
    if day not in ts:
        raise ValueError(f"solar_frame must contain {day} (00:00 of the issue day) to identify S.")
    accepted, rejected = build_joint_issue_schedule(ts, is_daylight(zen), history_steps_required=1)
    hit = accepted[accepted["Issue_Timestamp"] == issue_ts]
    if len(hit) == 1:
        row = hit.iloc[0]
        return {
            "issue_timestamp": issue_ts,
            "is_first_issue_S": bool(row["Is_First_Issue_S"]),
            "n_next_day_real_slots": int(row["N_NextDay_Real_Slots"]),
            "S_timestamp": row["S_Timestamp"],
        }
    valid_today = accepted.loc[
        pd.DatetimeIndex(accepted["Issue_Timestamp"]).normalize() == day, "Issue_Timestamp"
    ] if len(accepted) else []
    reason_rows = rejected[rejected["Candidate_Timestamp"] == issue_ts] if len(rejected) else rejected
    reason = reason_rows["Reason"].iloc[0] if len(reason_rows) else "not_a_schedule_candidate"
    raise ValueError(
        f"Invalid issue time {issue_ts}: {reason}. Valid issue times on {day.date()}: "
        f"{[str(pd.Timestamp(t).time()) for t in valid_today]}"
    )


def schedule_sha256(windows):
    return array_sha256(windows["house"].astype(np.int64), windows["issue_ts"].astype("datetime64[ns]").astype(np.int64))


# =============================================================================
# SECTION 8 - FORECAST WINDOWS (inputs, targets, masks)
# =============================================================================
def history_indices(issue_idx, lookback):
    """Same-Day history: t-(L-1)*15 min, ..., t-15 min, t  (oldest -> newest)."""
    offsets = np.arange(-(lookback - 1), 1, dtype=np.int64)
    return np.asarray(issue_idx, dtype=np.int64)[:, None] + offsets[None, :]


def assemble_windows(house, issue_idx, is_first, lookback):
    """Build all model inputs, targets and masks for the given issues of one house.

    Same-Day inputs : history (L, 2) = [PV_kW, daylight_flag] at t-(L-1)*15 ... t
                      + target-time weather (24, 8) at t+15 ... t+360 + static (4)
    Next-Day inputs : target-time weather (25, 8) at t+24h ... t+30h + static (4)
                      (NO PV of any kind). For the first issue S the last slot is a
                      structural padding row (zeros) with target 0 and mask 0.
    """
    issue_idx = np.asarray(issue_idx, dtype=np.int64)
    is_first = np.asarray(is_first, dtype=bool)
    n_rows = len(house["timestamps"])
    n_win = len(issue_idx)
    ts_values = house["timestamps"].values.astype("datetime64[ns]")

    hist_idx = history_indices(issue_idx, lookback)
    if n_win and hist_idx.min() < 0:
        raise RuntimeError("Same-Day history starts before the data.")
    if n_win and not np.array_equal(hist_idx[:, -1], issue_idx):
        raise RuntimeError("Same-Day history must end exactly at the issue time t.")

    sd_idx = issue_idx[:, None] + np.arange(1, SD_SLOTS + 1, dtype=np.int64)[None, :]
    nd_idx = issue_idx[:, None] + ND_LEAD_STEPS + np.arange(ND_SLOTS, dtype=np.int64)[None, :]
    nd_real = np.ones((n_win, ND_SLOTS), dtype=bool)
    nd_real[is_first, ND_PADDING_SLOT] = False
    nd_idx_safe = np.where(nd_real, nd_idx, issue_idx[:, None])   # padding never reads data
    if n_win and (sd_idx.max() >= n_rows or nd_idx_safe.max() >= n_rows):
        raise RuntimeError("Forecast targets outside the data.")

    daylight = house["daylight"]
    pv_kw = house["pv_kw"]
    sd_history = np.stack([pv_kw[hist_idx], daylight[hist_idx].astype(np.float32)], axis=-1).astype(np.float32)
    sd_weather = house["weather_scaled"][sd_idx].astype(np.float32)
    nd_weather = house["weather_scaled"][nd_idx_safe].astype(np.float32)
    nd_weather[~nd_real] = 0.0
    static = np.repeat(house["static_scaled"][None, :], n_win, axis=0).astype(np.float32)

    sd_daylight = daylight[sd_idx]
    nd_daylight = daylight[nd_idx_safe] & nd_real
    issue_ts = ts_values[issue_idx]
    sd_target_ts = ts_values[sd_idx]
    nd_target_ts = np.where(nd_real, ts_values[nd_idx_safe], np.datetime64("NaT", "ns"))

    windows = {
        "inputs": {
            "sd_history": sd_history,
            "sd_weather": sd_weather,
            "nd_weather": nd_weather,
            "static": static,
        },
        "sd_y_kw": pv_kw[sd_idx].astype(np.float32),
        "nd_y_kw": np.where(nd_real, pv_kw[nd_idx_safe], 0.0).astype(np.float32),
        "sd_actual_w": house["pv_w"][sd_idx],
        "nd_actual_w": np.where(nd_real, house["pv_w"][nd_idx_safe], 0.0),
        "sd_daylight": sd_daylight,
        "nd_daylight": nd_daylight,
        "sd_mask": sd_daylight.copy(),
        "nd_mask": nd_daylight & nd_real,
        "sd_real": np.ones((n_win, SD_SLOTS), dtype=bool),
        "nd_real": nd_real,
        "house": np.full(n_win, house["house_id"], dtype=np.int64),
        "issue_idx": issue_idx,
        "issue_ts": issue_ts,
        "is_first_issue": is_first,
        "rated_w": np.full(n_win, house["rated_w"], dtype=np.float64),
        "history_first_ts": ts_values[hist_idx[:, 0]] if n_win else np.empty(0, "datetime64[ns]"),
        "history_last_ts": ts_values[hist_idx[:, -1]] if n_win else np.empty(0, "datetime64[ns]"),
        "sd_target_ts": sd_target_ts,
        "nd_target_ts": nd_target_ts,
        "lookback": lookback,
    }
    check_window_timing(windows, lookback)
    return windows


def check_window_timing(w, lookback):
    """Hard timing / leakage checks on every window."""
    if len(w["issue_ts"]) == 0:
        return
    issue = w["issue_ts"].astype("datetime64[ns]")
    m15 = np.timedelta64(STEP_MINUTES, "m")
    if not (w["history_last_ts"] == issue).all():
        raise RuntimeError("History does not end at the issue time.")
    if not (issue - w["history_first_ts"] == (lookback - 1) * m15).all():
        raise RuntimeError("History does not start at t-(LOOKBACK-1)*15 min.")
    if not (w["sd_target_ts"][:, 0] - issue == m15).all():
        raise RuntimeError("Same-Day first target is not t+15 min.")
    if not (w["sd_target_ts"][:, -1] - issue == SD_SLOTS * m15).all():
        raise RuntimeError("Same-Day last target is not t+360 min.")
    if not (w["nd_target_ts"][:, 0] - issue == np.timedelta64(24, "h")).all():
        raise RuntimeError("Next-Day first slot is not t+24 h.")
    first = w["is_first_issue"]
    if (~first).any() and not (w["nd_target_ts"][~first, ND_SLOTS - 1] - issue[~first] == np.timedelta64(30, "h")).all():
        raise RuntimeError("Next-Day last slot of a normal issue is not t+30 h.")
    if first.any():
        last_real = w["nd_target_ts"][first, ND_FIRST_ISSUE_REAL_SLOTS - 1] - issue[first]
        if not (last_real == np.timedelta64(24 * 60 + 5 * 60 + 45, "m")).all():
            raise RuntimeError("First issue S: last real Next-Day target is not S+5h45 tomorrow.")
        if w["nd_real"][first, ND_PADDING_SLOT].any() or w["nd_mask"][first, ND_PADDING_SLOT].any():
            raise RuntimeError("First issue S: padding slot must be non-real and masked.")
        if np.any(w["inputs"]["nd_weather"][first, ND_PADDING_SLOT, :] != 0):
            raise RuntimeError("First issue S: padding row must be structural zeros.")
    if w["sd_mask"].sum(axis=1).min() < 1 or w["nd_mask"].sum(axis=1).min() < 1:
        raise RuntimeError("A window without any valid daylight target survived the schedule filter.")
    if (w["sd_mask"] & ~w["sd_daylight"]).any() or (w["nd_mask"] & ~w["nd_daylight"]).any():
        raise RuntimeError("Mask includes night targets.")


def concatenate_windows(parts):
    parts = [p for p in parts if len(p["issue_ts"])]
    if not parts:
        raise RuntimeError("No forecast windows were built.")
    out = {"inputs": {k: np.concatenate([p["inputs"][k] for p in parts], axis=0) for k in parts[0]["inputs"]}}
    for key, value in parts[0].items():
        if key in ("inputs", "lookback"):
            continue
        out[key] = np.concatenate([p[key] for p in parts], axis=0)
    out["lookback"] = parts[0]["lookback"]
    return out


def build_split_windows(houses, house_ids, lookback):
    """Windows and schedule audit for a set of houses (eligibility independent of LOOKBACK)."""
    parts, audits, rejections = [], [], []
    for h in house_ids:
        acc, rej = build_joint_issue_schedule(houses[h]["timestamps"], houses[h]["daylight"], COMMON_MAX_LOOKBACK_STEPS)
        parts.append(assemble_windows(houses[h], acc["issue_idx"].to_numpy(np.int64),
                                      acc["Is_First_Issue_S"].to_numpy(bool), lookback))
        audit = acc.drop(columns=["issue_idx"]).copy()
        audit.insert(0, "House", h)
        audits.append(audit)
        if len(rej):
            r = rej.copy()
            r.insert(0, "House", h)
            rejections.append(r)
    windows = concatenate_windows(parts)
    audit_df = pd.concat(audits, ignore_index=True)
    rej_df = pd.concat(rejections, ignore_index=True) if rejections else pd.DataFrame(columns=["House", "Candidate_Timestamp", "Reason"])
    return windows, audit_df, rej_df


def subset_windows(w, idx):
    idx = np.asarray(idx)
    out = {"inputs": {k: v[idx] for k, v in w["inputs"].items()}, "lookback": w["lookback"]}
    for key, value in w.items():
        if key in ("inputs", "lookback"):
            continue
        out[key] = value[idx]
    return out


def training_target_variances(train_w):
    """Population variances (kW^2) of the valid TRAINING targets (Houses 01-09 only)."""
    sd = train_w["sd_y_kw"][train_w["sd_mask"]].astype(np.float64)
    nd = train_w["nd_y_kw"][train_w["nd_mask"]].astype(np.float64)
    var_sd, var_nd = float(np.var(sd)), float(np.var(nd))
    if not (var_sd > 0 and var_nd > 0):
        raise RuntimeError("Training target variances must be positive.")
    return var_sd, var_nd, int(sd.size), int(nd.size)


# =============================================================================
# SECTION 9 - GLOBAL MASKED MSE, POST-PROCESSING AND METRICS (NumPy)
# =============================================================================
def numpy_global_masked_mse(y_true_kw, y_pred_kw, mask):
    """GlobalMaskedMSE = SUM(mask * error^2) / SUM(mask) over ALL valid target points."""
    m = np.asarray(mask, dtype=np.float64)
    err = np.asarray(y_pred_kw, dtype=np.float64) - np.asarray(y_true_kw, dtype=np.float64)
    count = m.sum()
    if count <= 0:
        raise ValueError("No valid target points.")
    return float((m * err * err).sum() / count)


def combined_monitor(mse_sd, mse_nd, var_sd, var_nd):
    """0.5*MSE_SD/Var_SD + 0.5*MSE_ND/Var_ND  (lower is better)."""
    return 0.5 * mse_sd / var_sd + 0.5 * mse_nd / var_nd


def postprocess_prediction_w(raw_kw, daylight):
    """Operational output in W: negative -> 0 W, night (zenith >= pi/2) -> exactly 0 W."""
    pred_w = np.asarray(raw_kw, dtype=np.float64) * W_PER_KW
    pred_w = np.maximum(pred_w, 0.0)
    pred_w[~np.asarray(daylight, dtype=bool)] = 0.0
    return pred_w


def point_metrics(actual_w, pred_w, rated_w, mask):
    """Point metrics on valid daylight real targets only.

    RMSE_W, MAE_W
    nRMSE_pct     = 100 * sqrt(mean((error / Array_Rated_Power_W of the house)^2))
    WAPE_pct      = 100 * SUM|error| / SUM|actual|
    R2            = 1 - SSE / SST
    sMAPE_pct     = 100 * mean(2|error| / (|actual| + |pred|)), points with both = 0 skipped
    MAPE_1pct_pct = 100 * mean(|error| / actual) over actual >= 1 % of the house's rated power
    """
    mask = np.asarray(mask, dtype=bool)
    a = np.asarray(actual_w, dtype=np.float64)[mask]
    p = np.asarray(pred_w, dtype=np.float64)[mask]
    r = np.broadcast_to(np.asarray(rated_w, dtype=np.float64), mask.shape)[mask]
    n = int(a.size)
    out = {k: np.nan for k in POINT_METRIC_KEYS}
    out.update({"N_points": n, "N_MAPE_1pct_included": 0, "N_MAPE_1pct_excluded": 0, "N_sMAPE_excluded_both_zero": 0})
    if n == 0:
        return out
    e = p - a
    out["RMSE_W"] = float(np.sqrt(np.mean(e * e)))
    out["MAE_W"] = float(np.mean(np.abs(e)))
    out["nRMSE_pct"] = float(100.0 * np.sqrt(np.mean((e / r) ** 2)))
    sum_abs_actual = float(np.sum(np.abs(a)))
    out["WAPE_pct"] = float(100.0 * np.sum(np.abs(e)) / sum_abs_actual) if sum_abs_actual > 0 else np.nan
    sst = float(np.sum((a - a.mean()) ** 2))
    out["R2"] = float(1.0 - np.sum(e * e) / sst) if (n > 1 and sst > 0) else np.nan
    den = np.abs(a) + np.abs(p)
    ok = den > 0
    out["sMAPE_pct"] = float(100.0 * np.mean(2.0 * np.abs(e[ok]) / den[ok])) if ok.any() else np.nan
    out["N_sMAPE_excluded_both_zero"] = int((~ok).sum())
    inc = a >= MAPE_MIN_FRACTION_OF_RATED * r
    out["N_MAPE_1pct_included"] = int(inc.sum())
    out["N_MAPE_1pct_excluded"] = int(n - inc.sum())
    out["MAPE_1pct_pct"] = float(100.0 * np.mean(np.abs(e[inc]) / a[inc])) if inc.any() else np.nan
    return out


def window_hems_table(actual_w, pred_w, mask):
    """Per-window HEMS quantities on valid daylight real targets only (no night, no padding)."""
    mask = np.asarray(mask, dtype=bool)
    a = np.where(mask, actual_w, 0.0)
    p = np.where(mask, pred_w, 0.0)
    e_act = a.sum(axis=1) * STEP_HOURS / W_PER_KW
    e_pred = p.sum(axis=1) * STEP_HOURS / W_PER_KW
    a_peak_idx = np.argmax(np.where(mask, actual_w, -np.inf), axis=1)
    p_peak_idx = np.argmax(np.where(mask, pred_w, -np.inf), axis=1)
    rows = np.arange(len(a))
    a_peak = np.asarray(actual_w, dtype=np.float64)[rows, a_peak_idx]
    p_peak = np.asarray(pred_w, dtype=np.float64)[rows, p_peak_idx]
    return pd.DataFrame({
        "Actual_Energy_kWh": e_act,
        "Predicted_Energy_kWh": e_pred,
        "Energy_Error_kWh": e_pred - e_act,
        "Actual_Peak_W": a_peak,
        "Predicted_Peak_W": p_peak,
        "Peak_Power_Error_W": p_peak - a_peak,
        "Actual_Peak_Slot": a_peak_idx + 1,
        "Predicted_Peak_Slot": p_peak_idx + 1,
        "Peak_Time_Error_min": (p_peak_idx - a_peak_idx).astype(np.int64) * STEP_MINUTES,
        "N_Valid_Targets": mask.sum(axis=1).astype(int),
    })


def hems_summary(window_df):
    if len(window_df) == 0:
        out = {k: np.nan for k in HEMS_METRIC_KEYS}
        out["N_windows"] = 0
        return out
    ee = window_df["Energy_Error_kWh"].to_numpy(np.float64)
    ea = window_df["Actual_Energy_kWh"].to_numpy(np.float64)
    pe = window_df["Peak_Power_Error_W"].to_numpy(np.float64)
    te = window_df["Peak_Time_Error_min"].to_numpy(np.float64)
    denom = float(np.sum(np.abs(ea)))
    return {
        "Energy_MAE_kWh": float(np.mean(np.abs(ee))),
        "Energy_RMSE_kWh": float(np.sqrt(np.mean(ee * ee))),
        "Energy_Bias_kWh": float(np.mean(ee)),
        "Energy_WAPE_pct": float(100.0 * np.sum(np.abs(ee)) / denom) if denom > 0 else np.nan,
        "Peak_Power_MAE_W": float(np.mean(np.abs(pe))),
        "Peak_Power_Bias_W": float(np.mean(pe)),
        "Peak_Time_MAE_min": float(np.mean(np.abs(te))),
        "Peak_Time_Bias_min": float(np.mean(te)),
        "N_windows": int(len(window_df)),
    }


def macro_average_row(rows, metric_keys, count_keys):
    out = {k: float(np.nanmean([r[k] for r in rows])) if rows else np.nan for k in metric_keys}
    for k in count_keys:
        out[k] = int(np.sum([r[k] for r in rows])) if rows else 0
    return out


# =============================================================================
# SECTION 10 - EVALUATION OF ONE SPLIT (Validation or Test evaluation)
# =============================================================================
def evaluate_split(model, windows, split):
    """Predict with the (reloaded) model and compute every metric table for both tasks."""
    sd_raw_kw, nd_raw_kw = engine_predict_raw_kw(model, windows)
    raw = {"same_day": sd_raw_kw, "next_day": nd_raw_kw}
    houses_in_split = SPLIT_HOUSES[split]
    if sorted(np.unique(windows["house"]).tolist()) != sorted(houses_in_split):
        raise RuntimeError(f"{split}: windows contain houses {np.unique(windows['house'])}, expected {houses_in_split}.")
    point_counts = ["N_points", "N_MAPE_1pct_included", "N_MAPE_1pct_excluded", "N_sMAPE_excluded_both_zero"]
    results = {"split": split, "raw_kw": raw}

    mse = {}
    for task in TASKS:
        pre = TASK_PREFIX[task]
        mse[task] = numpy_global_masked_mse(windows[f"{pre}_y_kw"], raw[task], windows[f"{pre}_mask"])
    results["global_masked_mse_kw2"] = mse

    for task in TASKS:
        pre = TASK_PREFIX[task]
        mask = windows[f"{pre}_mask"]
        actual_w = windows[f"{pre}_actual_w"]
        pred_w = postprocess_prediction_w(raw[task], windows[f"{pre}_daylight"])
        rated = np.repeat(windows["rated_w"][:, None], TASK_SLOTS[task], axis=1)
        house = windows["house"]

        point_rows, per_house_rows = [], []
        for h in houses_in_split:
            sel = house == h
            row = point_metrics(actual_w[sel], pred_w[sel], rated[sel], mask[sel])
            per_house_rows.append(row)
            point_rows.append({"Scope": f"House {h:02d}", **row})
        point_rows.append({"Scope": "POOLED", **point_metrics(actual_w, pred_w, rated, mask)})
        point_rows.append({"Scope": "MACRO_AVERAGE", **macro_average_row(per_house_rows, POINT_METRIC_KEYS, point_counts)})
        point_df = pd.DataFrame(point_rows)
        point_df.insert(0, "Task", task)
        point_df.insert(0, "Split", split)

        leads = lead_minutes(task)
        hz_rows = []
        for j in range(TASK_SLOTS[task]):
            per_house_j = []
            for h in houses_in_split:
                sel = house == h
                row = point_metrics(actual_w[sel, j:j + 1], pred_w[sel, j:j + 1], rated[sel, j:j + 1], mask[sel, j:j + 1])
                per_house_j.append(row)
                hz_rows.append({"Scope": f"House {h:02d}", "Slot": j + 1, "Lead_Minutes": int(leads[j]), **row})
            hz_rows.append({"Scope": "POOLED", "Slot": j + 1, "Lead_Minutes": int(leads[j]),
                            **point_metrics(actual_w[:, j:j + 1], pred_w[:, j:j + 1], rated[:, j:j + 1], mask[:, j:j + 1])})
            hz_rows.append({"Scope": "MACRO_AVERAGE", "Slot": j + 1, "Lead_Minutes": int(leads[j]),
                            **macro_average_row(per_house_j, POINT_METRIC_KEYS, point_counts)})
        horizon_df = pd.DataFrame(hz_rows)
        horizon_df.insert(0, "Task", task)
        horizon_df.insert(0, "Split", split)

        win_df = window_hems_table(actual_w, pred_w, mask)
        win_df.insert(0, "Is_First_Issue_S", windows["is_first_issue"])
        win_df.insert(0, "Issue_Timestamp", pd.to_datetime(windows["issue_ts"]))
        win_df.insert(0, "House", house)
        hems_rows, hems_house = [], []
        for h in houses_in_split:
            s = hems_summary(win_df[win_df["House"] == h])
            hems_house.append(s)
            hems_rows.append({"Scope": f"House {h:02d}", **s})
        hems_rows.append({"Scope": "POOLED", **hems_summary(win_df)})
        hems_rows.append({"Scope": "MACRO_AVERAGE", **macro_average_row(hems_house, HEMS_METRIC_KEYS, ["N_windows"])})
        hems_df = pd.DataFrame(hems_rows)
        hems_df.insert(0, "Task", task)
        hems_df.insert(0, "Split", split)

        results[task] = {
            "raw_kw_task": raw[task],
            "pred_w": pred_w,
            "actual_w": actual_w,
            "mask": mask,
            "real": windows[f"{pre}_real"],
            "daylight": windows[f"{pre}_daylight"],
            "rated": rated,
            "house": house,
            "point": point_df,
            "horizon": horizon_df,
            "hems": hems_df,
            "windows": win_df,
        }
    results["monitor_value"] = None
    return results


def forecast_long_table(windows, res, task):
    """Public forecasts in long format (the Next-Day padding slot is NEVER reported)."""
    pre = TASK_PREFIX[task]
    n_win, n_slots = res["pred_w"].shape
    real = res["real"]
    rows_w, rows_s = np.nonzero(real)
    leads = lead_minutes(task)
    df = pd.DataFrame({
        "House": windows["house"][rows_w],
        "Issue_Timestamp": pd.to_datetime(windows["issue_ts"][rows_w]),
        "Is_First_Issue_S": windows["is_first_issue"][rows_w],
        "Slot": rows_s + 1,
        "Lead_Minutes": leads[rows_s],
        "Target_Timestamp": pd.to_datetime(windows[f"{pre}_target_ts"][rows_w, rows_s]),
        "Daylight": res["daylight"][rows_w, rows_s],
        "Valid_Target": res["mask"][rows_w, rows_s],
        "Actual_W": res["actual_w"][rows_w, rows_s],
        "Predicted_W": res["pred_w"][rows_w, rows_s],
        "Raw_Model_Output_W": res["raw_kw_task"][rows_w, rows_s] * W_PER_KW,
    })
    return df


def save_split_outputs(windows, results, split, paths):
    metric_dir = paths["metrics_" + split]
    for task in TASKS:
        res = results[task]
        res["point"].to_csv(metric_dir / f"{task}_point_metrics.csv", index=False)
        res["horizon"].to_csv(metric_dir / f"{task}_metrics_by_horizon.csv", index=False)
        res["hems"].to_csv(metric_dir / f"{task}_hems_metrics.csv", index=False)
        res["windows"].to_csv(metric_dir / f"{task}_window_hems_metrics.csv.gz", index=False)
        fc = forecast_long_table(windows, res, task)
        fc_path = paths["forecasts"] / f"{split}_{task}_forecasts.csv.gz"
        fc.to_csv(fc_path, index=False)
        res["forecast_path"] = fc_path
    summary = {
        "split": split,
        "label": SPLIT_LABELS[split],
        "houses": SPLIT_HOUSES[split],
        "n_windows": int(len(windows["issue_ts"])),
        "global_masked_mse_kw2": results["global_masked_mse_kw2"],
        "combined_monitor": results.get("monitor_value"),
        "point_metrics": {t: results[t]["point"].to_dict(orient="records") for t in TASKS},
        "hems_metrics": {t: results[t]["hems"].to_dict(orient="records") for t in TASKS},
    }
    write_json(metric_dir / "summary.json", summary)


def verify_metrics_from_saved_forecasts(results, split):
    """Independent re-computation of pooled metrics from the saved forecast CSV files."""
    report = {}
    for task in TASKS:
        res = results[task]
        df = pd.read_csv(res["forecast_path"])
        v = df[df["Valid_Target"].astype(bool)]
        e = v["Predicted_W"].to_numpy(np.float64) - v["Actual_W"].to_numpy(np.float64)
        independent = {
            "RMSE_W": float(np.sqrt(np.mean(e ** 2))),
            "MAE_W": float(np.mean(np.abs(e))),
            "WAPE_pct": float(100 * np.abs(e).sum() / np.abs(v["Actual_W"].to_numpy(np.float64)).sum()),
            "N_points": int(len(v)),
        }
        pooled = res["point"][res["point"]["Scope"] == "POOLED"].iloc[0]
        diffs = {k: abs(independent[k] - float(pooled[k])) for k in independent}
        ok = all(np.isclose(independent[k], float(pooled[k]), rtol=1e-9, atol=1e-9) for k in independent)
        night = df[~df["Daylight"].astype(bool)]
        night_zero = bool((night["Predicted_W"] == 0).all())
        non_negative = bool((df["Predicted_W"] >= 0).all())
        no_padding = bool(len(df) == int(res["real"].sum()))
        report[task] = {
            "independent_from_csv": independent,
            "pipeline_pooled": {k: float(pooled[k]) for k in independent},
            "abs_differences": diffs,
            "metrics_match": bool(ok),
            "night_predictions_exactly_zero": night_zero,
            "predictions_non_negative": non_negative,
            "padding_slot_not_reported": no_padding,
        }
        if not (ok and night_zero and non_negative and no_padding):
            raise RuntimeError(f"{split}/{task}: independent metric verification failed: {report[task]}")
    return report


def print_split_summary(results, split):
    print(f"\n{SPLIT_LABELS[split]} - {ARCH_LABEL}")
    cols = ["Scope", "RMSE_W", "MAE_W", "nRMSE_pct", "WAPE_pct", "R2", "sMAPE_pct", "MAPE_1pct_pct", "N_points"]
    hcols = ["Scope", "Energy_MAE_kWh", "Energy_Bias_kWh", "Energy_WAPE_pct", "Peak_Power_MAE_W", "Peak_Power_Bias_W",
             "Peak_Time_MAE_min", "Peak_Time_Bias_min", "N_windows"]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        for task in TASKS:
            print(f"\n  {TASK_LABELS[task]} - point metrics")
            print(results[task]["point"][cols].to_string(index=False))
            print(f"  {TASK_LABELS[task]} - HEMS metrics (valid daylight real targets only)")
            print(results[task]["hems"][hcols].to_string(index=False))


# =============================================================================
# SECTION 11 - DEPLOYMENT PREDICTION (reconstructs and validates the schedule)
# =============================================================================
def predict_issue(model, issue_timestamp, history_frame, weather_frame, static_values, weather_scaler, static_scaler):
    """Operational prediction for ONE issue time.

    history_frame : Timestamp, PV_DC_Power_W, Solar_Zenith_rad - must contain the
                    LOOKBACK rows t-(LOOKBACK-1)*15 min ... t (measured PV).
    weather_frame : Timestamp + the 8 weather/solar features on the 15-min grid,
                    from 00:00 of the issue day through t+30 h (realized/forecast weather).
    static_values : dict with the 4 static PV-system features.

    Returns the 24 Same-Day values and the 24 (first issue S) or 25 real Next-Day
    values in W. The Next-Day padding slot is never returned.
    """
    issue_ts = pd.Timestamp(issue_timestamp)
    missing = [c for c in [TIMESTAMP_COL] + WEATHER_FEATURES if c not in weather_frame.columns]
    if missing:
        raise ValueError(f"weather_frame is missing columns {missing}")
    wf = weather_frame.copy()
    wts, _ = validate_solar_frame(wf, "weather_frame")
    wvals = wf[WEATHER_FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(np.float64)
    if not np.isfinite(wvals).all():
        raise ValueError("weather_frame contains NaN/Inf.")
    info = validate_deployment_issue_time(issue_ts, wf[[TIMESTAMP_COL, ZENITH_COL]])
    n_nd_real = info["n_next_day_real_slots"]

    missing_static = [c for c in STATIC_FEATURES if c not in static_values]
    if missing_static:
        raise ValueError(f"static_values is missing {missing_static}")
    static_raw = np.asarray([float(static_values[c]) for c in STATIC_FEATURES], dtype=np.float64)
    ok, expected, diff = check_rated_power_consistency(static_values[RATED_POWER_COL], static_values[PANELS_COL], static_values[PMAX_COL])
    if not ok:
        raise ValueError(f"Array_Rated_Power_W differs from Number_of_Panels*Pmax_Per_Panel_W by {diff} W.")

    hf = history_frame.copy()
    for c in (TIMESTAMP_COL, TARGET_COL, ZENITH_COL):
        if c not in hf.columns:
            raise ValueError(f"history_frame is missing {c}")
    hts = ns_index(pd.to_datetime(hf[TIMESTAMP_COL], errors="raise"))
    wanted = ns_index(pd.date_range(issue_ts - (LOOKBACK - 1) * STEP, issue_ts, freq=STEP))
    pos = hts.get_indexer(wanted)
    if (pos < 0).any():
        raise ValueError(f"history_frame must contain every timestamp {wanted[0]} ... {wanted[-1]}.")
    hist_pv_w = pd.to_numeric(hf[TARGET_COL], errors="raise").to_numpy(np.float64)[pos]
    hist_zen = pd.to_numeric(hf[ZENITH_COL], errors="raise").to_numpy(np.float64)[pos]
    if not (np.isfinite(hist_pv_w).all() and np.isfinite(hist_zen).all()):
        raise ValueError("history_frame contains NaN/Inf.")
    if (hist_pv_w < 0).any():
        raise ValueError("history_frame contains negative PV.")
    hist_dl = is_daylight(hist_zen)
    if ((~hist_dl) & (hist_pv_w != 0)).any():
        raise ValueError("history_frame has non-zero PV at night (Solar_Zenith_rad >= pi/2).")

    sd_ts = ns_index(pd.date_range(issue_ts + STEP, issue_ts + SD_SLOTS * STEP, freq=STEP))
    nd_ts = ns_index(pd.date_range(issue_ts + pd.Timedelta(hours=24), periods=n_nd_real, freq=STEP))
    sd_pos, nd_pos = wts.get_indexer(sd_ts), wts.get_indexer(nd_ts)
    if (sd_pos < 0).any() or (nd_pos < 0).any():
        raise ValueError("weather_frame does not cover all Same-Day / Next-Day target times.")

    nd_weather = np.zeros((1, ND_SLOTS, len(WEATHER_FEATURES)), np.float32)
    nd_weather[0, :n_nd_real] = weather_scaler.transform(wvals[nd_pos]).astype(np.float32)
    inputs = {
        "sd_history": np.stack([(hist_pv_w / W_PER_KW).astype(np.float32), hist_dl.astype(np.float32)], axis=-1)[None],
        "sd_weather": weather_scaler.transform(wvals[sd_pos]).astype(np.float32)[None],
        "nd_weather": nd_weather,
        "static": static_scaler.transform(static_raw[None, :]).astype(np.float32),
    }
    sd_raw, nd_raw = engine_predict_raw_kw(model, {"inputs": inputs})
    zen_col = WEATHER_FEATURES.index(ZENITH_COL)
    sd_dl = is_daylight(wvals[sd_pos, zen_col])
    nd_dl = is_daylight(wvals[nd_pos, zen_col])
    sd_w = postprocess_prediction_w(sd_raw[0], sd_dl)
    nd_w = postprocess_prediction_w(nd_raw[0, :n_nd_real], nd_dl)
    return {
        "issue_timestamp": issue_ts,
        "is_first_issue_S": info["is_first_issue_S"],
        "same_day": pd.DataFrame({"Target_Timestamp": sd_ts, "Lead_Minutes": lead_minutes("same_day"),
                                  "Daylight": sd_dl, "Predicted_W": sd_w}),
        "next_day": pd.DataFrame({"Target_Timestamp": nd_ts, "Lead_Minutes": lead_minutes("next_day")[:n_nd_real],
                                  "Daylight": nd_dl, "Predicted_W": nd_w}),
    }


def house_frames_for_issue(house, issue_idx):
    """Raw history / weather frames exactly as a deployment caller would supply them."""
    ts = house["timestamps"]
    day_start = int(ts.get_loc(ts[issue_idx].normalize()))
    stop = min(len(ts), issue_idx + ND_LEAD_STEPS + ND_SLOTS + 1)
    weather = pd.DataFrame(house["weather_raw"][day_start:stop], columns=WEATHER_FEATURES)
    weather.insert(0, TIMESTAMP_COL, ts[day_start:stop])
    h0 = issue_idx - (LOOKBACK - 1)
    history = pd.DataFrame({
        TIMESTAMP_COL: ts[h0:issue_idx + 1],
        TARGET_COL: house["pv_w"][h0:issue_idx + 1],
        ZENITH_COL: house["zenith"][h0:issue_idx + 1],
    })
    static_values = dict(zip(STATIC_FEATURES, house["static_raw"].tolist()))
    return history, weather, static_values


def verify_deployment_schedule(houses, house_ids):
    """The deployment schedule reconstruction must reproduce the training schedule."""
    n_days, n_rejected_checks = 0, 0
    for h in house_ids:
        d = houses[h]
        ts = d["timestamps"]
        acc, _ = build_joint_issue_schedule(ts, d["daylight"], COMMON_MAX_LOOKBACK_STEPS)
        expected_by_day = {}
        for t in acc["Issue_Timestamp"]:
            expected_by_day.setdefault(pd.Timestamp(t).normalize(), set()).add(pd.Timestamp(t))
        days = pd.DatetimeIndex(ts.normalize().unique())
        for day in days[2:-3]:
            start = int(ts.get_loc(day))
            stop = min(len(ts), start + 3 * STEPS_PER_DAY)
            frame = pd.DataFrame({TIMESTAMP_COL: ts[start:stop], ZENITH_COL: d["zenith"][start:stop]})
            dep, _ = build_joint_issue_schedule(frame[TIMESTAMP_COL], is_daylight(frame[ZENITH_COL]), 1)
            dep_today = {pd.Timestamp(t) for t in dep["Issue_Timestamp"] if pd.Timestamp(t).normalize() == day}
            if dep_today != expected_by_day.get(day, set()):
                raise RuntimeError(f"House {h:02d} {day.date()}: deployment schedule differs from training schedule.")
            for valid in sorted(dep_today)[:1]:
                validate_deployment_issue_time(valid, frame)
                for bad in (valid + STEP, valid - STEP):
                    if bad in dep_today:
                        continue
                    try:
                        validate_deployment_issue_time(bad, frame)
                    except ValueError:
                        n_rejected_checks += 1
                    else:
                        raise RuntimeError(f"House {h:02d}: invalid issue time {bad} was accepted.")
            n_days += 1
    return {"houses": list(house_ids), "days_checked": n_days, "invalid_issue_times_rejected": n_rejected_checks, "passed": True}


def verify_deployment_predictions(model, houses, windows, results, weather_scaler, static_scaler, n_samples=40):
    """predict_issue() on raw frames must reproduce the evaluation-pipeline forecasts."""
    n = len(windows["issue_ts"])
    idx = set(np.unique(np.linspace(0, n - 1, n_samples).astype(int)).tolist())
    idx |= set(np.flatnonzero(windows["is_first_issue"])[:5].tolist())
    max_sd, max_nd = 0.0, 0.0
    for i in sorted(idx):
        house = houses[int(windows["house"][i])]
        hist, weather, static_values = house_frames_for_issue(house, int(windows["issue_idx"][i]))
        out = predict_issue(model, windows["issue_ts"][i], hist, weather, static_values, weather_scaler, static_scaler)
        sd_ref = results["same_day"]["pred_w"][i]
        nd_ref = results["next_day"]["pred_w"][i][windows["nd_real"][i]]
        if len(out["next_day"]) != int(windows["nd_real"][i].sum()):
            raise RuntimeError("Deployment Next-Day output length differs from the number of real slots.")
        if bool(out["is_first_issue_S"]) != bool(windows["is_first_issue"][i]):
            raise RuntimeError("Deployment first-issue flag differs from the training schedule.")
        max_sd = max(max_sd, float(np.max(np.abs(out["same_day"]["Predicted_W"].to_numpy() - sd_ref))))
        max_nd = max(max_nd, float(np.max(np.abs(out["next_day"]["Predicted_W"].to_numpy() - nd_ref))))
    tol_w = 0.05
    report = {"n_windows_checked": len(idx), "max_abs_diff_same_day_W": max_sd, "max_abs_diff_next_day_W": max_nd,
              "tolerance_W": tol_w, "passed": bool(max_sd <= tol_w and max_nd <= tol_w)}
    if not report["passed"]:
        raise RuntimeError(f"Deployment helper does not reproduce the evaluation pipeline: {report}")
    return report


# =============================================================================
# SECTION 12 - PLOTTING (every figure is saved; display is optional)
# =============================================================================
HOUSE_COLORS = {10: "tab:blue", 11: "tab:orange", 12: "tab:green", 13: "tab:red"}


def house_color(h):
    return HOUSE_COLORS.get(int(h), plt.cm.tab20(int(h) % 20))


def show_saved_png(path):
    if not SHOW_PLOTS_IN_COLAB:
        return
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:
        pass


def finalize_figure(fig, path, title):
    """Wrapped title + footer (architecture, run ID, manual hyperparameters), save, optional display."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.suptitle(textwrap.fill(title, width=TITLE_WRAP_WIDTH), y=1.0, va="bottom", fontsize=11)
    footer = f"Architecture: {ARCH_LABEL} | Run ID: {RUN_STATE['run_id']} | {hyperparameter_text()}"
    fig.text(0.5, -0.01, textwrap.fill(footer, width=FOOTER_WRAP_WIDTH), ha="center", va="top", fontsize=7, color="0.25")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    show_saved_png(path)
    return path


def _split_task_title(split, task, what):
    return f"{ARCH_LABEL} - {SPLIT_LABELS[split]} - {TASK_LABELS[task]} - {what}"


def plot_metric_by_horizon(res, split, task, metric, out_dir):
    hz = res["horizon"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_scale = 60.0
    for h in SPLIT_HOUSES[split]:
        d = hz[hz["Scope"] == f"House {h:02d}"]
        ax.plot(d["Lead_Minutes"] / x_scale, d[metric], marker="o", ms=3, lw=1.2, color=house_color(h), label=f"House {h:02d}")
    d = hz[hz["Scope"] == "POOLED"]
    ax.plot(d["Lead_Minutes"] / x_scale, d[metric], marker="s", ms=3, lw=2.2, color="black", label="POOLED")
    d = hz[hz["Scope"] == "MACRO_AVERAGE"]
    ax.plot(d["Lead_Minutes"] / x_scale, d[metric], ls="--", lw=1.4, color="0.4", label="MACRO_AVERAGE")
    ax.set_xlabel("Lead time after issue t (hours)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return finalize_figure(fig, out_dir / f"{task}_{metric.lower()}_by_horizon.png",
                           _split_task_title(split, task, f"{metric} by forecast horizon (valid daylight targets)"))


def plot_scatter(res, split, task, out_dir):
    houses = SPLIT_HOUSES[split]
    ncols = len(houses) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.4 * ncols, 4.6))
    mask = res["mask"]
    vmax = max(float(res["actual_w"][mask].max()), float(res["pred_w"][mask].max()), 1.0) * 1.03
    for ax, h in zip(axes, houses + ["POOLED"]):
        sel = mask & (res["house"][:, None] == h) if h != "POOLED" else mask
        if h == "POOLED":
            for hh in houses:
                s2 = mask & (res["house"][:, None] == hh)
                ax.scatter(res["actual_w"][s2], res["pred_w"][s2], s=1.5, alpha=0.15, color=house_color(hh), rasterized=True, label=f"House {hh:02d}")
            ax.legend(fontsize=7, markerscale=6)
        else:
            ax.scatter(res["actual_w"][sel], res["pred_w"][sel], s=1.5, alpha=0.15, color=house_color(h), rasterized=True)
        ax.plot([0, vmax], [0, vmax], color="black", lw=1)
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual PV (W)")
        ax.set_ylabel("Predicted PV (W)")
        ax.set_title(f"House {h:02d}" if h != "POOLED" else "POOLED", fontsize=10)
        ax.grid(alpha=0.3)
    return finalize_figure(fig, out_dir / f"{task}_actual_vs_predicted_scatter.png",
                           _split_task_title(split, task, "actual vs predicted PV (valid daylight targets, diagonal = perfect)"))


def _residuals_by_house(res, split):
    out = {}
    for h in SPLIT_HOUSES[split]:
        sel = res["mask"] & (res["house"][:, None] == h)
        out[h] = (res["pred_w"][sel] - res["actual_w"][sel], res["pred_w"][sel], res["actual_w"][sel], res["rated"][sel])
    return out


def plot_residual_histogram(res, split, task, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    resid = _residuals_by_house(res, split)
    allr = np.concatenate([v[0] for v in resid.values()])
    lo, hi = np.percentile(allr, [0.1, 99.9])
    bins = np.linspace(lo, hi, 120)
    for h, (r, _, _, _) in resid.items():
        ax.hist(r, bins=bins, histtype="step", lw=1.4, color=house_color(h), density=True, label=f"House {h:02d} (mean {r.mean():.1f} W)")
    ax.hist(allr, bins=bins, histtype="step", lw=2.0, color="black", density=True, label=f"POOLED (mean {allr.mean():.1f} W)")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel("Residual = predicted - actual (W)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return finalize_figure(fig, out_dir / f"{task}_residual_histogram.png",
                           _split_task_title(split, task, "residual histogram (central 99.8 % range)"))


def plot_residual_vs_predicted(res, split, task, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for h, (r, p, _, _) in _residuals_by_house(res, split).items():
        ax.scatter(p, r, s=1.5, alpha=0.15, color=house_color(h), rasterized=True, label=f"House {h:02d}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Predicted PV (W)")
    ax.set_ylabel("Residual = predicted - actual (W)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=6)
    return finalize_figure(fig, out_dir / f"{task}_residual_vs_predicted.png",
                           _split_task_title(split, task, "residual vs predicted PV"))


def plot_abs_error_histogram(res, split, task, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    resid = _residuals_by_house(res, split)
    alla = np.abs(np.concatenate([v[0] for v in resid.values()]))
    bins = np.linspace(0, np.percentile(alla, 99.9) + 1e-9, 120)
    for h, (r, _, _, _) in resid.items():
        ax.hist(np.abs(r), bins=bins, histtype="step", lw=1.4, color=house_color(h), density=True, label=f"House {h:02d}")
    ax.hist(alla, bins=bins, histtype="step", lw=2.0, color="black", density=True, label="POOLED")
    ax.set_xlabel("|predicted - actual| (W)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return finalize_figure(fig, out_dir / f"{task}_absolute_error_histogram.png",
                           _split_task_title(split, task, "absolute error histogram (up to 99.9th percentile)"))


def plot_abs_error_cdf(res, split, task, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    resid = _residuals_by_house(res, split)
    for h, (r, _, _, _) in list(resid.items()) + [("POOLED", (np.concatenate([v[0] for v in resid.values()]), None, None, None))]:
        a = np.sort(np.abs(r))
        y = np.arange(1, len(a) + 1) / len(a)
        ax.plot(a, y, lw=2.0 if h == "POOLED" else 1.4, color="black" if h == "POOLED" else house_color(h),
                label="POOLED" if h == "POOLED" else f"House {h:02d}")
    ax.set_xlabel("|predicted - actual| (W)")
    ax.set_ylabel("Cumulative fraction of valid targets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return finalize_figure(fig, out_dir / f"{task}_absolute_error_cdf.png",
                           _split_task_title(split, task, "absolute error CDF"))


def plot_metrics_by_house(res, split, task, out_dir):
    pt = res["point"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    scopes = pt["Scope"].tolist()
    colors = [house_color(int(s.split()[1])) if s.startswith("House") else ("black" if s == "POOLED" else "0.55") for s in scopes]
    for ax, metric in zip(axes.ravel(), POINT_METRIC_KEYS):
        vals = pt[metric].to_numpy(np.float64)
        ax.bar(range(len(scopes)), vals, color=colors)
        for k, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(k, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(scopes)))
        ax.set_xticklabels([s.replace("MACRO_AVERAGE", "MACRO") for s in scopes], fontsize=7, rotation=20)
        ax.set_title(metric, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    hm = res["hems"]
    ax = axes.ravel()[-1]
    vals = hm["Energy_WAPE_pct"].to_numpy(np.float64)
    ax.bar(range(len(scopes)), vals, color=colors)
    ax.set_xticks(range(len(scopes)))
    ax.set_xticklabels([s.replace("MACRO_AVERAGE", "MACRO") for s in scopes], fontsize=7, rotation=20)
    ax.set_title("Energy_WAPE_pct (HEMS)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    return finalize_figure(fig, out_dir / f"{task}_metrics_by_house.png",
                           _split_task_title(split, task, "metrics by house, POOLED and MACRO_AVERAGE"))


def plot_example_windows(res, windows, split, task, out_dir):
    pre = TASK_PREFIX[task]
    wdf = res["windows"]
    picks = []
    for h in SPLIT_HOUSES[split]:
        idx = np.flatnonzero(wdf["House"].to_numpy() == h)
        if len(idx) == 0:
            continue
        energy = wdf["Actual_Energy_kWh"].to_numpy()[idx]
        order = idx[np.argsort(energy, kind="stable")]
        picks.append((h, int(order[-1]), "highest actual energy"))
        picks.append((h, int(order[len(order) // 2]), "median actual energy"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (h, i, why) in zip(axes.ravel(), picks):
        real = res["real"][i]
        t = pd.to_datetime(windows[f"{pre}_target_ts"][i][real])
        ax.plot(t, res["actual_w"][i][real], color="black", marker="o", ms=3, label="Actual")
        ax.plot(t, res["pred_w"][i][real], color=house_color(h), marker="s", ms=3, label="Predicted")
        night = ~res["daylight"][i][real]
        if night.any():
            ax.scatter(t[night], np.zeros(night.sum()), marker="x", color="0.5", label="night (0 W by rule)")
        issue = pd.Timestamp(windows["issue_ts"][i])
        first = " (first issue S)" if windows["is_first_issue"][i] else ""
        ax.set_title(f"House {h:02d} - issue {issue:%Y-%m-%d %H:%M}{first}\n{why}", fontsize=9)
        ax.set_ylabel("PV (W)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    return finalize_figure(fig, out_dir / f"{task}_example_forecast_windows.png",
                           _split_task_title(split, task, "example forecast windows (deterministic selection)"))


def _scatter_windows(res, split, xcol, ycol, xlabel, ylabel, fname, title, task, out_dir):
    wdf = res["windows"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = max(float(wdf[xcol].max()), float(wdf[ycol].max()), 1e-9) * 1.03
    for h in SPLIT_HOUSES[split]:
        d = wdf[wdf["House"] == h]
        ax.scatter(d[xcol], d[ycol], s=4, alpha=0.3, color=house_color(h), rasterized=True, label=f"House {h:02d}")
    ax.plot([0, vmax], [0, vmax], color="black", lw=1)
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=4)
    return finalize_figure(fig, out_dir / fname, _split_task_title(split, task, title))


def _hist_windows(res, split, col, xlabel, fname, title, task, out_dir, bins=None):
    wdf = res["windows"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    allv = wdf[col].to_numpy(np.float64)
    if bins is None:
        lo, hi = np.percentile(allv, [0.1, 99.9])
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, 80)
    for h in SPLIT_HOUSES[split]:
        v = wdf.loc[wdf["House"] == h, col].to_numpy(np.float64)
        ax.hist(v, bins=bins, histtype="step", lw=1.4, color=house_color(h), density=True,
                label=f"House {h:02d} (mean {v.mean():.3g}, MAE {np.abs(v).mean():.3g})")
    ax.hist(allv, bins=bins, histtype="step", lw=2.0, color="black", density=True,
            label=f"POOLED (mean {allv.mean():.3g}, MAE {np.abs(allv).mean():.3g})")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return finalize_figure(fig, out_dir / fname, _split_task_title(split, task, title))


def plot_residual_by_power_bin(res, split, task, out_dir):
    """Mean normalized residual as a function of actual output (fraction of rated power)."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    edges = np.linspace(0, 1.1, 12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for h, (r, _, a, rated) in _residuals_by_house(res, split).items():
        frac = a / rated
        means = [np.mean(r[(frac >= lo) & (frac < hi)] / rated[(frac >= lo) & (frac < hi)]) * 100
                 if np.any((frac >= lo) & (frac < hi)) else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centers, means, marker="o", color=house_color(h), label=f"House {h:02d}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Actual PV / Array_Rated_Power_W (bin centre)")
    ax.set_ylabel("Mean residual / Array_Rated_Power_W (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return finalize_figure(fig, out_dir / f"{task}_residual_by_actual_power_bin.png",
                           _split_task_title(split, task, "mean residual by actual-output bin (diagnostic only)"))


def make_split_plots(results, windows, split, paths):
    created = []
    for task in TASKS:
        res = results[task]
        out_dir = paths[f"plots_{split}_{task}"]
        for metric in POINT_METRIC_KEYS:
            created.append(plot_metric_by_horizon(res, split, task, metric, out_dir))
        created.append(plot_scatter(res, split, task, out_dir))
        created.append(plot_residual_histogram(res, split, task, out_dir))
        created.append(plot_residual_vs_predicted(res, split, task, out_dir))
        created.append(plot_abs_error_histogram(res, split, task, out_dir))
        created.append(plot_abs_error_cdf(res, split, task, out_dir))
        created.append(plot_metrics_by_house(res, split, task, out_dir))
        created.append(plot_example_windows(res, windows, split, task, out_dir))
        created.append(_scatter_windows(res, split, "Actual_Energy_kWh", "Predicted_Energy_kWh", "Actual window energy (kWh)",
                                        "Predicted window energy (kWh)", f"{task}_energy_actual_vs_predicted.png",
                                        "actual vs predicted window energy (valid daylight targets)", task, out_dir))
        created.append(_hist_windows(res, split, "Energy_Error_kWh", "Window energy error = predicted - actual (kWh)",
                                     f"{task}_energy_error_distribution.png", "window energy error distribution", task, out_dir))
        created.append(_scatter_windows(res, split, "Actual_Peak_W", "Predicted_Peak_W", "Actual window peak PV (W)",
                                        "Predicted window peak PV (W)", f"{task}_peak_power_actual_vs_predicted.png",
                                        "actual vs predicted window peak power", task, out_dir))
        created.append(_hist_windows(res, split, "Peak_Power_Error_W", "Peak power error = predicted - actual (W)",
                                     f"{task}_peak_power_error_distribution.png", "peak power error distribution", task, out_dir))
        max_slot = TASK_SLOTS[task]
        tbins = (np.arange(-max_slot, max_slot + 1) - 0.5) * STEP_MINUTES
        created.append(_hist_windows(res, split, "Peak_Time_Error_min", "Peak time error = predicted - actual (minutes)",
                                     f"{task}_peak_time_error_distribution.png", "peak time error distribution", task, out_dir, bins=tbins))
        created.append(plot_residual_by_power_bin(res, split, task, out_dir))
    return created


def plot_capacity_coverage(houses_static, path):
    """Static PV-system values of all houses with the training range shaded (descriptive only)."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    hs = sorted(houses_static)
    for ax, (k, col) in zip(axes, enumerate(STATIC_FEATURES)):
        vals = [houses_static[h][k] for h in hs]
        colors = ["tab:gray" if h in TRAIN_HOUSES else ("tab:blue" if h in VALIDATION_HOUSES else "tab:red") for h in hs]
        ax.bar([f"{h:02d}" for h in hs], vals, color=colors)
        tr = [houses_static[h][k] for h in TRAIN_HOUSES]
        ax.axhspan(min(tr), max(tr), color="0.85", zorder=0, label="training range (01-09)")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("House")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    return finalize_figure(fig, path, f"{ARCH_LABEL} - static PV-system inputs by house (grey = training, "
                                       "blue = validation, red = Test evaluation); shaded = training range")


# =============================================================================
# SECTION 13 - XGBOOST ENGINE: TASK/SLOT REGRESSORS UNDER ONE COMBINED INTERFACE
# =============================================================================
# Same-Day slot j (24 regressors): flattened history (LOOKBACK x [PV_kW, daylight_flag])
#                                  + flattened 24 x 8 target-time weather + 4 static.
# Next-Day slot j (25 regressors): flattened weather of Next-Day slots 0..j (causal prefix)
#                                  + 4 static. NO PV of any kind. Because slot j only sees
#                                  slots 0..j, the padding slot (index 24) cannot affect the
#                                  24 real outputs of the first issue S.
# Each regressor is trained on its valid daylight training targets (Houses 01-09) only.
# Its number of boosting rounds is chosen by early stopping on the valid daylight targets
# of Houses 10-11 only. Per-slot early stopping minimizes each slot's validation SSE, which
# minimizes the global masked validation MSE of each task (slots are independent).
ENGINE_KIND = "xgboost"
XGB_RUNTIME = {"device": "cpu"}


def engine_configure_runtime():
    random.seed(SEED)
    np.random.seed(SEED)
    info = {"seed": SEED, "requested_device": XGB_DEVICE}
    device = "cpu"
    if XGB_DEVICE in ("auto", "cuda"):
        try:
            rng = np.random.default_rng(0)
            X = rng.normal(size=(256, 4))
            probe = xgb.train({"tree_method": "hist", "device": "cuda", "verbosity": 0},
                              xgb.DMatrix(X, label=X[:, 0]), num_boost_round=2)
            used = json.loads(probe.save_config())["learner"]["generic_param"].get("device", "cpu")
            device = "cuda" if str(used).startswith("cuda") else "cpu"
        except Exception as exc:
            info["cuda_probe_error"] = repr(exc)
            if XGB_DEVICE == "cuda":
                raise
    elif XGB_DEVICE != "cpu":
        raise ValueError("XGB_DEVICE must be 'auto', 'cuda' or 'cpu'.")
    XGB_RUNTIME["device"] = device
    info["training_device"] = device
    info["prediction_device"] = "cpu"
    return info


def engine_environment_info():
    return {"xgboost": xgb.__version__}


def xgb_params():
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "lambda": REG_LAMBDA,
        "alpha": REG_ALPHA,
        "gamma": GAMMA,
        "max_bin": MAX_BIN,
        "tree_method": "hist",
        "device": XGB_RUNTIME["device"],
        "seed": SEED,
        "verbosity": 0,
    }


def _clean(name):
    return name.replace(" ", "_")


def xgb_same_day_feature_names(lookback):
    names = []
    for k in range(lookback):                       # oldest -> newest
        lag = (lookback - 1 - k) * STEP_MINUTES
        names += [f"hist_t-{lag}min_PV_kW", f"hist_t-{lag}min_daylight_flag"]
    for j in range(SD_SLOTS):
        names += [f"sd_t+{(j + 1) * STEP_MINUTES}min_{_clean(f)}" for f in WEATHER_FEATURES]
    names += [f"static_{_clean(f)}" for f in STATIC_FEATURES]
    return names


def xgb_next_day_feature_names(slot):
    names = []
    for s in range(slot + 1):                       # causal prefix: Next-Day slots 0..slot
        names += [f"nd_t+{ND_LEAD_STEPS * STEP_MINUTES + s * STEP_MINUTES}min_{_clean(f)}" for f in WEATHER_FEATURES]
    names += [f"static_{_clean(f)}" for f in STATIC_FEATURES]
    return names


def xgb_same_day_matrix(inputs):
    n = inputs["static"].shape[0]
    return np.concatenate([inputs["sd_history"].reshape(n, -1), inputs["sd_weather"].reshape(n, -1),
                           inputs["static"]], axis=1).astype(np.float32)


def xgb_next_day_matrix(inputs, slot):
    n = inputs["static"].shape[0]
    return np.concatenate([inputs["nd_weather"][:, : slot + 1, :].reshape(n, -1), inputs["static"]], axis=1).astype(np.float32)


def _train_one_slot(X_tr, y_tr, X_va, y_va, names):
    if len(y_tr) == 0 or len(y_va) == 0:
        raise RuntimeError("A slot regressor has no valid training or validation targets.")
    d_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=names)
    d_va = xgb.DMatrix(X_va, label=y_va, feature_names=names)
    evals_result = {}
    booster = xgb.train(
        xgb_params(), d_tr, num_boost_round=N_ESTIMATORS,
        evals=[(d_tr, "train"), (d_va, "validation")],      # early stopping uses the LAST set = validation
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, evals_result=evals_result, verbose_eval=False,
    )
    best_it = int(booster.best_iteration) if EARLY_STOPPING_ROUNDS else N_ESTIMATORS - 1
    final = booster[: best_it + 1]                         # the selected model contains exactly the best rounds
    final.set_param({"device": "cpu"})
    return final, {
        "best_iteration": best_it,
        "n_boosted_rounds_selected": best_it + 1,
        "n_rounds_run": len(evals_result["validation"]["rmse"]),
        "train_rmse_curve": [float(v) for v in evals_result["train"]["rmse"]],
        "validation_rmse_curve": [float(v) for v in evals_result["validation"]["rmse"]],
        "validation_rmse_at_best": float(evals_result["validation"]["rmse"][best_it]),
        "n_train_rows": int(len(y_tr)),
        "n_validation_rows": int(len(y_va)),
    }


def engine_train(train_w, val_w, var_sd, var_nd, paths):
    t0 = time.time()
    model = {"same_day": [], "next_day": [], "feature_names": {"same_day": xgb_same_day_feature_names(LOOKBACK),
                                                               "next_day": [xgb_next_day_feature_names(j) for j in range(ND_SLOTS)]}}
    slot_info = {"same_day": [], "next_day": []}
    X_tr_sd, X_va_sd = xgb_same_day_matrix(train_w["inputs"]), xgb_same_day_matrix(val_w["inputs"])
    for j in range(SD_SLOTS):
        rt, rv = train_w["sd_mask"][:, j], val_w["sd_mask"][:, j]
        b, info = _train_one_slot(X_tr_sd[rt], train_w["sd_y_kw"][rt, j], X_va_sd[rv], val_w["sd_y_kw"][rv, j],
                                  model["feature_names"]["same_day"])
        model["same_day"].append(b)
        slot_info["same_day"].append({"slot": j + 1, **info})
        print(f"  Same-Day slot {j + 1:02d}: best iteration {info['best_iteration']:4d} | "
              f"validation RMSE {info['validation_rmse_at_best'] * W_PER_KW:.2f} W")
    del X_tr_sd, X_va_sd
    for j in range(ND_SLOTS):
        rt, rv = train_w["nd_mask"][:, j], val_w["nd_mask"][:, j]
        X_tr, X_va = xgb_next_day_matrix(train_w["inputs"], j), xgb_next_day_matrix(val_w["inputs"], j)
        b, info = _train_one_slot(X_tr[rt], train_w["nd_y_kw"][rt, j], X_va[rv], val_w["nd_y_kw"][rv, j],
                                  model["feature_names"]["next_day"][j])
        model["next_day"].append(b)
        slot_info["next_day"].append({"slot": j + 1, **info})
        print(f"  Next-Day slot {j + 1:02d}: best iteration {info['best_iteration']:4d} | "
              f"validation RMSE {info['validation_rmse_at_best'] * W_PER_KW:.2f} W")
    train_seconds = time.time() - t0
    model["slot_info"] = slot_info

    sd_tr, nd_tr = engine_predict_raw_kw(model, train_w)
    sd_va, nd_va = engine_predict_raw_kw(model, val_w)
    mse = {
        "train_sd": numpy_global_masked_mse(train_w["sd_y_kw"], sd_tr, train_w["sd_mask"]),
        "train_nd": numpy_global_masked_mse(train_w["nd_y_kw"], nd_tr, train_w["nd_mask"]),
        "val_sd": numpy_global_masked_mse(val_w["sd_y_kw"], sd_va, val_w["sd_mask"]),
        "val_nd": numpy_global_masked_mse(val_w["nd_y_kw"], nd_va, val_w["nd_mask"]),
    }
    val_monitor = combined_monitor(mse["val_sd"], mse["val_nd"], var_sd, var_nd)
    train_monitor = combined_monitor(mse["train_sd"], mse["train_nd"], var_sd, var_nd)
    print(f"  selected model: train_monitor={train_monitor:.6f} | val_monitor={val_monitor:.6f} (lower is better)")

    history = _round_curves(slot_info, var_sd, var_nd)
    return model, {
        "history": history,
        "best_epoch": None,
        "best_val_monitor": float(val_monitor),
        "train_monitor": float(train_monitor),
        "val_global_mse_same_day_kw2": mse["val_sd"],
        "val_global_mse_next_day_kw2": mse["val_nd"],
        "train_global_mse_same_day_kw2": mse["train_sd"],
        "train_global_mse_next_day_kw2": mse["train_nd"],
        "selection_rule": "per-slot early stopping on Houses 10-11 validation RMSE (patience EARLY_STOPPING_ROUNDS); "
                          "the saved regressors contain exactly rounds 1..best_iteration+1",
        "per_slot": {t: [{k: v for k, v in s.items() if not k.endswith("_curve")} for s in slot_info[t]] for t in TASKS},
        "n_regressors": SD_SLOTS + ND_SLOTS,
        "device": XGB_RUNTIME["device"],
        "train_seconds": float(train_seconds),
    }


def _round_curves(slot_info, var_sd, var_nd):
    """Global training/validation monitor vs boosting round (each slot's curve is held at its last value
    after it stopped). Diagnostic only: the selected model uses each slot's own best iteration."""
    max_rounds = max(len(s["validation_rmse_curve"]) for t in TASKS for s in slot_info[t])
    rows = []
    for r in range(max_rounds):
        row = {"round": r + 1}
        for split, key, nkey in (("train", "train_rmse_curve", "n_train_rows"), ("val", "validation_rmse_curve", "n_validation_rows")):
            for task in TASKS:
                sse = sum((s[key][min(r, len(s[key]) - 1)] ** 2) * s[nkey] for s in slot_info[task])
                cnt = sum(s[nkey] for s in slot_info[task])
                row[f"{split}_global_mse_{task}_kw2"] = sse / cnt
            row[f"{split}_monitor"] = combined_monitor(row[f"{split}_global_mse_same_day_kw2"],
                                                       row[f"{split}_global_mse_next_day_kw2"], var_sd, var_nd)
        rows.append(row)
    return pd.DataFrame(rows)


def engine_predict_raw_kw(model, windows):
    inputs = windows["inputs"]
    d_sd = xgb.DMatrix(xgb_same_day_matrix(inputs), feature_names=model["feature_names"]["same_day"])
    sd = np.column_stack([b.predict(d_sd) for b in model["same_day"]])
    nd = np.column_stack([
        b.predict(xgb.DMatrix(xgb_next_day_matrix(inputs, j), feature_names=model["feature_names"]["next_day"][j]))
        for j, b in enumerate(model["next_day"])
    ])
    return sd.astype(np.float64), nd.astype(np.float64)


def load_xgb_model(model_dir):
    manifest = json.loads((Path(model_dir) / "manifest.json").read_text(encoding="utf-8"))
    model = {"same_day": [], "next_day": [], "feature_names": manifest["feature_names"]}
    for task in TASKS:
        for entry in manifest[task]:
            b = xgb.Booster()
            b.load_model(str(Path(model_dir) / entry["file"]))
            b.set_param({"device": "cpu"})
            if b.num_boosted_rounds() != entry["n_boosted_rounds"]:
                raise RuntimeError(f"{entry['file']}: unexpected number of boosted rounds after reload.")
            model[task].append(b)
    return model


def engine_save_and_reload(model, paths, val_w, var_sd, var_nd, training):
    model_dir = paths["model"] / f"combined_{ARCH_KEY}"
    model_dir.mkdir(parents=False, exist_ok=False)
    manifest = {"architecture": ARCH_LABEL, "feature_names": model["feature_names"], "same_day": [], "next_day": [],
                "public_outputs": {"same_day": SD_SLOTS, "next_day": "25 (24 for the first issue S; padding never reported)"}}
    for task in TASKS:
        for j, b in enumerate(model[task]):
            f = model_dir / f"{task}_slot_{j + 1:02d}.json"
            b.save_model(str(f))
            manifest[task].append({"slot": j + 1, "file": f.name, "sha256": file_sha256(f),
                                   "n_boosted_rounds": int(b.num_boosted_rounds())})
    write_json(model_dir / "manifest.json", manifest)
    reloaded = load_xgb_model(model_dir)
    sd0, nd0 = engine_predict_raw_kw(model, val_w)
    sd1, nd1 = engine_predict_raw_kw(reloaded, val_w)
    max_diff = float(max(np.max(np.abs(sd0 - sd1)), np.max(np.abs(nd0 - nd1))))
    mon = combined_monitor(numpy_global_masked_mse(val_w["sd_y_kw"], sd1, val_w["sd_mask"]),
                           numpy_global_masked_mse(val_w["nd_y_kw"], nd1, val_w["nd_mask"]), var_sd, var_nd)
    report = {
        "model_dir": str(model_dir),
        "n_regressor_files": SD_SLOTS + ND_SLOTS,
        "max_abs_prediction_diff_kw_validation": max_diff,
        "reloaded_val_monitor": float(mon),
        "logged_val_monitor": training["best_val_monitor"],
        "reloaded_reproduces_val_monitor": bool(np.isclose(mon, training["best_val_monitor"], rtol=1e-9, atol=1e-12)),
    }
    report["passed"] = bool(max_diff == 0.0 and report["reloaded_reproduces_val_monitor"])
    if not report["passed"]:
        raise RuntimeError(f"Save/reload verification failed: {report}")
    reloaded["slot_info"] = model.get("slot_info")
    return reloaded, report


def engine_verify_global_metric(model, val_w, var_sd, var_nd):
    """Vectorized global masked MSE vs an independent per-slot SSE accumulation and vs XGBoost's own
    validation RMSE at each selected iteration."""
    sd, nd = engine_predict_raw_kw(model, val_w)
    report = {}
    ok = True
    for task, pred in (("same_day", sd), ("next_day", nd)):
        pre = TASK_PREFIX[task]
        vec = numpy_global_masked_mse(val_w[f"{pre}_y_kw"], pred, val_w[f"{pre}_mask"])
        sse, cnt, slot_rel = 0.0, 0, []
        for j in range(TASK_SLOTS[task]):
            m = val_w[f"{pre}_mask"][:, j]
            e = pred[m, j] - val_w[f"{pre}_y_kw"][m, j].astype(np.float64)
            sse += float(np.dot(e, e))
            cnt += int(m.sum())
            if model.get("slot_info"):
                xgb_rmse = model["slot_info"][task][j]["validation_rmse_at_best"]
                slot_rel.append(abs(np.sqrt(np.mean(e * e)) - xgb_rmse) / max(xgb_rmse, 1e-12))
        loop = sse / cnt
        report[task] = {"vectorized_global_mse_kw2": vec, "per_slot_accumulated_global_mse_kw2": loop,
                        "max_rel_diff_vs_xgboost_validation_rmse": float(max(slot_rel)) if slot_rel else None}
        ok &= bool(np.isclose(vec, loop, rtol=1e-10)) and (not slot_rel or max(slot_rel) < 1e-4)
    report["val_monitor"] = combined_monitor(report["same_day"]["vectorized_global_mse_kw2"],
                                             report["next_day"]["vectorized_global_mse_kw2"], var_sd, var_nd)
    report["passed"] = bool(ok)
    if not ok:
        raise RuntimeError(f"Global metric verification failed: {report}")
    return report


def engine_information_flow_audit(model, sample_inputs, label):
    rng = np.random.default_rng(SEED)
    x = {k: np.array(v, dtype=np.float32, copy=True) for k, v in sample_inputs.items()}
    base_sd, base_nd = engine_predict_raw_kw(model, {"inputs": x})

    xh = {k: v.copy() for k, v in x.items()}
    xh["sd_history"][..., 0] = rng.uniform(0.0, 8.0, size=xh["sd_history"][..., 0].shape).astype(np.float32)
    xh["sd_history"][..., 1] = rng.integers(0, 2, size=xh["sd_history"][..., 1].shape).astype(np.float32)
    sd_h, nd_h = engine_predict_raw_kw(model, {"inputs": xh})

    xp = {k: v.copy() for k, v in x.items()}
    xp["nd_weather"][:, ND_PADDING_SLOT, :] += 3.0
    sd_p, nd_p = engine_predict_raw_kw(model, {"inputs": xp})

    xm = {k: v.copy() for k, v in x.items()}
    xm["nd_weather"][:, 12, :] += 3.0
    _, nd_m = engine_predict_raw_kw(model, {"inputs": xm})

    nd_names = model["feature_names"]["next_day"]
    sd_names = model["feature_names"]["same_day"]
    checks = {
        "next_day_features_contain_no_pv_or_history": all(not any(("hist" in n) or ("PV" in n) for n in names) for names in nd_names),
        "same_day_features_contain_full_history": sum(n.startswith("hist_") for n in sd_names) == 2 * LOOKBACK,
        "next_day_real_slots_exclude_padding_features": all(
            not any(n.startswith(f"nd_t+{ND_LEAD_STEPS * STEP_MINUTES + ND_PADDING_SLOT * STEP_MINUTES}min_") for n in nd_names[j])
            for j in range(ND_PADDING_SLOT)),
        "history_change_leaves_next_day_unchanged": bool(np.array_equal(nd_h, base_nd)),
        "padding_change_leaves_24_real_next_day_outputs_unchanged": bool(np.array_equal(nd_p[:, :ND_PADDING_SLOT], base_nd[:, :ND_PADDING_SLOT])),
        "padding_change_leaves_same_day_unchanged": bool(np.array_equal(sd_p, base_sd)),
        "next_day_slot12_change_leaves_slots_1_to_12_unchanged": bool(np.array_equal(nd_m[:, :12], base_nd[:, :12])),
    }
    report = {
        "label": label,
        "n_samples": int(len(x["static"])),
        "checks": checks,
        "history_change_changed_same_day_predictions": bool(not np.array_equal(sd_h, base_sd)),
        "max_abs_same_day_change_from_history": float(np.max(np.abs(sd_h - base_sd))),
        "max_abs_next_day_change_from_history": float(np.max(np.abs(nd_h - base_nd))),
        "max_abs_real_next_day_change_from_padding": float(np.max(np.abs(nd_p[:, :ND_PADDING_SLOT] - base_nd[:, :ND_PADDING_SLOT]))),
    }
    report["passed"] = bool(all(checks.values()))
    if not report["passed"]:
        raise RuntimeError(f"Model information-flow audit failed ({label}): {checks}")
    return report


def engine_self_tests():
    results = []

    def record(name, passed, details=None):
        results.append({"test": name, "passed": bool(passed), "details": details or {}})

    rng = np.random.default_rng(SEED)
    n = 40
    inputs = {
        "sd_history": np.stack([rng.uniform(0, 5, (n, LOOKBACK)), rng.integers(0, 2, (n, LOOKBACK))], -1).astype(np.float32),
        "sd_weather": rng.normal(size=(n, SD_SLOTS, len(WEATHER_FEATURES))).astype(np.float32),
        "nd_weather": rng.normal(size=(n, ND_SLOTS, len(WEATHER_FEATURES))).astype(np.float32),
        "static": rng.normal(size=(n, len(STATIC_FEATURES))).astype(np.float32),
    }
    sd_names = xgb_same_day_feature_names(LOOKBACK)
    X = xgb_same_day_matrix(inputs)
    record("xgb_same_day_feature_layout",
           X.shape[1] == len(sd_names) == 2 * LOOKBACK + SD_SLOTS * len(WEATHER_FEATURES) + len(STATIC_FEATURES)
           and np.array_equal(X[:, :2], inputs["sd_history"][:, 0, :])
           and np.array_equal(X[:, 2 * LOOKBACK - 2: 2 * LOOKBACK], inputs["sd_history"][:, -1, :])
           and len(set(sd_names)) == len(sd_names),
           {"n_features": int(X.shape[1])})
    nd_ok = True
    for j in range(ND_SLOTS):
        Xj = xgb_next_day_matrix(inputs, j)
        nd_ok &= Xj.shape[1] == len(xgb_next_day_feature_names(j)) == (j + 1) * len(WEATHER_FEATURES) + len(STATIC_FEATURES)
    record("xgb_next_day_causal_prefix_layout", nd_ok, {})
    mod = {k: v.copy() for k, v in inputs.items()}
    mod["sd_history"] += 5.0
    mod["nd_weather"][:, ND_PADDING_SLOT, :] += 5.0
    record("xgb_next_day_matrices_independent_of_history_and_padding",
           all(np.array_equal(xgb_next_day_matrix(inputs, j), xgb_next_day_matrix(mod, j)) for j in range(ND_PADDING_SLOT)), {})

    # Tiny trained combined model on random data: information-flow audit must pass.
    params = dict(xgb_params(), device="cpu")
    tiny = {"same_day": [], "next_day": [],
            "feature_names": {"same_day": sd_names, "next_day": [xgb_next_day_feature_names(j) for j in range(ND_SLOTS)]}}
    y = rng.normal(size=n)
    for j in range(SD_SLOTS):
        b = xgb.train(params, xgb.DMatrix(X, label=y + X[:, 0], feature_names=sd_names), num_boost_round=3)
        b.set_param({"device": "cpu"})
        tiny["same_day"].append(b)
    for j in range(ND_SLOTS):
        Xj = xgb_next_day_matrix(inputs, j)
        b = xgb.train(params, xgb.DMatrix(Xj, label=y + Xj[:, -9], feature_names=tiny["feature_names"]["next_day"][j]), num_boost_round=3)
        b.set_param({"device": "cpu"})
        tiny["next_day"].append(b)
    try:
        rep = engine_information_flow_audit(tiny, inputs, "self_test_tiny_model")
        record("model_information_flow_tiny_model", rep["passed"], rep["checks"])
    except RuntimeError as exc:
        record("model_information_flow_tiny_model", False, {"error": str(exc)})
    return results


def engine_plot_training(training, paths):
    hist = training["history"]
    created = []
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["round"], hist["train_monitor"], color="tab:blue", label="training loss (global, Houses 01-09)")
    ax.plot(hist["round"], hist["val_monitor"], color="tab:red", lw=2.2, label="validation loss (global, Houses 10-11)")
    ax.axhline(training["best_val_monitor"], color="tab:green", ls=":", lw=1.5,
               label=f"selected model (per-slot best iterations): val_monitor = {training['best_val_monitor']:.6f}")
    ax.set_xlabel("Boosting round (each slot held at its last value after early stopping)")
    ax.set_ylabel("0.5*MSE_SD/Var_SD + 0.5*MSE_ND/Var_ND")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    created.append(finalize_figure(
        fig, paths["plots_training"] / "loss_training_vs_validation.png",
        f"{ARCH_LABEL} - training vs validation loss. Official selection criterion: global validation monitor "
        f"(Houses 10-11), lower is better. Selected val_monitor = {training['best_val_monitor']:.6f}"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, task in zip(axes, TASKS):
        info = training["per_slot"][task]
        ax.bar([s["slot"] for s in info], [s["best_iteration"] + 1 for s in info], color="tab:purple")
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.set_xlabel("Forecast slot")
        ax.set_ylabel("Selected boosting rounds")
        ax.grid(alpha=0.3, axis="y")
    created.append(finalize_figure(fig, paths["plots_training"] / "selected_boosting_rounds_by_slot.png",
                                   f"{ARCH_LABEL} - boosting rounds selected by early stopping on Houses 10-11"))
    return created


# =============================================================================
# SECTION 14 - GENERIC SELF-TESTS (identical in all five programs)
# =============================================================================
def make_synthetic_house(day_specs, start="2022-06-01", house_id=99, daylight_zero_at=None, seed=SEED):
    """Synthetic 15-minute house with exactly known sunrise/sunset (for self-tests only)."""
    rng = np.random.default_rng(seed)
    n = len(day_specs) * STEPS_PER_DAY
    ts = ns_index(pd.date_range(start, periods=n, freq=STEP))
    minutes = (ts.hour * 60 + ts.minute).to_numpy()
    day_no = np.repeat(np.arange(len(day_specs)), STEPS_PER_DAY)
    zen = np.empty(n)
    for k, (sunrise, sunset) in enumerate(day_specs):
        sr = int(sunrise[:2]) * 60 + int(sunrise[3:])
        ss = int(sunset[:2]) * 60 + int(sunset[3:])
        sel = day_no == k
        m = minutes[sel]
        inside = (m > sr) & (m < ss)
        zen[sel] = np.where(inside, math.pi / 2 - 0.8 * np.sin(math.pi * (m - sr) / (ss - sr)), math.pi / 2 + 0.3)
    daylight = is_daylight(zen)
    pv_w = np.where(daylight, 3000.0 * np.clip(np.cos(zen), 0, None) + 50.0, 0.0)
    if daylight_zero_at is not None:
        pv_w[ts.get_loc(pd.Timestamp(daylight_zero_at))] = 0.0
    weather_raw = rng.normal(size=(n, len(WEATHER_FEATURES)))
    weather_raw[:, WEATHER_FEATURES.index(ZENITH_COL)] = zen
    house = {
        "house_id": house_id,
        "timestamps": ts,
        "weather_raw": weather_raw,
        "zenith": zen,
        "daylight": daylight,
        "pv_w": pv_w,
        "pv_kw": (pv_w / W_PER_KW).astype(np.float32),
        "static_raw": np.asarray([-0.0035, 10.0, 400.0, 4000.0]),
        "rated_w": 4000.0,
    }
    house["weather_scaled"] = weather_raw.astype(np.float32)
    house["static_scaled"] = np.zeros(len(STATIC_FEATURES), np.float32)
    return house


def run_generic_self_tests():
    results = []

    def record(name, passed, details=None):
        results.append({"test": name, "passed": bool(passed), "details": details or {}})

    def T(s):
        return pd.Timestamp(s)

    # Feature definitions.
    try:
        validate_static_configuration()
        ok = (len(WEATHER_FEATURES) == 8 and len(STATIC_FEATURES) == 4 and PANELS_COL in STATIC_FEATURES
              and "Solar_Azimuth_rad" in WEATHER_FEATURES and "Panel_Azimuth_rad" not in WEATHER_FEATURES + STATIC_FEATURES)
        record("feature_definitions_exact", ok, {"weather": WEATHER_FEATURES, "static": STATIC_FEATURES, "excluded": NON_INPUT_COLUMNS})
    except AssertionError as exc:
        record("feature_definitions_exact", False, {"error": repr(exc)})

    # Synthetic days: D0, D1 sunrise 07:03 / sunset 19:03; D2 07:20 / 16:50; D3, D4 07:00 / 19:00.
    days = [("07:03", "19:03"), ("07:03", "19:03"), ("07:20", "16:50"), ("07:00", "19:00"), ("07:00", "19:00")]
    house = make_synthetic_house(days, start="2022-06-01", daylight_zero_at="2022-06-02 08:45")
    acc, rej = build_joint_issue_schedule(house["timestamps"], house["daylight"], COMMON_MAX_LOOKBACK_STEPS)
    d1 = acc[pd.DatetimeIndex(acc["Issue_Timestamp"]).normalize() == T("2022-06-02")]
    got = [pd.Timestamp(t).strftime("%H:%M") for t in d1["Issue_Timestamp"]]
    expected = ["07:15", "09:00", "11:00", "13:00", "15:00"]
    first_flags = d1["Is_First_Issue_S"].tolist()
    record("schedule_construction_sunrise_0703",
           got == expected and first_flags == [True, False, False, False, False]
           and pd.Timestamp(d1["S_Timestamp"].iloc[0]) == T("2022-06-02 07:15")
           and pd.Timestamp(d1["H_Timestamp"].iloc[0]) == T("2022-06-02 07:00"),
           {"issues_day1": got, "expected": expected, "first_issue_flags": first_flags})
    d0 = rej[pd.DatetimeIndex(rej["Candidate_Timestamp"]).normalize() == T("2022-06-01")]
    record("schedule_rejects_insufficient_history", (d0["Reason"] == "insufficient_history").sum() >= 5 and
           not (pd.DatetimeIndex(acc["Issue_Timestamp"]).normalize() == T("2022-06-01")).any(),
           {"day0_rejections": d0["Reason"].value_counts().to_dict()})

    reasons = {pd.Timestamp(r.Candidate_Timestamp).strftime("%Y-%m-%d %H:%M"): r.Reason for r in rej.itertuples()}
    record("rejection_of_all_masked_windows",
           reasons.get("2022-06-02 17:00") == "next_day_all_targets_masked"
           and reasons.get("2022-06-02 19:00") == "same_day_all_targets_masked"
           and reasons.get("2022-06-02 21:00") == "issue_time_not_daylight",
           {k: v for k, v in reasons.items() if k.startswith("2022-06-02")})

    # Windows for the synthetic house.
    w = assemble_windows(house, acc["issue_idx"].to_numpy(), acc["Is_First_Issue_S"].to_numpy(bool), 5)
    its = pd.DatetimeIndex(w["issue_ts"])
    i_first = int(np.flatnonzero(its == T("2022-06-02 07:15"))[0])
    i_0900 = int(np.flatnonzero(its == T("2022-06-02 09:00"))[0])
    i_1300 = int(np.flatnonzero(its == T("2022-06-02 13:00"))[0])
    ndt = w["nd_target_ts"]
    record("first_issue_24_plus_1_padding",
           w["nd_real"][i_first].sum() == 24 and not w["nd_real"][i_first, ND_PADDING_SLOT]
           and not w["nd_mask"][i_first, ND_PADDING_SLOT]
           and np.all(w["inputs"]["nd_weather"][i_first, ND_PADDING_SLOT] == 0)
           and w["nd_y_kw"][i_first, ND_PADDING_SLOT] == 0
           and pd.Timestamp(ndt[i_first, 0]) == T("2022-06-03 07:15")
           and pd.Timestamp(ndt[i_first, 23]) == T("2022-06-03 13:00")
           and pd.isna(ndt[i_first, ND_PADDING_SLOT])
           and w["nd_real"][i_0900].all()
           and pd.Timestamp(ndt[i_0900, 0]) == T("2022-06-03 09:00")
           and pd.Timestamp(ndt[i_0900, 24]) == T("2022-06-03 15:00"),
           {"first_issue_real_slots": int(w["nd_real"][i_first].sum()), "normal_issue_real_slots": int(w["nd_real"][i_0900].sum())})
    record("tomorrow_daylight_masking",
           (not w["nd_mask"][i_first, 0]) and w["nd_mask"][i_first, 1] and int(w["nd_mask"][i_first].sum()) == 23
           and int(w["nd_mask"][i_1300].sum()) == 16,
           {"first_issue_valid": int(w["nd_mask"][i_first].sum()), "issue_1300_valid": int(w["nd_mask"][i_1300].sum())})
    record("same_day_targets_t15_to_t360",
           pd.Timestamp(w["sd_target_ts"][i_0900, 0]) == T("2022-06-02 09:15")
           and pd.Timestamp(w["sd_target_ts"][i_0900, -1]) == T("2022-06-02 15:00") and w["sd_target_ts"].shape[1] == 24, {})

    # Deployment schedule validation.
    frame = pd.DataFrame({TIMESTAMP_COL: house["timestamps"][96:], ZENITH_COL: house["zenith"][96:]})
    accepted_ok, rejected_ok, details = True, True, {}
    for t, first in (("2022-06-02 07:15", True), ("2022-06-02 09:00", False), ("2022-06-02 15:00", False)):
        try:
            info = validate_deployment_issue_time(T(t), frame)
            accepted_ok &= info["is_first_issue_S"] == first
            details[t] = "accepted"
        except ValueError as exc:
            accepted_ok = False
            details[t] = f"wrongly rejected: {exc}"
    for t in ("2022-06-02 09:15", "2022-06-02 07:30", "2022-06-02 10:00", "2022-06-02 05:00", "2022-06-02 17:00"):
        try:
            validate_deployment_issue_time(T(t), frame)
            rejected_ok = False
            details[t] = "wrongly accepted"
        except ValueError:
            details[t] = "rejected"
    record("deployment_schedule_validation", accepted_ok and rejected_ok, details)

    # Lookback timestamps (oldest -> newest).
    lb_details, lb_ok = {}, True
    for L, exp in ((3, ["08:30", "08:45", "09:00"]), (5, ["08:00", "08:15", "08:30", "08:45", "09:00"])):
        idx = history_indices([int(acc["issue_idx"].to_numpy()[acc["Issue_Timestamp"] == T("2022-06-02 09:00")][0])], L)[0]
        got_l = [house["timestamps"][k].strftime("%H:%M") for k in idx]
        lb_ok &= got_l == exp
        lb_details[f"L={L}"] = got_l
    idx96 = history_indices([int(acc["issue_idx"].to_numpy()[acc["Issue_Timestamp"] == T("2022-06-02 09:00")][0])], 96)[0]
    first96, last96 = house["timestamps"][idx96[0]], house["timestamps"][idx96[-1]]
    lb_ok &= (len(idx96) == 96 and first96 == T("2022-06-01 09:15") and last96 == T("2022-06-02 09:00"))
    lb_details["L=96"] = [str(first96), str(last96)]
    record("lookback_timestamps", lb_ok, lb_details)

    # Identical eligible issue timestamps for every LOOKBACK <= 96.
    issue_sets = {L: assemble_windows(house, acc["issue_idx"].to_numpy(), acc["Is_First_Issue_S"].to_numpy(bool), L)["issue_ts"]
                  for L in (1, 3, 16, 96)}
    record("identical_issue_timestamps_for_all_lookbacks",
           all(np.array_equal(issue_sets[1], v) for v in issue_sets.values()), {"n_issues": int(len(issue_sets[1]))})

    # Night-zero vs daylight-zero history rows.
    hist_first = w["inputs"]["sd_history"][i_first]          # 06:15 ... 07:15, L=5
    w3 = assemble_windows(house, acc["issue_idx"].to_numpy(), acc["Is_First_Issue_S"].to_numpy(bool), 3)
    hist_0900 = w3["inputs"]["sd_history"][i_0900]           # 08:30, 08:45 (daylight PV = 0), 09:00
    record("night_zero_vs_daylight_zero_history",
           np.array_equal(hist_first[:4], np.zeros((4, 2), np.float32)) and hist_first[4, 1] == 1.0 and hist_first[4, 0] > 0
           and hist_0900[1, 0] == 0.0 and hist_0900[1, 1] == 1.0,
           {"history_0715_L5": hist_first.tolist(), "history_0900_L3": hist_0900.tolist()})

    # NumPy masking.
    y = np.array([[1.0, 2.0, 3.0]])
    p = np.array([[2.0, 2.0, 1000.0]])
    m = np.array([[True, True, False]])
    record("numpy_global_masked_mse_ignores_masked_points", np.isclose(numpy_global_masked_mse(y, p, m), 0.5),
           {"value": numpy_global_masked_mse(y, p, m)})
    record("val_monitor_formula", np.isclose(combined_monitor(2.0, 6.0, 4.0, 3.0), 1.25), {})

    # Point metrics against hand-computed values.
    a = np.array([[100.0, 200.0, 0.0, 50.0, 9999.0]])
    p = np.array([[110.0, 190.0, 10.0, 40.0, 0.0]])
    m = np.array([[True, True, True, True, False]])
    r = np.full(a.shape, 1000.0)
    pm = point_metrics(a, p, r, m)
    expected_pm = {
        "RMSE_W": 10.0, "MAE_W": 10.0, "nRMSE_pct": 1.0, "WAPE_pct": 100 * 40 / 350,
        "R2": 1 - 400 / 21875, "sMAPE_pct": 100 * np.mean([20 / 210, 20 / 390, 2.0, 20 / 90]),
        "MAPE_1pct_pct": 100 * np.mean([0.1, 0.05, 0.2]),
    }
    record("point_metrics_known_values",
           all(np.isclose(pm[k], v, rtol=1e-12) for k, v in expected_pm.items())
           and pm["N_MAPE_1pct_included"] == 3 and pm["N_MAPE_1pct_excluded"] == 1 and pm["N_points"] == 4,
           {k: pm[k] for k in list(expected_pm) + ["N_MAPE_1pct_included", "N_MAPE_1pct_excluded"]})

    # Post-processing.
    pw = postprocess_prediction_w(np.array([-0.5, 1.2, 0.3, 0.0]), np.array([True, True, False, True]))
    record("postprocessing_clip_and_night_zero", np.array_equal(pw, np.array([0.0, 1200.0, 0.0, 0.0])), {"output_W": pw.tolist()})

    # HEMS metrics.
    hw = window_hems_table(np.array([[0.0, 1000.0, 2000.0, 1000.0]]), np.array([[500.0, 1000.0, 1500.0, 1800.0]]),
                           np.array([[False, True, True, True]]))
    hs = hems_summary(hw)
    record("hems_metrics_known_values",
           np.isclose(hw["Actual_Energy_kWh"].iloc[0], 1.0) and np.isclose(hw["Predicted_Energy_kWh"].iloc[0], 1.075)
           and np.isclose(hs["Peak_Power_Bias_W"], -200.0) and np.isclose(hs["Peak_Time_Bias_min"], 15.0)
           and np.isclose(hs["Energy_WAPE_pct"], 7.5),
           {k: hs[k] for k in HEMS_METRIC_KEYS})

    # Rated-power consistency.
    record("rated_power_consistency_check",
           check_rated_power_consistency(4000.0, 10, 400.0)[0] and not check_rated_power_consistency(4001.0, 10, 400.0)[0]
           and not check_rated_power_consistency(4000.1, 10, 400.0)[0], {})

    # Scalers use Houses 01-09 only.
    fake = {}
    for h in TRAIN_HOUSES:
        fake[h] = {"weather_raw": np.full((5, 8), float(h)), "static_raw": np.full(4, float(h))}
    fake[10] = {"weather_raw": np.full((5, 8), 1e6), "static_raw": np.full(4, 1e6)}
    ws, ss, meta = fit_scalers(fake)
    record("scalers_fitted_on_houses_01_09_only",
           np.allclose(ws.mean_, 5.0) and np.allclose(ss.mean_, 5.0) and meta["fitted_on_houses"] == TRAIN_HOUSES, {})

    # Houses 12-13 locked before the Test-evaluation phase.
    locked = False
    if not HOUSE_ACCESS["test_unlocked"]:
        try:
            load_house(TEST_HOUSES[0], "self_test")
        except RuntimeError:
            locked = True
    record("test_houses_locked_before_test_evaluation", locked, {})
    return results


# =============================================================================
# SECTION 15 - RUN DIRECTORY, METADATA, MAIN EXPERIMENT
# =============================================================================
def create_run_paths():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ARCH_KEY}_{stamp}_{uuid.uuid4().hex[:8]}"
    root = Path(RESULTS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"run_{run_id}"
    run_dir.mkdir(parents=False, exist_ok=False)   # an existing run can never be overwritten
    layout = {
        "model": "model", "scalers": "scalers", "config": "config", "training": "training",
        "integrity": "integrity", "self_tests": "self_tests", "forecasts": "forecasts",
        "metrics_validation": "metrics/validation", "metrics_test_evaluation": "metrics/test_evaluation",
        "plots_training": "plots/training", "plots_data": "plots/data",
    }
    for split in SPLIT_HOUSES:
        for task in TASKS:
            layout[f"plots_{split}_{task}"] = f"plots/{split}/{task}"
    paths = {"run": run_dir}
    for key, rel in layout.items():
        paths[key] = run_dir / rel
        paths[key].mkdir(parents=True, exist_ok=True)
    RUN_STATE["run_id"] = run_id
    RUN_STATE["run_dir"] = run_dir
    return run_id, paths


def feature_definitions():
    return {
        "weather_solar_inputs_target_time": WEATHER_FEATURES,
        "static_pv_system_inputs": STATIC_FEATURES,
        "static_redundancy_note": "Array_Rated_Power_W = Number_of_Panels * Pmax_Per_Panel_W is intentionally kept "
                                  "(identical in all five architectures).",
        "excluded_non_input_columns": NON_INPUT_COLUMNS,
        "target": TARGET_COL,
        "target_unit_inside_model": "kW (fixed W/1000 conversion); all reported forecasts and metrics in W",
        "daylight_rule": "Solar_Zenith_rad < pi/2 (the only daylight rule; night is never inferred from PV == 0)",
        "same_day_inputs": {
            "sd_history": f"(LOOKBACK={LOOKBACK}, 2) rows [PV_kW, daylight_flag] at t-(LOOKBACK-1)*15 min ... t, oldest -> newest; "
                          "night = [0, 0], daylight zero = [0, 1]",
            "sd_weather": "(24, 8) scaled weather at the target times t+15 min ... t+360 min",
            "static": "(4,) scaled static PV-system inputs",
        },
        "next_day_inputs": {
            "nd_weather": "(25, 8) scaled weather at t+24 h ... t+30 h; first issue S: 24 real rows (S ... S+5h45 tomorrow) "
                          "+ 1 structural zero padding row at the END",
            "static": "(4,) scaled static PV-system inputs",
            "pv_inputs": "NONE (no past, current, predicted or future PV)",
        },
        "outputs": {"same_day": "(24,) kW, t+15 ... t+360 min", "next_day": "(25,) kW; padding slot never reported"},
        "common_max_lookback_steps": COMMON_MAX_LOOKBACK_STEPS,
        "issue_schedule": "S = first daylight 15-min grid point of the day; H = floor(S to hour); issues S, H+2h, H+4h, ... "
                          "while the issue time is daylight; kept only if both tasks have >= 1 valid daylight target",
        "postprocessing": "negative -> 0 W; night target -> exactly 0 W",
        "oracle_weather_statement": ORACLE_WEATHER_STATEMENT,
    }


def environment_info(runtime_info, drive_status):
    import sklearn
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "google_drive": drive_status,
        "runtime": runtime_info,
    }
    info.update(engine_environment_info())
    return info


def capacity_coverage_table(houses):
    rows = []
    train_static = np.vstack([houses[h]["static_raw"] for h in TRAIN_HOUSES])
    lo, hi = train_static.min(axis=0), train_static.max(axis=0)
    for h in sorted(houses):
        split = "train" if h in TRAIN_HOUSES else ("validation" if h in VALIDATION_HOUSES else "test_evaluation")
        row = {"House": h, "Split": split}
        for k, col in enumerate(STATIC_FEATURES):
            v = houses[h]["static_raw"][k]
            row[col] = v
            row[f"{col}_inside_training_range"] = bool(lo[k] <= v <= hi[k])
        row["PV_max_W"] = houses[h]["integrity"]["pv_max_w"]
        row["PV_max_fraction_of_rated"] = houses[h]["integrity"]["pv_max_fraction_of_rated"]
        rows.append(row)
    return pd.DataFrame(rows)


def lookback_audit(windows, n=3):
    rows = []
    for i in range(min(n, len(windows["issue_ts"]))):
        issue = pd.Timestamp(windows["issue_ts"][i])
        hist_ts = pd.date_range(end=issue, periods=LOOKBACK, freq=STEP)
        rows.append({
            "House": int(windows["house"][i]),
            "Issue_Timestamp": issue,
            "History_Timestamps_oldest_to_newest": [str(t) for t in hist_ts],
            "History_first_from_windows": str(pd.Timestamp(windows["history_first_ts"][i])),
            "History_last_from_windows": str(pd.Timestamp(windows["history_last_ts"][i])),
            "History_rows_[PV_kW,daylight_flag]": windows["inputs"]["sd_history"][i].tolist(),
        })
    return rows


def information_flow_sample(windows, n=64):
    first = np.flatnonzero(windows["is_first_issue"])[: n // 2]
    other = np.flatnonzero(~windows["is_first_issue"])[: n - len(first)]
    idx = np.sort(np.concatenate([first, other]))
    return {k: v[idx] for k, v in windows["inputs"].items()}


def window_counts(w):
    return {
        "n_windows": int(len(w["issue_ts"])),
        "n_first_issue_S_windows": int(w["is_first_issue"].sum()),
        "n_same_day_valid_targets": int(w["sd_mask"].sum()),
        "n_next_day_valid_targets": int(w["nd_mask"].sum()),
        "n_next_day_padding_slots": int((~w["nd_real"]).sum()),
        "schedule_sha256": schedule_sha256(w),
    }


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)   # live progress when run with "!python"
    except Exception:
        pass
    t_start = time.time()
    started_utc = utc_now_iso()
    section(f"{ARCH_LABEL} - COMBINED SAME-DAY + NEXT-DAY PV FORECASTING - ONE COMPLETE EXPERIMENT")
    print(ORACLE_WEATHER_STATEMENT)
    validate_static_configuration()
    drive_status = mount_google_drive_if_colab()
    if not Path(DATA_DIR).exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")
    runtime_info = engine_configure_runtime()
    run_id, paths = create_run_paths()
    print(f"Run ID       : {run_id}")
    print(f"Run directory: {paths['run']}")
    print(f"Manual hyperparameters: {hyperparameter_text()}")

    write_json(paths["config"] / "manual_hyperparameters.json",
               {"architecture": ARCH_LABEL, "run_id": run_id, "hyperparameters": MANUAL_HYPERPARAMETERS})
    write_json(paths["config"] / "feature_definitions.json", feature_definitions())
    write_json(paths["config"] / "environment.json", environment_info(runtime_info, drive_status))

    # ------------------------------------------------------------------ self-tests
    section("SELF-TESTS")
    tests = run_generic_self_tests() + engine_self_tests()
    write_json(paths["self_tests"] / "self_test_results.json", tests)
    for t in tests:
        print(f"  [{'PASS' if t['passed'] else 'FAIL'}] {t['test']}")
    failed = [t["test"] for t in tests if not t["passed"]]
    if failed:
        raise RuntimeError(f"Self-tests failed: {failed}")

    # ------------------------------------------------------------------ data (Houses 01-11 only)
    section("DATA - Houses 01-11 only (Houses 12-13 remain locked)")
    houses = {}
    for h in TRAIN_HOUSES + VALIDATION_HOUSES:
        houses[h] = load_house(h, "train_validation")
        rep = houses[h]["integrity"]
        print(f"  House {h:02d}: {rep['n_rows']:,} rows | {rep['first_timestamp']} -> {rep['last_timestamp']} | "
              f"rated {houses[h]['rated_w']:.1f} W | panels {rep['static_values'][PANELS_COL]:.0f} | "
              f"daylight zero-PV rows {rep['n_daylight_rows_with_zero_pv']}")
    write_json(paths["integrity"] / "data_integrity_report_houses_01_11.json", {h: houses[h]["integrity"] for h in houses})

    # ------------------------------------------------------------------ scalers
    section("SCALERS - fitted on Houses 01-09 only")
    weather_scaler, static_scaler, scaler_meta = fit_scalers(houses)
    for h in houses:
        apply_scalers(houses[h], weather_scaler, static_scaler)
    joblib.dump(weather_scaler, paths["scalers"] / "weather_scaler.joblib")
    joblib.dump(static_scaler, paths["scalers"] / "static_scaler.joblib")
    write_json(paths["scalers"] / "scaler_metadata.json", scaler_meta)
    print(f"  weather scaler: {scaler_meta['weather_scaler']['n_rows']:,} rows of Houses {TRAIN_HOUSES}")
    print(f"  static scaler : {scaler_meta['static_scaler']['n_rows']} rows of Houses {TRAIN_HOUSES}")

    # ------------------------------------------------------------------ schedule + windows
    section("JOINT ISSUE SCHEDULE AND FORECAST WINDOWS")
    train_w, train_audit, train_rej = build_split_windows(houses, TRAIN_HOUSES, LOOKBACK)
    val_w, val_audit, val_rej = build_split_windows(houses, VALIDATION_HOUSES, LOOKBACK)
    train_audit.to_csv(paths["integrity"] / "schedule_train_houses_01_09.csv.gz", index=False)
    val_audit.to_csv(paths["integrity"] / "schedule_validation_houses_10_11.csv.gz", index=False)
    pd.concat([train_rej.assign(Split="train"), val_rej.assign(Split="validation")]).groupby(
        ["Split", "House", "Reason"]).size().rename("N").reset_index().to_csv(
        paths["integrity"] / "schedule_rejections_houses_01_11.csv", index=False)
    var_sd, var_nd, n_var_sd, n_var_nd = training_target_variances(train_w)
    counts = {"train": window_counts(train_w), "validation": window_counts(val_w)}
    for k, v in counts.items():
        print(f"  {k:10s}: {v['n_windows']:,} windows ({v['n_first_issue_S_windows']:,} first issues S) | "
              f"valid targets SD {v['n_same_day_valid_targets']:,}, ND {v['n_next_day_valid_targets']:,}")
    print(f"  Var_SameDay_training = {var_sd:.6f} kW^2 ({n_var_sd:,} points) | "
          f"Var_NextDay_training = {var_nd:.6f} kW^2 ({n_var_nd:,} points)")
    write_json(paths["integrity"] / "lookback_audit.json", {"LOOKBACK": LOOKBACK, "examples": lookback_audit(val_w)})
    dep_sched = verify_deployment_schedule(houses, VALIDATION_HOUSES)
    print(f"  deployment schedule reconstruction matches training schedule on {dep_sched['days_checked']} validation days")

    # ------------------------------------------------------------------ training
    section("TRAINING (Houses 01-09) - SELECTION ON THE GLOBAL VALIDATION MONITOR (Houses 10-11)")
    print(VAL_MONITOR_DEFINITION)
    model, training = engine_train(train_w, val_w, var_sd, var_nd, paths)
    training["history"].to_csv(paths["training"] / "training_history.csv", index=False)
    best_info = {k: v for k, v in training.items() if k not in ("history",)}
    best_info.update({"val_monitor_definition": VAL_MONITOR_DEFINITION, "variance_same_day_training_kw2": var_sd,
                      "variance_next_day_training_kw2": var_nd, "lower_is_better": True})
    write_json(paths["training"] / "best_model_selection.json", best_info)
    engine_plot_training(training, paths)
    if training.get("best_epoch") is not None:
        print(f"  BEST EPOCH = {training['best_epoch']} | BEST val_monitor = {training['best_val_monitor']:.6f} (lower is better)")
    else:
        print(f"  val_monitor = {training['best_val_monitor']:.6f} (lower is better)")

    # ------------------------------------------------------------------ save -> reload -> verify
    section("SAVE -> RELOAD -> VERIFY")
    model, save_report = engine_save_and_reload(model, paths, val_w, var_sd, var_nd, training)
    write_json(paths["integrity"] / "save_reload_verification.json", save_report)
    print(f"  save/reload verification passed: {save_report['passed']}")
    metric_report = engine_verify_global_metric(model, val_w, var_sd, var_nd)
    print(f"  global metric verification passed: {metric_report['passed']}")
    flow_report = engine_information_flow_audit(model, information_flow_sample(val_w), "trained_reloaded_model_validation_windows")
    write_json(paths["integrity"] / "model_information_flow_audit.json", flow_report)
    print(f"  model information-flow audit passed: {flow_report['passed']}")

    # ------------------------------------------------------------------ validation evaluation
    section("VALIDATION - Houses 10-11 (official hyperparameter-selection houses)")
    val_results = evaluate_split(model, val_w, "validation")
    val_results["monitor_value"] = combined_monitor(val_results["global_masked_mse_kw2"]["same_day"],
                                                    val_results["global_masked_mse_kw2"]["next_day"], var_sd, var_nd)
    if not np.isclose(val_results["monitor_value"], training["best_val_monitor"], rtol=1e-6, atol=1e-12):
        raise RuntimeError("Validation evaluation does not reproduce the selected val_monitor.")
    save_split_outputs(val_w, val_results, "validation", paths)
    val_csv_check = verify_metrics_from_saved_forecasts(val_results, "validation")
    print_split_summary(val_results, "validation")
    dep_pred = verify_deployment_predictions(model, houses, val_w, val_results, weather_scaler, static_scaler)
    print(f"\n  deployment helper reproduces evaluation forecasts: {dep_pred['passed']} "
          f"(max |diff| SD {dep_pred['max_abs_diff_same_day_W']:.2e} W, ND {dep_pred['max_abs_diff_next_day_W']:.2e} W)")
    make_split_plots(val_results, val_w, "validation", paths)

    # ------------------------------------------------------------------ Test evaluation (Houses 12-13)
    section("TEST EVALUATION - Houses 12-13 (unlocked only now; never used for any selection)")
    fingerprint_before = scaler_fingerprint(weather_scaler, static_scaler)
    HOUSE_ACCESS["test_unlocked"] = True
    for h in TEST_HOUSES:
        houses[h] = load_house(h, "test_evaluation")
        apply_scalers(houses[h], weather_scaler, static_scaler)
    if scaler_fingerprint(weather_scaler, static_scaler) != fingerprint_before:
        raise RuntimeError("Scalers changed after loading the Test-evaluation houses.")
    test_w, test_audit, test_rej = build_split_windows(houses, TEST_HOUSES, LOOKBACK)
    test_audit.to_csv(paths["integrity"] / "schedule_test_evaluation_houses_12_13.csv.gz", index=False)
    test_rej.groupby(["House", "Reason"]).size().rename("N").reset_index().to_csv(
        paths["integrity"] / "schedule_rejections_houses_12_13.csv", index=False)
    counts["test_evaluation"] = window_counts(test_w)
    test_results = evaluate_split(model, test_w, "test_evaluation")
    test_results["monitor_value"] = combined_monitor(test_results["global_masked_mse_kw2"]["same_day"],
                                                     test_results["global_masked_mse_kw2"]["next_day"], var_sd, var_nd)
    save_split_outputs(test_w, test_results, "test_evaluation", paths)
    test_csv_check = verify_metrics_from_saved_forecasts(test_results, "test_evaluation")
    print_split_summary(test_results, "test_evaluation")
    make_split_plots(test_results, test_w, "test_evaluation", paths)

    coverage = capacity_coverage_table(houses)
    coverage.to_csv(paths["integrity"] / "static_input_coverage_vs_training_range.csv", index=False)
    plot_capacity_coverage({h: houses[h]["static_raw"] for h in houses}, paths["plots_data"] / "static_inputs_by_house.png")
    outside = coverage[(coverage["Split"] != "train") & (~coverage[f"{RATED_POWER_COL}_inside_training_range"])]["House"].tolist()
    write_json(paths["integrity"] / "data_integrity_report_all_houses.json", {h: houses[h]["integrity"] for h in houses})
    write_json(paths["integrity"] / "independent_metric_verification.json", {
        "global_metric_tf_or_engine_vs_numpy": metric_report,
        "validation_metrics_recomputed_from_saved_csv": val_csv_check,
        "test_evaluation_metrics_recomputed_from_saved_csv": test_csv_check,
        "validation_evaluation_reproduces_selected_val_monitor": True,
        "deployment_schedule_verification": dep_sched,
        "deployment_prediction_verification": dep_pred,
    })

    # ------------------------------------------------------------------ run metadata + summary
    section("RUN SUMMARY")
    elapsed = time.time() - t_start
    def pooled(results, task, key):
        pt = results[task]["point"]
        return float(pt.loc[pt["Scope"] == "POOLED", key].iloc[0])

    def macro(results, task, key):
        pt = results[task]["point"]
        return float(pt.loc[pt["Scope"] == "MACRO_AVERAGE", key].iloc[0])

    metadata = {
        "run_id": run_id,
        "architecture": ARCH_LABEL,
        "architecture_key": ARCH_KEY,
        "engine": ENGINE_KIND,
        "started_utc": started_utc,
        "finished_utc": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "data_dir": str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "run_dir": str(paths["run"]),
        "house_file_pattern": HOUSE_FILE_PATTERN,
        "train_houses": TRAIN_HOUSES,
        "validation_houses": VALIDATION_HOUSES,
        "test_evaluation_houses": TEST_HOUSES,
        "test_evaluation_note": "Houses 12-13 are labelled 'Test evaluation' because their results are visible during manual "
                                "experimentation. They were never used for training, scaler fitting, early stopping, "
                                "checkpoint selection or hyperparameter selection.",
        "house_file_sha256": {h: houses[h]["integrity"]["sha256"] for h in houses},
        "house_access_log": HOUSE_ACCESS["log"],
        "oracle_weather_statement": ORACLE_WEATHER_STATEMENT,
        "manual_hyperparameters": MANUAL_HYPERPARAMETERS,
        "val_monitor_definition": VAL_MONITOR_DEFINITION,
        "official_selection_result": {
            "best_epoch": training.get("best_epoch"),
            "best_val_monitor": training["best_val_monitor"],
            "lower_is_better": True,
        },
        "variance_same_day_training_kw2": var_sd,
        "variance_next_day_training_kw2": var_nd,
        "window_counts": counts,
        "secondary_validation_diagnostics": {
            task: {"macro_nRMSE_pct": macro(val_results, task, "nRMSE_pct"), "macro_WAPE_pct": macro(val_results, task, "WAPE_pct"),
                   "pooled_R2": pooled(val_results, task, "R2"),
                   "by_house": val_results[task]["point"][["Scope"] + POINT_METRIC_KEYS].to_dict(orient="records")}
            for task in TASKS},
        "test_evaluation_monitor_reporting_only": test_results["monitor_value"],
        "validation_or_test_houses_outside_training_rated_power_range": outside,
        "interpretation_guidance": (
            "Systematic under-prediction at high PV output for houses whose static PV-system values lie outside the "
            "training range is consistent with extrapolation beyond the system-capacity range represented in the "
            "training houses; this does not prove that system capacity is the only cause. Panel tilt is excluded from "
            "the inputs and no claim about tilt is made without a controlled experiment."),
        "scaler_metadata": scaler_meta,
        "schedule_definition": feature_definitions()["issue_schedule"],
    }
    if ENGINE_KIND == "neural":
        metadata["training_summary"] = {k: training[k] for k in ("epochs_run", "stopped_early", "stopped_epoch",
                                                                  "trainable_parameters", "train_seconds")}
    else:
        metadata["training_summary"] = {k: training[k] for k in ("n_regressors", "device", "train_seconds")}
    write_json(paths["config"] / "run_metadata.json", metadata)

    lines = [
        f"{ARCH_LABEL} - combined Same-Day + Next-Day PV forecasting",
        f"Run ID: {run_id}",
        f"Run directory: {paths['run']}",
        ORACLE_WEATHER_STATEMENT,
        f"Manual hyperparameters: {hyperparameter_text()}",
        "",
        "OFFICIAL SELECTION CRITERION (Validation, Houses 10-11; lower is better):",
        f"  best epoch     : {training.get('best_epoch')}",
        f"  best val_monitor: {training['best_val_monitor']:.6f}",
        "",
        "Secondary validation diagnostics (Houses 10-11):",
    ]
    for task in TASKS:
        lines.append(f"  {TASK_LABELS[task]}: macro nRMSE {macro(val_results, task, 'nRMSE_pct'):.3f} % | "
                     f"macro WAPE {macro(val_results, task, 'WAPE_pct'):.3f} % | pooled R2 {pooled(val_results, task, 'R2'):.4f}")
        for row in val_results[task]["point"].itertuples():
            if row.Scope.startswith("House"):
                lines.append(f"      {row.Scope}: nRMSE {row.nRMSE_pct:.3f} % | WAPE {row.WAPE_pct:.3f} % | R2 {row.R2:.4f}")
    lines += ["", "Test evaluation (Houses 12-13) - visible results, NEVER used for selection:"]
    for task in TASKS:
        lines.append(f"  {TASK_LABELS[task]}: macro nRMSE {macro(test_results, task, 'nRMSE_pct'):.3f} % | "
                     f"macro WAPE {macro(test_results, task, 'WAPE_pct'):.3f} % | pooled R2 {pooled(test_results, task, 'R2'):.4f}")
        for row in test_results[task]["point"].itertuples():
            if row.Scope.startswith("House"):
                lines.append(f"      {row.Scope}: nRMSE {row.nRMSE_pct:.3f} % | WAPE {row.WAPE_pct:.3f} % | R2 {row.R2:.4f}")
    lines += [
        f"  combined monitor on Test evaluation (reporting only): {test_results['monitor_value']:.6f}",
        "",
        f"Validation/Test houses with Array_Rated_Power_W outside the training range (descriptive): {outside}",
        "Interpretation guidance: " + metadata["interpretation_guidance"],
        f"Elapsed: {elapsed / 60:.1f} min",
    ]
    summary_text = "\n".join(lines)
    (paths["run"] / "RUN_SUMMARY.txt").write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    print(f"\nAll outputs saved in: {paths['run']}")
    return paths["run"]


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A failed run keeps its (partial) folder but is clearly marked, so it can never be
        # mistaken for a completed experiment.
        if RUN_STATE.get("run_dir") is not None:
            Path(RUN_STATE["run_dir"], "RUN_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
