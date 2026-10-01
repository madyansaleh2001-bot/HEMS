# =============================================================================
# COMBINED SAME-DAY + NEXT-DAY EXPERIMENT (Experiment A) — PHASE 2
# FINAL UNSEEN TEST — Houses 12-13
#
# This is the ONLY script that opens Houses 12-13, and it runs only after
# select_architecture.py has written frozen_selection.json.
#
# It changes NOTHING: no training, no tuning, no refitting of scalers, no change of
# models, hyperparameters, seeds or schedules. Every frozen Phase-1 model
# (5 architectures x 2 tasks x seeds 42, 43, 44) is evaluated, not only the winner.
# The architecture selected on House 11 stays the selected architecture whatever the
# test results show.
#
# Schedules (identical to Phase 1): first target of each day = first 15-min daylight
# grid point; first issue = first target - offset - 15 min; then every cadence while the
# first target is daylight. Same-Day: offset 0, every 15 min, targets issue+15..issue+360.
# Next-Day: offset 24 h, every 2 h, targets issue+24h15..issue+30h.
#
# Consistency checks before any test metric is computed:
#   - Phase-1 summaries are unchanged since selection (SHA-256).
#   - House 1-11 data files are unchanged (SHA-256 of House 11 re-checked).
#   - All ten runs saved identical scalers (fitted on Houses 1-10); they are re-used.
#   - The House-11 schedule rebuilt here equals the Phase-1 schedule (per task).
#   - Each frozen model, run through THIS script's pipeline on House 11, reproduces the
#     predictions saved in Phase 1 (proves identical features, order and scaling).
#
# Reporting: Primary RMSE, MAE, nRMSE, WAPE, R2 | Secondary sMAPE |
# Supplementary MAPE_1pct (+ N included / N excluded), reported per task and never
# pooled across tasks | HEMS planning metrics. Per seed and mean ± SD (ddof=1).
# =============================================================================

import os, gc, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBRegressor

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = Path('/content/drive/MyDrive/Multi-House')
except ImportError:  # outside Colab: point PV_DATA_DIR at the folder with the CSVs
    DATA_DIR = Path(os.environ.get('PV_DATA_DIR', 'Multi-House'))
try:
    from IPython.display import display
except ImportError:
    display = print

EXPERIMENT_DIR = DATA_DIR / 'combined_sameday_nextday_experiment'
PHASE1_DIR = EXPERIMENT_DIR / 'phase1'
SELECTION_FILE = EXPERIMENT_DIR / 'frozen_selection.json'
PHASE2_DIR = EXPERIMENT_DIR / 'phase2_final_unseen_test'

ARCHITECTURES = ['lstm', 'gru', 'cnn', 'lstm_cnn', 'xgboost']
NEURAL_ARCHITECTURES = ['lstm', 'gru', 'cnn', 'lstm_cnn']
TASKS = ['same_day', 'next_day']
FINAL_SEEDS = [42, 43, 44]
DEPLOYMENT_SEED = 42

TRAIN_HOUSES = list(range(1, 11))
VAL_HOUSES = [11]
TEST_HOUSES = [12, 13]

HORIZON = 24
STEP_MIN = 15
STEPS_PER_DAY = (24 * 60) // STEP_MIN  # 96
DAY_AHEAD_OFFSET_STEPS = STEPS_PER_DAY
DEPLOYMENT_UPDATE_HOURS = 2
DEPLOYMENT_CADENCE_STEPS = (DEPLOYMENT_UPDATE_HOURS * 60) // STEP_MIN  # 8
FUTURE_OFFSETS = np.arange(1, HORIZON + 1, dtype=np.int32)   # 1 ... 24
SAME_DAY_CADENCE_STEPS = 1                                    # every 15 min
MAPE_THRESHOLD_PU = 0.01
# Same schedule rule as Phase 1: offset 0 / 1 day; cadence 15 min / 2 h.
TARGET_OFFSET_STEPS_BY_TASK = {'same_day': 0, 'next_day': DAY_AHEAD_OFFSET_STEPS}
OPERATIONAL_CADENCE_STEPS_BY_TASK = {'same_day': SAME_DAY_CADENCE_STEPS, 'next_day': DEPLOYMENT_CADENCE_STEPS}

FUTURE_FEATURES = [
    'Temperature', 'Relative Humidity', 'GHI', 'DNI', 'DHI', 'Wind Speed',
    'Solar_Zenith_rad', 'Solar_Azimuth_rad'
]
STATIC_FEATURES = [
    'Gamma_Pmax_per_C',
    'Panel_Tilt_rad',
    'Panel_Azimuth_rad',
]
TARGET = 'PV_DC_Power_W'
RATED_COL = 'Array_Rated_Power_W'

# -----------------------------------------------------------------------------
# Test lock: Phase 2 runs only after the frozen House-11 selection.
# -----------------------------------------------------------------------------
if not SELECTION_FILE.exists():
    raise RuntimeError(
        f'{SELECTION_FILE} not found. Houses 12-13 stay locked until '
        'select_architecture.py has frozen the architecture decision.'
    )
with open(SELECTION_FILE) as f:
    selection = json.load(f)
SELECTED_ARCH = selection['selected_architecture']
if selection.get('final_seeds') != FINAL_SEEDS or selection.get('deployment_seed') != DEPLOYMENT_SEED:
    raise RuntimeError('frozen_selection.json seeds do not match the frozen protocol.')
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Loader, schedules and metrics — verbatim copies of the Phase-1 code
# (Phase-1 test-house guard removed from load_house: this is Phase 2).
# -----------------------------------------------------------------------------
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require_target_column(house_id, df):
    # Final experiment is intentionally strict:
    # old/ambiguous PV_Power_Generation_W files are NOT accepted.
    if TARGET not in df.columns:
        raise ValueError(
            f'House {house_id:02d}: missing required final DC target '
            f'{TARGET!r}. Do not silently fall back to an older target column.'
        )
    return TARGET


def load_house(h, path):
    df = pd.read_csv(path)
    target = require_target_column(h, df)
    required = ['Timestamp'] + FUTURE_FEATURES + STATIC_FEATURES + [RATED_COL, target]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f'House {h:02d}: missing columns {miss}')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='raise')
    if df['Timestamp'].duplicated().any() or not df['Timestamp'].is_monotonic_increasing:
        raise ValueError(f'House {h:02d}: bad timestamp ordering/duplicates.')
    if len(df) < DAY_AHEAD_OFFSET_STEPS + HORIZON + 1:
        raise ValueError(f'House {h:02d}: dataset is too short for the next-day 6-hour horizon.')
    dt = (df['Timestamp'].iloc[1:].to_numpy() - df['Timestamp'].iloc[:-1].to_numpy()) / np.timedelta64(1, 'm')
    if not np.all(dt == STEP_MIN):
        raise ValueError(f'House {h:02d}: not continuous 15-minute data.')
    numeric_cols = FUTURE_FEATURES + STATIC_FEATURES + [RATED_COL, target]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if df[numeric_cols].isna().any().any() or not np.isfinite(df[numeric_cols].to_numpy(float)).all():
        raise ValueError(f'House {h:02d}: NaN/Inf in required data.')
    if (df[target] < 0).any():
        raise ValueError(f'House {h:02d}: negative DC power.')
    rated_u = df[RATED_COL].unique()
    if len(rated_u) != 1 or rated_u[0] <= 0:
        raise ValueError(f'House {h:02d}: rated power must be one positive constant.')
    static = []
    for c in STATIC_FEATURES:
        u = df[c].unique()
        if len(u) != 1:
            raise ValueError(f'House {h:02d}: static feature {c} changes inside file.')
        static.append(float(u[0]))
    rated = float(rated_u[0])
    pv_w = df[target].to_numpy(np.float32)
    pv_pu = (pv_w / rated).astype(np.float32)
    zen = df['Solar_Zenith_rad'].to_numpy(np.float32)
    daylight = zen < (np.pi / 2)
    if np.any((~daylight) & (pv_w != 0)):
        raise ValueError(f'House {h:02d}: non-zero PV at night.')
    return {
        'timestamps': df['Timestamp'].to_numpy(),
        'rated': rated,
        'pv_w': pv_w,
        'pv_pu': pv_pu,
        'future_raw': df[FUTURE_FEATURES].to_numpy(np.float32),
        'static_raw': np.asarray(static, np.float32),
        'daylight': daylight,
    }


_OPERATIONAL_ORIGIN_CACHE = {}


def operational_origins_for_house(h, offset_steps, cadence_steps):
    """
    Exact operational issue indices (origins) for one house.

    For every target day:
      first target = first 15-minute daylight grid point of that day;
      first issue  = first target - offset_steps - 1 step;
      then one issue every cadence_steps while the window's FIRST target is
      still daylight on that same target day. The issue itself need not be daylight.
    Targets of an issue = origin + offset_steps + FUTURE_OFFSETS (1..24).

    Same-Day: offset_steps = 0,  cadence_steps = 1 (15 min).
      sunrise 07:08 -> first target 07:15 -> issue 07:00 -> 07:15 ... 13:00,
      issue 07:15 -> 07:30 ... 13:15, ...
    Next-Day: offset_steps = 96, cadence_steps = 8 (2 h).
      tomorrow sunrise 07:08 -> first target 07:15 -> issue today 07:00 ->
      tomorrow 07:15 ... 13:00, issue 09:00 -> 09:15 ... 15:00, ...
    """
    key = (int(h), int(offset_steps), int(cadence_steps))
    if key in _OPERATIONAL_ORIGIN_CACHE:
        return _OPERATIONAL_ORIGIN_CACHE[key]
    h, offset_steps, cadence_steps = key

    d = houses[h]
    ts = pd.DatetimeIndex(d['timestamps'])
    normalized_dates = ts.normalize().to_numpy(dtype='datetime64[ns]')

    # Timestamps are already strictly increasing, so each date is contiguous.
    unique_dates, day_starts, day_counts = np.unique(
        normalized_dates,
        return_index=True,
        return_counts=True,
    )

    first_target_delta = pd.Timedelta(minutes=(offset_steps + 1) * STEP_MIN)
    house_origins = []

    for target_date64, day_start, day_count in zip(unique_dates, day_starts, day_counts):
        day_idx = np.arange(
            int(day_start),
            int(day_start + day_count),
            dtype=np.int32,
        )
        daylight_idx = day_idx[d['daylight'][day_idx]]
        if len(daylight_idx) == 0:
            continue

        # First prediction = first 15-minute daylight timestamp of the target day.
        first_target_idx = int(daylight_idx[0])

        # first target = issue + offset + 15 min, therefore:
        # issue = first target - offset steps - 1 step.
        first_origin = first_target_idx - offset_steps - 1
        if first_origin < 0:
            # e.g. the first target day of the file for Next-Day: no issue inside the file.
            continue

        k = 0
        while True:
            origin = first_origin + k * cadence_steps
            first_pred_idx = origin + offset_steps + 1
            last_pred_idx = origin + offset_steps + HORIZON

            if last_pred_idx >= len(ts):
                break

            # Stop if the later slot moved into another target day.
            if normalized_dates[first_pred_idx] != target_date64:
                break

            # Stop when the first prediction is no longer daylight.
            if not bool(d['daylight'][first_pred_idx]):
                break

            issue_ts = ts[origin]
            first_target_ts = ts[first_pred_idx]

            # Hard timing checks.
            if first_target_ts - issue_ts != first_target_delta:
                raise RuntimeError(
                    f'House {h:02d}: first target is not issue + {first_target_delta} at {issue_ts}.'
                )
            if offset_steps == DAY_AHEAD_OFFSET_STEPS and ts[origin + offset_steps] - issue_ts != pd.Timedelta(hours=24):
                raise RuntimeError(
                    f'House {h:02d}: next-day reference is not exactly +24 h at {issue_ts}.'
                )

            # The first operational window of each target day MUST start
            # exactly at that day's first daylight 15-minute grid point.
            if k == 0 and first_pred_idx != first_target_idx:
                raise RuntimeError(
                    f'House {h:02d}: first target is not the first daylight grid.'
                )

            house_origins.append(origin)
            k += 1

    out = np.asarray(house_origins, dtype=np.int32)
    _OPERATIONAL_ORIGIN_CACHE[key] = out
    return out


def schedule_sha256(sample_house, sample_origin):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sample_house, dtype=np.int16).tobytes())
    h.update(np.ascontiguousarray(sample_origin, dtype=np.int32).tobytes())
    return h.hexdigest()


def point_metrics(actual_w, pred_w, actual_pu, pred_pu, mask):
    """
    Point-wise daylight metrics. Night points are excluded by `mask`,
    so the primary/secondary metrics use ALL daylight points.

    Primary      : RMSE (W), MAE (W), nRMSE (% of rated power), WAPE (%), R2
    Secondary    : sMAPE (%)
    Supplementary: MAPE_1pct (%) only where actual PV >= 1% of rated power.

    MAPE_1pct is NOT used for Optuna/model selection. The included and excluded
    daylight-point counts are reported so the threshold is transparent.
    """
    aw = actual_w[mask]
    pw = pred_w[mask]
    ap = actual_pu[mask]
    pp = pred_pu[mask]

    n_daylight = int(len(aw))

    if n_daylight == 0:
        return {
            'RMSE_W': np.nan,
            'MAE_W': np.nan,
            'nRMSE_percent': np.nan,
            'WAPE_percent': np.nan,
            'R2': np.nan,
            'sMAPE_percent': np.nan,
            'MAPE_1pct_percent': np.nan,
            'N_MAPE_1pct_included': 0,
            'N_MAPE_1pct_excluded': 0,
            'N_daylight_points': 0,
        }

    # ----- Primary metrics -----
    rmse = float(np.sqrt(mean_squared_error(aw, pw)))
    mae = float(mean_absolute_error(aw, pw))

    # nRMSE = per-unit RMSE x 100, i.e. RMSE normalized by each house's
    # rated array power (PV / Array_Rated_Power_W).
    nrmse = float(np.sqrt(np.mean((pp - ap) ** 2)) * 100.0)

    # WAPE = total absolute error / total actual PV over daylight points.
    actual_sum = float(np.sum(np.abs(aw)))
    wape = (
        float(100.0 * np.sum(np.abs(pw - aw)) / actual_sum)
        if actual_sum > 1e-9
        else np.nan
    )

    r2 = (
        float(r2_score(aw, pw))
        if len(aw) > 1 and np.std(aw) > 0
        else np.nan
    )

    # ----- Secondary metric -----
    # sMAPE skips points where actual and predicted are both effectively zero.
    smape_den = np.abs(aw) + np.abs(pw)
    smape_valid = smape_den > 1e-9
    smape = (
        float(
            100.0
            * np.mean(
                2.0
                * np.abs(pw[smape_valid] - aw[smape_valid])
                / smape_den[smape_valid]
            )
        )
        if np.any(smape_valid)
        else np.nan
    )

    # ----- Supplementary MAPE_1pct -----
    # Because ap = actual_w / rated_power, ap >= 0.01 is exactly:
    # actual PV >= 1% of that house's Array_Rated_Power_W.
    mape_valid = ap >= MAPE_THRESHOLD_PU
    n_mape_included = int(np.sum(mape_valid))
    n_mape_excluded = int(n_daylight - n_mape_included)
    mape_1pct = (
        float(
            100.0
            * np.mean(
                np.abs(pp[mape_valid] - ap[mape_valid])
                / ap[mape_valid]
            )
        )
        if n_mape_included > 0
        else np.nan
    )

    return {
        'RMSE_W': rmse,
        'MAE_W': mae,
        'nRMSE_percent': nrmse,
        'WAPE_percent': wape,
        'R2': r2,
        'sMAPE_percent': smape,
        'MAPE_1pct_percent': mape_1pct,
        'N_MAPE_1pct_included': n_mape_included,
        'N_MAPE_1pct_excluded': n_mape_excluded,
        'N_daylight_points': n_daylight,
    }


def planning_metrics(actual_w, pred_w, mask):
    """
    Per-window HEMS planning metrics for the 6-hour forecast.

    Each sample is a 24-step, 15-minute window. Night points are excluded
    structurally by mask and predictions are already forced to zero there.
    """
    if len(actual_w) == 0:
        return {
            'Energy_MAE_kWh': np.nan,
            'Energy_RMSE_kWh': np.nan,
            'Energy_Bias_kWh': np.nan,
            'Energy_WAPE_percent': np.nan,
            'Peak_Power_MAE_W': np.nan,
            'Peak_Power_Bias_W': np.nan,
            'Peak_Time_MAE_min': np.nan,
            'Peak_Time_Bias_min': np.nan,
            'N_windows': 0,
        }, pd.DataFrame()

    actual_masked = np.where(mask, actual_w, 0.0)
    pred_masked = np.where(mask, pred_w, 0.0)

    step_hours = STEP_MIN / 60.0
    actual_energy_kwh = (
        np.sum(actual_masked, axis=1) * step_hours / 1000.0
    )
    pred_energy_kwh = (
        np.sum(pred_masked, axis=1) * step_hours / 1000.0
    )
    energy_error_kwh = pred_energy_kwh - actual_energy_kwh

    energy_mae = float(np.mean(np.abs(energy_error_kwh)))
    energy_rmse = float(np.sqrt(np.mean(energy_error_kwh ** 2)))
    energy_bias = float(np.mean(energy_error_kwh))

    total_actual_energy = float(np.sum(np.abs(actual_energy_kwh)))
    energy_wape = (
        float(
            100.0
            * np.sum(np.abs(energy_error_kwh))
            / total_actual_energy
        )
        if total_actual_energy > 1e-12
        else np.nan
    )

    # Every retained sample has at least one daylight target.
    actual_for_peak = np.where(mask, actual_w, -np.inf)
    pred_for_peak = np.where(mask, pred_w, -np.inf)

    actual_peak_idx = np.argmax(actual_for_peak, axis=1)
    pred_peak_idx = np.argmax(pred_for_peak, axis=1)

    rows = np.arange(len(actual_w))
    actual_peak_w = actual_w[rows, actual_peak_idx]
    pred_peak_w = pred_w[rows, pred_peak_idx]

    peak_power_error_w = pred_peak_w - actual_peak_w
    peak_time_error_min = (
        pred_peak_idx.astype(np.int32)
        - actual_peak_idx.astype(np.int32)
    ) * STEP_MIN

    summary = {
        'Energy_MAE_kWh': energy_mae,
        'Energy_RMSE_kWh': energy_rmse,
        'Energy_Bias_kWh': energy_bias,
        'Energy_WAPE_percent': energy_wape,
        'Peak_Power_MAE_W': float(
            np.mean(np.abs(peak_power_error_w))
        ),
        'Peak_Power_Bias_W': float(
            np.mean(peak_power_error_w)
        ),
        'Peak_Time_MAE_min': float(
            np.mean(np.abs(peak_time_error_min))
        ),
        'Peak_Time_Bias_min': float(
            np.mean(peak_time_error_min)
        ),
        'N_windows': int(len(actual_w)),
    }

    window_df = pd.DataFrame({
        'Actual_Energy_kWh': actual_energy_kwh,
        'Predicted_Energy_kWh': pred_energy_kwh,
        'Energy_Error_kWh': energy_error_kwh,
        'Actual_Peak_W': actual_peak_w,
        'Predicted_Peak_W': pred_peak_w,
        'Peak_Power_Error_W': peak_power_error_w,
        'Actual_Peak_Step': actual_peak_idx + 1,
        'Predicted_Peak_Step': pred_peak_idx + 1,
        'Peak_Time_Error_min': peak_time_error_min,
        'Daylight_Points': np.sum(mask, axis=1).astype(int),
    })

    return summary, window_df


# -----------------------------------------------------------------------------
# Frozen Phase-1 artefacts must be exactly those the selection was made on.
# -----------------------------------------------------------------------------
summaries = {}
for arch in ARCHITECTURES:
    for task in TASKS:
        p = PHASE1_DIR / f'{arch}_{task}' / 'phase1_summary.json'
        if file_sha256(p) != selection['phase1_summary_sha256'][f'{arch}_{task}']:
            raise RuntimeError(f'{p} changed after the frozen selection.')
        with open(p) as f:
            summaries[(arch, task)] = json.load(f)
print(f'Frozen selection verified. Selected architecture (House 11): {SELECTED_ARCH}')

# Scalers: fitted on Houses 1-10 in Phase 1; identical in every run; never refitted.
scaler_runs = {}
for arch in ARCHITECTURES:
    for task in TASKS:
        rd = PHASE1_DIR / f'{arch}_{task}'
        scaler_runs[(arch, task)] = (joblib.load(rd / 'future_weather_scaler.joblib'),
                                     joblib.load(rd / 'static_scaler.joblib'))
future_scaler, static_scaler = scaler_runs[(ARCHITECTURES[0], TASKS[0])]
for key, (fs, ss) in scaler_runs.items():
    for a, b in ((fs, future_scaler), (ss, static_scaler)):
        if not (np.allclose(a.mean_, b.mean_, rtol=1e-6, atol=1e-9)
                and np.allclose(a.scale_, b.scale_, rtol=1e-6, atol=1e-9)):
            raise RuntimeError(f'Scaler of {key} differs from the other Phase-1 runs.')
print('Phase-1 scalers (Houses 1-10) are identical across all ten runs and are re-used.')

HOUSE_FILES = {h: DATA_DIR / f'house_{h:02d}_model_dataset.csv' for h in VAL_HOUSES + TEST_HOUSES}
houses = {}
house_file_sha256 = {}


def load_and_transform(h):
    houses[h] = load_house(h, HOUSE_FILES[h])
    house_file_sha256[h] = file_sha256(HOUSE_FILES[h])
    houses[h]['future'] = future_scaler.transform(houses[h]['future_raw']).astype(np.float32)
    houses[h]['static'] = static_scaler.transform(
        houses[h]['static_raw'].reshape(1, -1)
    )[0].astype(np.float32)
    print(f"House {h:02d}: {len(houses[h]['pv_w']):,} rows, rated={houses[h]['rated']:.1f} W")


# House 11 (consistency checks only — its metrics were already used for selection).
load_and_transform(11)
if house_file_sha256[11] != selection['house_file_sha256_houses_01_11']['11']:
    raise RuntimeError('House 11 data file changed since Phase 1.')

# FIRST ACCESS to the unseen test houses.
missing_test = [str(HOUSE_FILES[h]) for h in TEST_HOUSES if not HOUSE_FILES[h].exists()]
if missing_test:
    raise FileNotFoundError('Missing test files:\n' + '\n'.join(missing_test))
print('\nFIRST ACCESS to unseen test Houses 12-13 (after the frozen selection).')
for h in TEST_HOUSES:
    load_and_transform(h)


def build_task_index(task, house_ids):
    """Operational schedule of `task` (Phase-1 rule, full operational cadence)."""
    hs, origins_all = [], []
    for h in house_ids:
        origins = operational_origins_for_house(
            int(h), TARGET_OFFSET_STEPS_BY_TASK[task], OPERATIONAL_CADENCE_STEPS_BY_TASK[task])
        if len(origins) == 0:
            continue
        hs.append(np.full(len(origins), int(h), dtype=np.int16))
        origins_all.append(origins)
    if not origins_all:
        return np.empty(0, np.int16), np.empty(0, np.int32)
    return np.concatenate(hs), np.concatenate(origins_all)


def build_arrays(task, house_ids):
    """Model inputs exactly as in Phase 1: (24, 8) weather + (3,) static; XGB = time-major flatten."""
    offset = TARGET_OFFSET_STEPS_BY_TASK[task]
    sample_house, sample_origin = build_task_index(task, house_ids)
    n = len(sample_origin)
    future = np.empty((n, HORIZON, len(FUTURE_FEATURES)), np.float32)
    static = np.empty((n, len(STATIC_FEATURES)), np.float32)
    y_pu = np.empty((n, HORIZON), np.float32)
    mask = np.empty((n, HORIZON), bool)
    rated = np.empty(n, np.float32)
    origin_ts = np.empty(n, dtype='datetime64[ns]')
    target_start_ts = np.empty(n, dtype='datetime64[ns]')
    offsets = FUTURE_OFFSETS
    for h in np.unique(sample_house):
        pos = np.flatnonzero(sample_house == h)
        origins = sample_origin[pos]
        d = houses[int(h)]
        fidx = origins[:, None] + offset + offsets[None, :]
        future[pos] = d['future'][fidx]
        static[pos] = d['static']
        y_pu[pos] = d['pv_pu'][fidx]
        mask[pos] = d['daylight'][fidx]
        rated[pos] = d['rated']
        origin_ts[pos] = d['timestamps'][origins]
        target_start_ts[pos] = d['timestamps'][origins + offset + 1]
    return {
        'future': future, 'static': static,
        'X': np.concatenate([future.reshape(n, -1), static], axis=1),
        'y_pu': y_pu, 'mask': mask, 'rated': rated,
        'house': sample_house, 'origin': sample_origin,
        'origin_ts': origin_ts, 'target_start_ts': target_start_ts,
    }


def schedule_audit(task, house_ids):
    data = build_arrays(task, house_ids)
    first_daylight = data['mask'][:, 0]
    return pd.DataFrame({
        'House': data['house'].astype(int),
        'Issue_Timestamp': pd.to_datetime(data['origin_ts']),
        'First_Target_Timestamp': pd.to_datetime(data['target_start_ts']),
        'Last_Target_Timestamp': pd.to_datetime(data['target_start_ts'])
        + pd.to_timedelta((HORIZON - 1) * STEP_MIN, unit='m'),
        'First_Target_Daylight': first_daylight,
        'Daylight_Target_Count': data['mask'].sum(axis=1).astype(int),
    })


val_data, test_data = {}, {}
for task in TASKS:
    val_data[task] = build_arrays(task, VAL_HOUSES)
    expected = summaries[(ARCHITECTURES[0], task)]['house11_validation_schedule_sha256']
    if schedule_sha256(val_data[task]['house'], val_data[task]['origin']) != expected:
        raise RuntimeError(f'{task}: House-11 schedule rebuilt in Phase 2 differs from Phase 1.')
    test_data[task] = build_arrays(task, TEST_HOUSES)
    schedule_audit(task, TEST_HOUSES).to_csv(
        PHASE2_DIR / f'operational_schedule_audit_houses_12_13_{task}.csv.gz', index=False)
    print(f"{task}: House-11 schedule matches Phase 1 ({len(val_data[task]['origin']):,} windows); "
          f"test windows (Houses 12-13): {len(test_data[task]['origin']):,}")


def minutes_ahead(task, j):
    return TARGET_OFFSET_STEPS_BY_TASK[task] * STEP_MIN + (j + 1) * STEP_MIN


def evaluate_predictions(task, data, raw_pred_pu):
    # Physical post-processing for reporting: no negative PV, exactly zero at night.
    pred_pu = np.maximum(raw_pred_pu.astype(np.float32), 0.0)
    pred_pu[~data['mask']] = 0.0
    actual_pu = data['y_pu']
    mask = data['mask']
    rated = data['rated']
    actual_w = actual_pu * rated[:, None]
    pred_w = pred_pu * rated[:, None]

    overall = point_metrics(actual_w, pred_w, actual_pu, pred_pu, mask)
    planning_overall, window_df = planning_metrics(actual_w, pred_w, mask)
    window_df.insert(0, 'House', data['house'].astype(int))
    window_df.insert(1, 'Issue_Timestamp', pd.to_datetime(data['origin_ts']))
    window_df.insert(2, 'Target_Start_Timestamp', pd.to_datetime(data['target_start_ts']))

    by_h = []
    for j in range(HORIZON):
        row = point_metrics(actual_w[:, j:j+1], pred_w[:, j:j+1],
                            actual_pu[:, j:j+1], pred_pu[:, j:j+1], mask[:, j:j+1])
        row['Horizon_step'] = j + 1
        row['Minutes_ahead'] = minutes_ahead(task, j)
        by_h.append(row)

    by_house, planning_by_house = [], []
    for h in np.unique(data['house']):
        s = data['house'] == h
        row = point_metrics(actual_w[s], pred_w[s], actual_pu[s], pred_pu[s], mask[s])
        row['House'] = int(h)
        by_house.append(row)
        prow, _ = planning_metrics(actual_w[s], pred_w[s], mask[s])
        prow['House'] = int(h)
        planning_by_house.append(prow)

    return {
        'overall': overall, 'planning_overall': planning_overall,
        'by_horizon': pd.DataFrame(by_h), 'by_house': pd.DataFrame(by_house),
        'planning_by_house': pd.DataFrame(planning_by_house),
        'window_metrics': window_df, 'pred_w': pred_w, 'actual_w': actual_w,
    }


def load_frozen_predictor(arch, task, seed):
    run_dir = PHASE1_DIR / f'{arch}_{task}'
    info = summaries[(arch, task)]['per_seed'][str(seed)]
    if arch in NEURAL_ARCHITECTURES:
        model = keras.models.load_model(run_dir / info['model_file'], compile=False)

        def predict(data):
            return model.predict({'future_weather': data['future'], 'static': data['static']},
                                 batch_size=2048, verbose=0).astype(np.float32)
        return predict
    models = []
    for name in info['model_files']:
        m = XGBRegressor()
        m.load_model(run_dir / info['model_dir'] / name)
        models.append(m)
    if len(models) != HORIZON:
        raise RuntimeError(f'{arch}_{task} seed {seed}: expected {HORIZON} regressors.')

    def predict(data):
        return np.stack([m.predict(data['X']) for m in models], axis=1).astype(np.float32)
    return predict


# -----------------------------------------------------------------------------
# Evaluate EVERY frozen model on Houses 12-13.
# -----------------------------------------------------------------------------
point_rows, planning_rows, house_rows, house_planning_rows, horizon_rows = [], [], [], [], []
example_curves = {}
for arch in ARCHITECTURES:
    for task in TASKS:
        for seed in FINAL_SEEDS:
            tag = dict(Architecture=arch, Task=task, Seed=seed)
            predict = load_frozen_predictor(arch, task, seed)

            # Consistency: this pipeline must reproduce Phase-1 House-11 predictions.
            saved = np.load(PHASE1_DIR / f'{arch}_{task}' / 'models' / f'seed_{seed}' / 'house11_predictions.npz')
            if not (np.array_equal(saved['origin'], val_data[task]['origin'])
                    and np.array_equal(saved['house'], val_data[task]['house'])):
                raise RuntimeError(f'{arch}_{task} seed {seed}: House-11 windows differ from Phase 1.')
            max_diff = float(np.max(np.abs(predict(val_data[task]) - saved['raw_pred_pu'])))
            if max_diff > 1e-4:
                raise RuntimeError(f'{arch}_{task} seed {seed}: Phase-2 pipeline does not reproduce '
                                   f'Phase-1 House-11 predictions (max |diff| = {max_diff:.2e} pu).')

            res = evaluate_predictions(task, test_data[task], predict(test_data[task]))
            point_rows.append({**tag, 'Split': 'Test_Houses_12_13', **res['overall']})
            planning_rows.append({**tag, 'Split': 'Test_Houses_12_13', **res['planning_overall']})
            house_rows += [{**tag, **r} for r in res['by_house'].to_dict(orient='records')]
            house_planning_rows += [{**tag, **r} for r in res['planning_by_house'].to_dict(orient='records')]
            horizon_rows += [{**tag, **r} for r in res['by_horizon'].to_dict(orient='records')]
            out_dir = PHASE2_DIR / f'{arch}_{task}' / f'seed_{seed}'
            out_dir.mkdir(parents=True, exist_ok=True)
            res['window_metrics'].to_csv(out_dir / 'test_window_planning_metrics.csv.gz', index=False)
            if arch == SELECTED_ARCH and seed == DEPLOYMENT_SEED:
                example_curves[task] = res
            print(f"{arch:9s} {task:9s} seed {seed}: House-11 reproduction max |diff| = {max_diff:.1e} pu | "
                  f"test nRMSE = {res['overall']['nRMSE_percent']:.4f} %")
            del predict
            gc.collect()

point_df = pd.DataFrame(point_rows)
planning_df = pd.DataFrame(planning_rows)
house_df = pd.DataFrame(house_rows)
house_planning_df = pd.DataFrame(house_planning_rows)
horizon_df = pd.DataFrame(horizon_rows)
point_df.to_csv(PHASE2_DIR / 'test_point_metrics_per_seed.csv', index=False)
planning_df.to_csv(PHASE2_DIR / 'test_planning_metrics_per_seed.csv', index=False)
house_df.to_csv(PHASE2_DIR / 'test_point_metrics_by_house_per_seed.csv', index=False)
house_planning_df.to_csv(PHASE2_DIR / 'test_planning_metrics_by_house_per_seed.csv', index=False)
horizon_df.to_csv(PHASE2_DIR / 'test_point_metrics_by_horizon_per_seed.csv', index=False)


def mean_sd(df, keys):
    num_cols = [c for c in df.columns if c not in keys + ['Seed', 'Split']]
    g = df.groupby(keys, sort=False)[num_cols]
    out = g.mean().add_suffix('_mean').join(g.std(ddof=1).add_suffix('_sd'))
    return out.reset_index()


# Mean ± SD over seeds 42-44, always per task (MAPE_1pct is never pooled across tasks).
point_ms = mean_sd(point_df, ['Architecture', 'Task'])
planning_ms = mean_sd(planning_df, ['Architecture', 'Task'])
house_ms = mean_sd(house_df, ['Architecture', 'Task', 'House'])
house_planning_ms = mean_sd(house_planning_df, ['Architecture', 'Task', 'House'])
horizon_ms = mean_sd(horizon_df, ['Architecture', 'Task', 'Horizon_step'])
point_ms.to_csv(PHASE2_DIR / 'test_point_metrics_mean_sd.csv', index=False)
planning_ms.to_csv(PHASE2_DIR / 'test_planning_metrics_mean_sd.csv', index=False)
house_ms.to_csv(PHASE2_DIR / 'test_point_metrics_by_house_mean_sd.csv', index=False)
house_planning_ms.to_csv(PHASE2_DIR / 'test_planning_metrics_by_house_mean_sd.csv', index=False)
horizon_ms.to_csv(PHASE2_DIR / 'test_point_metrics_by_horizon_mean_sd.csv', index=False)

# Combined test score (REPORTED ONLY — the selection was frozen on House 11).
wide = point_df.pivot_table(index=['Architecture', 'Seed'], columns='Task', values='nRMSE_percent').reset_index()
wide['CombinedScore_test'] = 0.5 * wide['same_day'] + 0.5 * wide['next_day']
house11 = pd.DataFrame(selection['ranking']).set_index('Architecture')
final_table = wide.groupby('Architecture', sort=False).agg(
    SameDay_test_nRMSE_mean=('same_day', 'mean'), SameDay_test_nRMSE_sd=('same_day', lambda x: x.std(ddof=1)),
    NextDay_test_nRMSE_mean=('next_day', 'mean'), NextDay_test_nRMSE_sd=('next_day', lambda x: x.std(ddof=1)),
    CombinedScore_test_mean=('CombinedScore_test', 'mean'),
    CombinedScore_test_sd=('CombinedScore_test', lambda x: x.std(ddof=1)),
).reset_index()
final_table['House11_CombinedScore_mean'] = final_table['Architecture'].map(house11['CombinedScore_mean'])
final_table['House11_Rank'] = final_table['Architecture'].map(house11['Rank'])
final_table['Selected_on_House11'] = final_table['Architecture'] == SELECTED_ARCH
final_table = final_table.sort_values('House11_Rank').reset_index(drop=True)
final_table.to_csv(PHASE2_DIR / 'final_architecture_comparison.csv', index=False)

print('\nFINAL UNSEEN TEST (Houses 12-13) — all frozen architectures; selection fixed on House 11')
display(final_table)
print('\nTEST POINT METRICS (mean over seeds) — per task')
display(point_ms[['Architecture', 'Task', 'RMSE_W_mean', 'MAE_W_mean', 'nRMSE_percent_mean',
                  'WAPE_percent_mean', 'R2_mean', 'sMAPE_percent_mean', 'MAPE_1pct_percent_mean',
                  'N_MAPE_1pct_included_mean', 'N_MAPE_1pct_excluded_mean']])

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
for task in TASKS:
    plt.figure(figsize=(10, 5))
    for arch in ARCHITECTURES:
        s = horizon_ms[(horizon_ms['Architecture'] == arch) & (horizon_ms['Task'] == task)]
        plt.plot(s['Minutes_ahead_mean'], s['nRMSE_percent_mean'], marker='o',
                 label=arch + (' (selected)' if arch == SELECTED_ARCH else ''))
    plt.xlabel('Lead time from issue (minutes)'); plt.ylabel('nRMSE (% of rated power)')
    plt.title(f'Unseen Houses 12-13 — {task} — nRMSE by horizon (mean over seeds 42-44)')
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(PHASE2_DIR / f'test_nrmse_by_horizon_{task}.png', dpi=160); plt.show()

    res = example_curves[task]
    m = test_data[task]['mask']
    a, p = res['actual_w'][m], res['pred_w'][m]
    rng = np.random.default_rng(DEPLOYMENT_SEED)
    if len(a) > 100_000:
        idx = rng.choice(len(a), size=100_000, replace=False)
        a, p = a[idx], p[idx]
    plt.figure(figsize=(7, 7)); plt.scatter(a, p, s=5, alpha=.25)
    top = max(float(np.max(a)), float(np.max(p)))
    plt.plot([0, top], [0, top], '--')
    plt.xlabel('Actual DC PV (W)'); plt.ylabel('Predicted DC PV (W)')
    plt.title(f'{SELECTED_ARCH} (seed {DEPLOYMENT_SEED}) — {task} — Unseen Houses 12-13')
    plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(PHASE2_DIR / f'test_actual_vs_predicted_selected_{task}.png', dpi=160); plt.show()
plt.close('all')

# -----------------------------------------------------------------------------
# Deployment manifest: selected architecture, predefined seed 42, both tasks.
# -----------------------------------------------------------------------------
deployment = {
    'selected_architecture': SELECTED_ARCH,
    'deployment_seed': DEPLOYMENT_SEED,
    'scalers': {
        'future_weather': str(PHASE1_DIR / f'{SELECTED_ARCH}_same_day' / 'future_weather_scaler.joblib'),
        'static': str(PHASE1_DIR / f'{SELECTED_ARCH}_same_day' / 'static_scaler.joblib'),
        'fitted_on_houses': TRAIN_HOUSES,
    },
    'models': {},
    'future_features': FUTURE_FEATURES, 'static_features': STATIC_FEATURES,
    'grid': '15-minute grid; daylight = Solar_Zenith_rad < pi/2; predictions clipped >= 0 and 0 at night',
    'same_day_schedule': 'first target = first daylight 15-min grid point of today; first issue = that - 15 min; then every 15 min while the first target is daylight; targets t+15 ... t+360',
    'next_day_schedule': 'first target = first daylight 15-min grid point of tomorrow; issue = that - 24 h - 15 min; then every 2 h while the first target is daylight; targets t+24h15m ... t+30h',
    'zenith_convention_note': 'Before Raspberry Pi deployment verify timestamp semantics, time zone, true vs apparent zenith and instantaneous vs interval representation against the training datasets.',
}
for task in TASKS:
    info = summaries[(SELECTED_ARCH, task)]['per_seed'][str(DEPLOYMENT_SEED)]
    rd = PHASE1_DIR / f'{SELECTED_ARCH}_{task}'
    deployment['models'][task] = (
        str(rd / info['model_file']) if SELECTED_ARCH in NEURAL_ARCHITECTURES
        else [str(rd / info['model_dir'] / n) for n in info['model_files']]
    )
with open(PHASE2_DIR / 'deployment_manifest.json', 'w') as f:
    json.dump(deployment, f, indent=2)

report = {
    'phase': 2,
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'selected_architecture_house11': SELECTED_ARCH,
    'selection_unchanged_by_test_results': True,
    'test_houses': TEST_HOUSES,
    'test_house_file_sha256': {str(h): house_file_sha256[h] for h in TEST_HOUSES},
    'models_evaluated': [f'{a}_{t}_seed{s}' for a in ARCHITECTURES for t in TASKS for s in FINAL_SEEDS],
    'no_changes': 'no training, tuning, refitting, hyperparameter, seed or schedule changes in Phase 2',
    'combined_test_score_note': 'CombinedScore_test is reported only; the decision is the House-11 selection',
    'mape_1pct_note': 'supplementary; reported per task, never pooled across tasks; not used for any decision',
}
with open(PHASE2_DIR / 'phase2_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('\nPHASE 2 DONE. Outputs:', PHASE2_DIR)
