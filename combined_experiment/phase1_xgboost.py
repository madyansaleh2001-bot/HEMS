# =============================================================================
# COMBINED SAME-DAY + NEXT-DAY EXPERIMENT (Experiment A) — PHASE 1 — XGBOOST
# Google Colab / XGBoost / Optuna — T4 GPU enabled when available — NO PV LOOKBACK
#
# Run this file TWICE, once per task:  TASK = 'same_day'  and  TASK = 'next_day'.
# Each run is an independent experiment with its OWN Optuna study and its own
# 24 horizon-specific regressors (24 same-day + 24 next-day = 48 in total).
# Hyperparameters are never shared between the two tasks.
#
# PHASE 1 ISOLATION (corrected XGBoost reference):
#   Only Houses 1-11 are opened. Houses 12-13 are not read, validated,
#   transformed, scheduled or evaluated here. Their file paths are not even built.
#
# SAME-DAY schedule (Same-Day LSTM reference): origin_is_daylight & future_has_daylight,
#   targets origin+1..origin+24; Optuna on TUNING_STRIDE = 4 origins (exactly as the
#   neural same-day runs), final fit and House-11 evaluation on FINAL_STRIDE = 1.
# NEXT-DAY schedule (corrected XGBoost reference): first target = first 15-min daylight
#   grid point of the target day, issue = first target - 96 - 1 steps, then every 8 steps
#   (2 h) while the first target is daylight; same schedule for Optuna, fit and validation.
#
# Features: 24 x 8 target-time weather (flattened time-major) + 3 static = 195.
# Optuna: House-11 daylight-masked MSE of the RAW per-unit predictions, pooled over every
#   daylight point of ALL 24 horizons (same quantity as the neural val_loss). No early
#   stopping; n_estimators is tuned, as in the reference. Existing search ranges kept.
# Night targets excluded from each horizon's training rows.
# Reporting: predictions clipped >= 0 and exactly 0 at night.
# Final frozen configuration retrained with seeds 42, 43, 44 (mean ± SD reported);
# deployment seed 42.
# Metrics: Primary RMSE, MAE, nRMSE, WAPE, R2 | Secondary sMAPE |
#          Supplementary MAPE_1pct (+ N included / N excluded) | HEMS planning metrics.
# =============================================================================

import os, gc, json, math, random, shutil, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import optuna
import xgboost as xgb
from xgboost import XGBRegressor

# -----------------------------------------------------------------------------
# RUN SETTING — change and run again for the second task.
# -----------------------------------------------------------------------------
TASK = 'same_day'              # 'same_day' or 'next_day'
OVERWRITE_PHASE1_RUN = False   # True only to deliberately redo an unfinished/invalid run

# >>> ARCHITECTURE IDENTITY (architecture-specific) >>>
ARCH_KEY = 'xgboost'
ARCH_LABEL = 'XGBoost'
# <<< ARCHITECTURE IDENTITY <<<

SEED = 42                       # Optuna sampler + tuning fits (as in the reference)
FINAL_SEEDS = [42, 43, 44]      # frozen final seeds
DEPLOYMENT_SEED = 42            # frozen deployment seed
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

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

TASKS = ('same_day', 'next_day')
if TASK not in TASKS:
    raise ValueError(f'TASK must be one of {TASKS}, got {TASK!r}.')

EXPERIMENT_DIR = DATA_DIR / 'combined_sameday_nextday_experiment'
PHASE1_DIR = EXPERIMENT_DIR / 'phase1'
SELECTION_FILE = EXPERIMENT_DIR / 'frozen_selection.json'
RUN_DIR = PHASE1_DIR / f'{ARCH_KEY}_{TASK}'
MODEL_DIR = RUN_DIR / 'models'

# Phase-1 lock: once the architecture decision is frozen, Phase 1 may not be re-run.
if SELECTION_FILE.exists():
    raise RuntimeError(
        f'{SELECTION_FILE} exists: the architecture decision is frozen. '
        'Phase 1 must not be re-run after selection.'
    )
if (RUN_DIR / 'phase1_summary.json').exists() and not OVERWRITE_PHASE1_RUN:
    raise RuntimeError(
        f'{RUN_DIR} already holds a completed Phase-1 run. '
        'Set OVERWRITE_PHASE1_RUN = True only if you deliberately redo it.'
    )
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_HOUSES = list(range(1, 11))
VAL_HOUSES = [11]
TEST_HOUSES = [12, 13]   # listed for documentation only; never opened in Phase 1
PRETEST_HOUSES = TRAIN_HOUSES + VAL_HOUSES

HORIZON = 24
STEP_MIN = 15
STEPS_PER_DAY = (24 * 60) // STEP_MIN  # 96
DAY_AHEAD_OFFSET_STEPS = STEPS_PER_DAY
DEPLOYMENT_UPDATE_HOURS = 2
DEPLOYMENT_CADENCE_STEPS = (DEPLOYMENT_UPDATE_HOURS * 60) // STEP_MIN  # 8
TUNING_STRIDE = 4   # same-day only: every hour during Optuna
FINAL_STRIDE = 1    # same-day only: every 15 min for final fit/evaluation
OPTUNA_TRIALS = 20
MAPE_THRESHOLD_PU = 0.01  # supplementary MAPE only when actual PV >= 1% of rated power

# Task timing: same-day targets start at origin + 1; next-day at origin + 96 + 1.
TARGET_OFFSET_STEPS = 0 if TASK == 'same_day' else DAY_AHEAD_OFFSET_STEPS
# Same-day: hourly origins for Optuna, every 15 min afterwards (reference workflow).
# Next-day: the exact operational schedule everywhere (no stride).
TUNE_STRIDE_FOR_TASK = TUNING_STRIDE if TASK == 'same_day' else None
FINAL_STRIDE_FOR_TASK = FINAL_STRIDE if TASK == 'same_day' else None

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

# Strict isolation: Phase 1 builds paths ONLY for Houses 1-11.
HOUSE_FILES = {h: DATA_DIR / f'house_{h:02d}_model_dataset.csv' for h in PRETEST_HOUSES}
missing_pretest = [str(p) for p in HOUSE_FILES.values() if not p.exists()]
if missing_pretest:
    raise FileNotFoundError('Missing train/validation files:\n' + '\n'.join(missing_pretest))
print(f'{ARCH_LABEL} | TASK = {TASK} | train/validation files found. '
      'Houses 12-13 are not accessed in Phase 1.')


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
    if h not in PRETEST_HOUSES:
        raise RuntimeError(f'Phase 1 must never read House {h:02d}.')
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


houses = {}
house_file_sha256 = {}
# Phase 1: load ONLY Houses 1-10 (train) and House 11 (validation).
for h in PRETEST_HOUSES:
    houses[h] = load_house(h, HOUSE_FILES[h])
    house_file_sha256[h] = file_sha256(HOUSE_FILES[h])
    print(f"House {h:02d}: {len(houses[h]['pv_w']):,} rows, rated={houses[h]['rated']:.1f} W")

# Fit scalers ONLY on training Houses 1-10 (same definition for both tasks).
future_scaler = StandardScaler()
for h in TRAIN_HOUSES:
    future_scaler.partial_fit(houses[h]['future_raw'])
static_scaler = StandardScaler().fit(
    np.vstack([houses[h]['static_raw'] for h in TRAIN_HOUSES])
)
for h in PRETEST_HOUSES:
    houses[h]['future'] = future_scaler.transform(houses[h]['future_raw']).astype(np.float32)
    houses[h]['static'] = static_scaler.transform(
        houses[h]['static_raw'].reshape(1, -1)
    )[0].astype(np.float32)
joblib.dump(future_scaler, RUN_DIR / 'future_weather_scaler.joblib')
joblib.dump(static_scaler, RUN_DIR / 'static_scaler.joblib')


# -----------------------------------------------------------------------------
# SAME-DAY index — exact copy of the Same-Day LSTM reference.
# -----------------------------------------------------------------------------
def build_index(house_ids, stride):
    hs, origins_all = [], []
    future_offsets = np.arange(1, HORIZON + 1, dtype=np.int32)
    for h in house_ids:
        d = houses[h]
        # Same-day forecast origins. Final stride=1 means every 15 minutes.
        origins = np.arange(0, len(d['pv_pu']) - HORIZON, stride, dtype=np.int32)
        future_idx = origins[:, None] + future_offsets[None, :]

        # Keep the original same-day sunrise/sunset rule:
        # the issue time itself must be daylight, and the future 6-hour
        # window must contain at least one daylight target.
        # Example: sunrise 07:08 -> 07:00 excluded, 07:15 included.
        origin_is_daylight = d['daylight'][origins]
        future_has_daylight = np.any(d['daylight'][future_idx], axis=1)
        origins = origins[origin_is_daylight & future_has_daylight]

        hs.append(np.full(len(origins), h, np.int16))
        origins_all.append(origins)

    if not origins_all:
        return np.empty(0, np.int16), np.empty(0, np.int32)
    return np.concatenate(hs), np.concatenate(origins_all)


# -----------------------------------------------------------------------------
# NEXT-DAY index — exact copy of the corrected XGBoost / LSTM-CNN references.
# -----------------------------------------------------------------------------
_OPERATIONAL_ORIGIN_CACHE = {}


def operational_origins_for_house(h):
    """Return cached exact operational origins for one house."""
    h = int(h)
    if h in _OPERATIONAL_ORIGIN_CACHE:
        return _OPERATIONAL_ORIGIN_CACHE[h]

    d = houses[h]
    ts = pd.DatetimeIndex(d['timestamps'])
    normalized_dates = ts.normalize().to_numpy(dtype='datetime64[ns]')

    # Timestamps are already strictly increasing, so each date is contiguous.
    unique_dates, day_starts, day_counts = np.unique(
        normalized_dates,
        return_index=True,
        return_counts=True,
    )

    house_origins = []

    # The first target date has no previous-day issue inside the file.
    for target_date64, day_start, day_count in zip(
        unique_dates[1:], day_starts[1:], day_counts[1:]
    ):
        day_idx = np.arange(
            int(day_start),
            int(day_start + day_count),
            dtype=np.int32,
        )
        daylight_idx = day_idx[d['daylight'][day_idx]]
        if len(daylight_idx) == 0:
            continue

        # First prediction = first 15-minute daylight timestamp of target day.
        first_target_idx = int(daylight_idx[0])

        # target = issue + 24h + 15min, therefore:
        # issue = first_target - 96 steps - 1 step.
        first_origin = first_target_idx - DAY_AHEAD_OFFSET_STEPS - 1
        if first_origin < 0:
            continue

        k = 0
        while True:
            origin = first_origin + k * DEPLOYMENT_CADENCE_STEPS
            reference_idx = origin + DAY_AHEAD_OFFSET_STEPS
            first_pred_idx = reference_idx + 1
            last_pred_idx = reference_idx + HORIZON

            if last_pred_idx >= len(ts):
                break

            # Stop if the later 2-hour slot moved into another target day.
            if normalized_dates[first_pred_idx] != target_date64:
                break

            # Stop when the first prediction is no longer daylight.
            if not bool(d['daylight'][first_pred_idx]):
                break

            issue_ts = ts[origin]
            reference_ts = ts[reference_idx]
            first_target_ts = ts[first_pred_idx]

            # Hard timing checks.
            if reference_ts - issue_ts != pd.Timedelta(hours=24):
                raise RuntimeError(
                    f'House {h:02d}: next-day reference is not exactly +24 h at {issue_ts}.'
                )
            if first_target_ts - issue_ts != pd.Timedelta(hours=24, minutes=15):
                raise RuntimeError(
                    f'House {h:02d}: first target is not exactly +24 h 15 min at {issue_ts}.'
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
    _OPERATIONAL_ORIGIN_CACHE[h] = out
    return out


def build_operational_index(house_ids):
    """Exact next-day schedule used everywhere in this experiment."""
    hs = []
    origins_all = []

    for h in house_ids:
        origins = operational_origins_for_house(int(h))
        if len(origins) == 0:
            continue
        hs.append(np.full(len(origins), int(h), dtype=np.int16))
        origins_all.append(origins)

    if not origins_all:
        return np.empty(0, np.int16), np.empty(0, np.int32)

    return np.concatenate(hs), np.concatenate(origins_all)


def build_task_index(house_ids, stride):
    """Same-day: reference build_index(stride). Next-day: operational schedule."""
    if TASK == 'same_day':
        return build_index(house_ids, stride)
    return build_operational_index(house_ids)


def build_schedule_audit(house_ids, stride):
    """Human-readable timing audit (one row per retained forecast window)."""
    sample_house, sample_origin = build_task_index(house_ids, stride)
    rows = []
    for h in np.unique(sample_house):
        d = houses[int(h)]
        origins = sample_origin[sample_house == h].astype(np.int64)
        first_idx = origins + TARGET_OFFSET_STEPS + 1
        last_idx = origins + TARGET_OFFSET_STEPS + HORIZON
        fidx = origins[:, None] + TARGET_OFFSET_STEPS + np.arange(1, HORIZON + 1)[None, :]
        rows.append(pd.DataFrame({
            'House': int(h),
            'Issue_Timestamp': d['timestamps'][origins],
            'Issue_Daylight': d['daylight'][origins],
            'First_Target_Timestamp': d['timestamps'][first_idx],
            'Last_Target_Timestamp': d['timestamps'][last_idx],
            'First_Target_Daylight': d['daylight'][first_idx],
            'Daylight_Target_Count': d['daylight'][fidx].sum(axis=1).astype(int),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def schedule_sha256(sample_house, sample_origin):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sample_house, dtype=np.int16).tobytes())
    h.update(np.ascontiguousarray(sample_origin, dtype=np.int32).tobytes())
    return h.hexdigest()


# Schedule audit for Houses 1-11 only (Houses 12-13 are audited in Phase 2).
schedule_audit = build_schedule_audit(PRETEST_HOUSES, FINAL_STRIDE_FOR_TASK)
schedule_audit.to_csv(RUN_DIR / 'operational_schedule_audit_houses_01_11.csv.gz', index=False)
print(f'Schedule audit (Houses 1-11): {len(schedule_audit):,} windows')
val_final_house, val_final_origin = build_task_index(VAL_HOUSES, FINAL_STRIDE_FOR_TASK)
VAL_SCHEDULE_SHA256 = schedule_sha256(val_final_house, val_final_origin)


# -----------------------------------------------------------------------------
# XGBoost feature matrix: EXACT SAME INFORMATION as the neural models
#   24 target-time weather rows x 8 features (time-major flatten) + 3 static.
# No PV history, no House_ID, no AC/inverter features.
# -----------------------------------------------------------------------------
def build_matrix(house_ids, stride):
    sample_house, sample_origin = build_task_index(house_ids, stride)
    n = len(sample_origin)
    n_features = HORIZON * len(FUTURE_FEATURES) + len(STATIC_FEATURES)

    X = np.empty((n, n_features), dtype=np.float32)
    y_pu = np.empty((n, HORIZON), dtype=np.float32)
    mask = np.empty((n, HORIZON), dtype=bool)
    rated = np.empty(n, dtype=np.float32)
    origin_ts = np.empty(n, dtype='datetime64[ns]')
    target_start_ts = np.empty(n, dtype='datetime64[ns]')

    offsets = np.arange(1, HORIZON + 1, dtype=np.int32)
    weather_end = HORIZON * len(FUTURE_FEATURES)

    for h in np.unique(sample_house):
        pos = np.flatnonzero(sample_house == h)
        origins = sample_origin[pos]
        d = houses[int(h)]
        fidx = origins[:, None] + TARGET_OFFSET_STEPS + offsets[None, :]

        X[pos, :weather_end] = d['future'][fidx].reshape(len(pos), -1)
        X[pos, weather_end:] = d['static']
        y_pu[pos] = d['pv_pu'][fidx]
        mask[pos] = d['daylight'][fidx]
        rated[pos] = d['rated']
        origin_ts[pos] = d['timestamps'][origins]
        target_start_ts[pos] = d['timestamps'][origins + TARGET_OFFSET_STEPS + 1]

    return {
        'X': X,
        'y_pu': y_pu,
        'mask': mask,
        'rated': rated,
        'house': sample_house,
        'origin': sample_origin,
        'origin_ts': origin_ts,
        'target_start_ts': target_start_ts,
        'n_features': n_features,
    }


XGB_DEVICE = 'cuda' if shutil.which('nvidia-smi') else 'cpu'
print('XGBoost device:', XGB_DEVICE)
if XGB_DEVICE != 'cuda':
    print('WARNING: T4 GPU not detected. In Colab choose Runtime -> Change runtime type -> T4 GPU.')


def make_xgb_model(params, seed_offset=0, base_seed=SEED):
    return XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device=XGB_DEVICE,
        random_state=int(base_seed) + int(seed_offset),
        n_jobs=-1,
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=float(params['learning_rate']),
        subsample=float(params['subsample']),
        colsample_bytree=float(params['colsample_bytree']),
        min_child_weight=float(params['min_child_weight']),
        reg_alpha=float(params['reg_alpha']),
        reg_lambda=float(params['reg_lambda']),
        max_bin=int(params['max_bin']),
    )


# Optuna evaluates ALL 24 horizons. Selection criterion = the same quantity the
# neural networks use as val_loss: daylight-masked MSE of the raw per-unit output,
# pooled over every daylight point of every horizon on validation House 11.
TUNE_HORIZONS = list(range(HORIZON))

# Tuning data: same-day uses TUNING_STRIDE origins (as the neural same-day Optuna);
# next-day uses the exact operational schedule.
tune_train = build_matrix(TRAIN_HOUSES, TUNE_STRIDE_FOR_TASK)
tune_val = build_matrix(VAL_HOUSES, TUNE_STRIDE_FOR_TASK)
print(f"Optuna train samples: {len(tune_train['X']):,} | Optuna val samples: {len(tune_val['X']):,} "
      f"| features: {tune_train['n_features']}")


def objective(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [300, 500, 700, 900]),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'max_bin': trial.suggest_categorical('max_bin', [128, 256, 512]),
    }

    sq_err_sum = 0.0
    n_daylight_points = 0
    for j in TUNE_HORIZONS:
        tr_mask = tune_train['mask'][:, j]
        va_mask = tune_val['mask'][:, j]
        if not np.any(tr_mask) or not np.any(va_mask):
            raise RuntimeError(f'No daylight rows available for tuning horizon {j+1}.')

        m = make_xgb_model(params, seed_offset=j)
        m.fit(tune_train['X'][tr_mask], tune_train['y_pu'][tr_mask, j], verbose=False)
        pred = m.predict(tune_val['X'][va_mask]).astype(np.float64)
        err = pred - tune_val['y_pu'][va_mask, j].astype(np.float64)
        sq_err_sum += float(np.sum(err ** 2))
        n_daylight_points += int(np.sum(va_mask))
        del m
        gc.collect()

    # Pooled masked MSE over all daylight points of all 24 horizons (raw predictions).
    return sq_err_sum / n_daylight_points


# -----------------------------------------------------------------------------
# Metrics — exact copy of the corrected XGBoost reference.
# -----------------------------------------------------------------------------
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


def minutes_ahead(j):
    return TARGET_OFFSET_STEPS * STEP_MIN + (j + 1) * STEP_MIN


def evaluate_predictions(meta, raw_pred_pu, name):
    # Physical post-processing for reporting: no negative PV, exactly zero at night.
    pred_pu = np.maximum(raw_pred_pu.astype(np.float32), 0.0)
    pred_pu[~meta['mask']] = 0.0
    actual_pu = meta['y_pu']
    mask = meta['mask']
    rated = meta['rated']
    actual_w = actual_pu * rated[:, None]
    pred_w = pred_pu * rated[:, None]

    overall = point_metrics(actual_w, pred_w, actual_pu, pred_pu, mask)
    overall['Split'] = name
    planning_overall, window_df = planning_metrics(actual_w, pred_w, mask)
    planning_overall['Split'] = name

    window_df.insert(0, 'House', meta['house'].astype(int))
    window_df.insert(1, 'Issue_Timestamp', pd.to_datetime(meta['origin_ts']))
    window_df.insert(2, 'Target_Start_Timestamp', pd.to_datetime(meta['target_start_ts']))
    window_df.insert(3, 'Target_End_Timestamp', pd.to_datetime(meta['target_start_ts'])
                     + pd.to_timedelta((HORIZON - 1) * STEP_MIN, unit='m'))

    by_h = []
    for j in range(HORIZON):
        row = point_metrics(
            actual_w[:, j:j+1], pred_w[:, j:j+1],
            actual_pu[:, j:j+1], pred_pu[:, j:j+1],
            mask[:, j:j+1],
        )
        row['Horizon_step'] = j + 1
        row['Window_minutes'] = (j + 1) * STEP_MIN
        row['Minutes_ahead'] = minutes_ahead(j)
        by_h.append(row)

    return {
        'overall': overall,
        'planning_overall': planning_overall,
        'by_horizon': pd.DataFrame(by_h),
        'window_metrics': window_df,
        'pred_pu': pred_pu,
    }


# -----------------------------------------------------------------------------
# Deployment helpers (task-specific). Same feature layout as training:
# 24 target-time weather rows x 8 features (scaled) + 3 static features (scaled).
# -----------------------------------------------------------------------------
def build_next_day_issue_schedule(target_day_weather):
    """
    Build the real deployment issue schedule from one next-day 15-minute weather table.

    Required columns: Timestamp, Solar_Zenith_rad.
    Example: sunrise 07:08 -> first daylight grid 07:15 -> issue today 07:00,
    then 09:00, 11:00, ... while each window's first target is daylight.
    """
    x = target_day_weather.copy()
    if 'Timestamp' not in x.columns or 'Solar_Zenith_rad' not in x.columns:
        raise ValueError('target_day_weather must contain Timestamp and Solar_Zenith_rad.')

    x['Timestamp'] = pd.to_datetime(x['Timestamp'], errors='raise')
    if x['Timestamp'].duplicated().any() or not x['Timestamp'].is_monotonic_increasing:
        raise ValueError('target_day_weather timestamps must be unique and increasing.')

    if len(x) > 1:
        dt = x['Timestamp'].diff().dropna().dt.total_seconds().to_numpy() / 60.0
        if not np.all(dt == STEP_MIN):
            raise ValueError(f'target_day_weather must be continuous {STEP_MIN}-minute data.')

    daylight = x['Solar_Zenith_rad'].to_numpy(float) < (np.pi / 2.0)
    daylight_pos = np.flatnonzero(daylight)
    if len(daylight_pos) == 0:
        return pd.DataFrame(columns=[
            'Issue_Timestamp_Today',
            'Next_Day_Reference_Timestamp',
            'First_Target_Timestamp',
            'Last_Target_Timestamp',
        ])

    first_target_pos = int(daylight_pos[0])
    first_target_ts = pd.Timestamp(x['Timestamp'].iloc[first_target_pos])
    reference_ts = first_target_ts - pd.Timedelta(minutes=STEP_MIN)
    ts_values = x['Timestamp'].to_numpy(dtype='datetime64[ns]')

    rows = []
    while True:
        target_start = reference_ts + pd.Timedelta(minutes=STEP_MIN)
        match = np.flatnonzero(ts_values == np.datetime64(target_start))
        if len(match) == 0:
            break
        pos = int(match[0])
        if not daylight[pos]:
            break

        rows.append({
            'Issue_Timestamp_Today': reference_ts - pd.Timedelta(days=1),
            'Next_Day_Reference_Timestamp': reference_ts,
            'First_Target_Timestamp': target_start,
            'Last_Target_Timestamp': reference_ts + pd.Timedelta(minutes=HORIZON * STEP_MIN),
        })
        reference_ts += pd.Timedelta(hours=DEPLOYMENT_UPDATE_HOURS)

    return pd.DataFrame(rows)


def build_same_day_issue_schedule(day_weather):
    """
    Same-day deployment issue schedule on the fixed 15-minute grid.

    Required columns: Timestamp, Solar_Zenith_rad. The table must extend at least
    6 h beyond the last issue time considered. Same rule as training:
    the issue time must be daylight AND its next 24 targets must contain daylight.
    Example: sunrise 07:08 -> 07:00 excluded, 07:15 first issue -> 07:30 ... 13:15.
    """
    x = day_weather.copy()
    if 'Timestamp' not in x.columns or 'Solar_Zenith_rad' not in x.columns:
        raise ValueError('day_weather must contain Timestamp and Solar_Zenith_rad.')
    x['Timestamp'] = pd.to_datetime(x['Timestamp'], errors='raise')
    if x['Timestamp'].duplicated().any() or not x['Timestamp'].is_monotonic_increasing:
        raise ValueError('day_weather timestamps must be unique and increasing.')
    if len(x) > 1:
        dt = x['Timestamp'].diff().dropna().dt.total_seconds().to_numpy() / 60.0
        if not np.all(dt == STEP_MIN):
            raise ValueError(f'day_weather must be continuous {STEP_MIN}-minute data.')

    daylight = x['Solar_Zenith_rad'].to_numpy(np.float32) < (np.pi / 2)
    ts = pd.DatetimeIndex(x['Timestamp'])
    rows = []
    for pos in range(0, len(x) - HORIZON):
        if daylight[pos] and np.any(daylight[pos + 1:pos + 1 + HORIZON]):
            rows.append({
                'Issue_Timestamp': ts[pos],
                'First_Target_Timestamp': ts[pos + 1],
                'Last_Target_Timestamp': ts[pos + HORIZON],
            })
    return pd.DataFrame(rows, columns=['Issue_Timestamp', 'First_Target_Timestamp',
                                       'Last_Target_Timestamp'])


deployment_models = None  # 24 seed-42 regressors, set after final training


def predict_6h(future_weather_24, static_values, rated_power_w):
    """
    Predict one 24-step 6-hour window with the 24 deployment (seed 42) regressors.
    future_weather_24 = the 24 TARGET-time weather rows (t+15 ... t+360 for same-day,
    next-day t+24h15m ... t+30h for next-day).
    """
    if deployment_models is None:
        raise RuntimeError('Deployment models are not loaded.')
    if len(future_weather_24) != HORIZON:
        raise ValueError(f'future_weather_24 must contain exactly {HORIZON} rows.')
    missing = [c for c in FUTURE_FEATURES if c not in future_weather_24.columns]
    if missing:
        raise ValueError(f'Missing future weather columns: {missing}')
    missing_static = [c for c in STATIC_FEATURES if c not in static_values]
    if missing_static:
        raise ValueError(f'Missing static values: {missing_static}')
    rated_power_w = float(rated_power_w)
    if rated_power_w <= 0:
        raise ValueError('rated_power_w must be positive.')

    raw_future = future_weather_24[FUTURE_FEATURES].to_numpy(np.float32)
    raw_static = np.asarray([static_values[c] for c in STATIC_FEATURES], np.float32).reshape(1, -1)
    if not np.isfinite(raw_future).all() or not np.isfinite(raw_static).all():
        raise ValueError('Future weather/static values contain NaN/Inf.')

    future_scaled = future_scaler.transform(raw_future).astype(np.float32).reshape(1, -1)
    static_scaled = static_scaler.transform(raw_static).astype(np.float32)
    X = np.concatenate([future_scaled, static_scaled], axis=1)

    pred_pu = np.asarray([deployment_models[j].predict(X)[0] for j in range(HORIZON)], dtype=np.float32)
    pred_pu = np.maximum(pred_pu, 0.0)
    zen_col = FUTURE_FEATURES.index('Solar_Zenith_rad')
    daylight = raw_future[:, zen_col] < (np.pi / 2.0)
    pred_pu[~daylight] = 0.0
    pred_w = pred_pu * rated_power_w

    out = pd.DataFrame({'PV_Predicted_pu': pred_pu, 'PV_Predicted_W': pred_w, 'Daylight': daylight})
    if 'Timestamp' in future_weather_24.columns:
        out.insert(0, 'Timestamp', pd.to_datetime(future_weather_24['Timestamp']).to_numpy())
    return out, out[out['Daylight']].reset_index(drop=True)


predict_same_day_6h = predict_6h
predict_next_day_6h = predict_6h


def check_deployment_schedule_matches_training(h):
    """The deployment issue-schedule helper must reproduce the training index (House 11)."""
    d = houses[h]
    ts = pd.DatetimeIndex(d['timestamps'])
    _, origins = build_task_index([h], FINAL_STRIDE_FOR_TASK)
    issue_ts = ts[origins]
    # Training issues grouped by the day they belong to (target day for next-day).
    group_day = (issue_ts + pd.Timedelta(hours=24, minutes=15)).normalize() if TASK == 'next_day' \
        else issue_ts.normalize()
    expected_by_day = pd.Series(issue_ts).groupby(group_day).apply(set).to_dict()
    weather = pd.DataFrame({'Timestamp': ts, 'Solar_Zenith_rad': d['future_raw'][:, FUTURE_FEATURES.index('Solar_Zenith_rad')]})
    days = ts.normalize()
    unique_days = days.unique()
    n_checked = 0
    # Skip the first and last two days of the file (file-edge truncation).
    for day in unique_days[2:-2]:
        day_pos = np.flatnonzero(days == day)
        expected = expected_by_day.get(day, set())
        if TASK == 'next_day':
            helper = build_next_day_issue_schedule(weather.iloc[day_pos[0]:day_pos[-1] + 1])
            helper_issue = set(pd.DatetimeIndex(helper['Issue_Timestamp_Today']))
        else:
            table = weather.iloc[day_pos[0]:day_pos[-1] + 1 + HORIZON]
            helper = build_same_day_issue_schedule(table)
            helper_issue = set(pd.DatetimeIndex(helper['Issue_Timestamp']))
        if helper_issue != expected:
            raise RuntimeError(f'Deployment issue schedule differs from training index on {day.date()}.')
        n_checked += 1
    print(f'Deployment issue-schedule helper matches the training index on {n_checked} House-{h} days.')


check_deployment_schedule_matches_training(VAL_HOUSES[0])


def check_deployment_predictions_match_evaluation(meta, eval_pred_pu, n_windows=48):
    """predict_6h on raw weather rows must reproduce the evaluation pipeline outputs."""
    idx = np.unique(np.linspace(0, len(meta['origin']) - 1, n_windows).astype(int))
    max_diff = 0.0
    for i in idx:
        h = int(meta['house'][i])
        d = houses[h]
        fidx = meta['origin'][i] + TARGET_OFFSET_STEPS + np.arange(1, HORIZON + 1)
        wx = pd.DataFrame(d['future_raw'][fidx], columns=FUTURE_FEATURES)
        wx.insert(0, 'Timestamp', d['timestamps'][fidx])
        static_values = dict(zip(STATIC_FEATURES, d['static_raw'].tolist()))
        out, _ = predict_6h(wx, static_values, d['rated'])
        max_diff = max(max_diff, float(np.max(np.abs(out['PV_Predicted_pu'].to_numpy() - eval_pred_pu[i]))))
    if max_diff > 1e-4:
        raise RuntimeError(f'Deployment helper differs from evaluation pipeline (max |diff| = {max_diff:.2e} pu).')
    print(f'Deployment helper reproduces evaluation outputs on {len(idx)} House-11 windows '
          f'(max |diff| = {max_diff:.2e} pu).')


# -----------------------------------------------------------------------------
# Optuna — ONE study for this (XGBoost, task); House 11 only.
# -----------------------------------------------------------------------------
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=OPTUNA_TRIALS, gc_after_trial=True)
print('Best params:', study.best_params)
study.trials_dataframe().to_csv(RUN_DIR / 'optuna_trials.csv', index=False)
with open(RUN_DIR / 'best_hyperparameters.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
best = dict(study.best_params)
del tune_train, tune_val
gc.collect()

# Final fit / evaluation data: same-day every 15 min, next-day operational schedule.
train = build_matrix(TRAIN_HOUSES, FINAL_STRIDE_FOR_TASK)
val = build_matrix(VAL_HOUSES, FINAL_STRIDE_FOR_TASK)
if schedule_sha256(val['house'], val['origin']) != VAL_SCHEDULE_SHA256:
    raise RuntimeError('Validation schedule differs from the audited schedule.')
print(f"Final train samples: {len(train['X']):,} | House-11 samples: {len(val['X']):,}")
val_meta = {k: val[k] for k in ('y_pu', 'mask', 'rated', 'house', 'origin', 'origin_ts', 'target_start_ts')}

# -----------------------------------------------------------------------------
# Final frozen configuration retrained with seeds 42, 43, 44.
# Regressor j uses random_state = seed + j (reference convention SEED + j).
# -----------------------------------------------------------------------------
seed_rows, seed_planning_rows, seed_info = [], [], {}
for s in FINAL_SEEDS:
    print(f'\n========== {ARCH_LABEL} | {TASK} | final seed {s} ==========')
    seed_dir = MODEL_DIR / f'seed_{s}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    raw_pred_pu = np.zeros((len(val['X']), HORIZON), dtype=np.float32)
    model_rows, seed_models = [], []

    for j in range(HORIZON):
        tr_mask = train['mask'][:, j]
        va_mask = val['mask'][:, j]
        if not np.any(tr_mask):
            raise RuntimeError(f'No daylight training targets for output {j+1}.')
        if not np.any(va_mask):
            raise RuntimeError(f'No daylight validation targets for output {j+1}.')

        m = make_xgb_model(best, seed_offset=j, base_seed=s)
        m.fit(train['X'][tr_mask], train['y_pu'][tr_mask, j], verbose=False)
        raw_pred_pu[:, j] = m.predict(val['X']).astype(np.float32)

        lead = minutes_ahead(j)
        model_path = seed_dir / f'xgb_output_{j+1:02d}_lead_{lead:04d}min.json'
        m.save_model(model_path)
        model_rows.append({
            'Horizon_step': j + 1,
            'Window_minutes': (j + 1) * STEP_MIN,
            'Minutes_ahead': lead,
            'Random_state': s + j,
            'Training_daylight_rows': int(np.sum(tr_mask)),
            'Validation_daylight_rows': int(np.sum(va_mask)),
            'Model_file': model_path.name,
        })
        if s == DEPLOYMENT_SEED:
            seed_models.append(m)
        else:
            del m
        gc.collect()
        print(f'  output {j+1:02d}/24 (lead {lead} min) fitted')

    pd.DataFrame(model_rows).to_csv(seed_dir / 'xgboost_model_summary.csv', index=False)
    raw_val_mse = float(np.sum(((raw_pred_pu - val['y_pu']) ** 2) * val['mask']) / np.sum(val['mask']))
    res = evaluate_predictions(val_meta, raw_pred_pu, 'Validation_House_11')

    row = dict(res['overall']); row['Seed'] = s; seed_rows.append(row)
    prow = dict(res['planning_overall']); prow['Seed'] = s; seed_planning_rows.append(prow)
    res['by_horizon'].to_csv(seed_dir / 'validation_metrics_by_horizon.csv', index=False)
    res['window_metrics'].to_csv(seed_dir / 'validation_window_planning_metrics.csv.gz', index=False)
    np.savez_compressed(
        seed_dir / 'house11_predictions.npz',
        house=val['house'], origin=val['origin'],
        raw_pred_pu=raw_pred_pu, pred_pu=res['pred_pu'],
        y_pu=val['y_pu'], mask=val['mask'],
    )
    seed_info[s] = {
        'model_dir': str(seed_dir.relative_to(RUN_DIR)),
        'model_files': [r['Model_file'] for r in model_rows],
        'raw_masked_val_mse_pu': raw_val_mse,
        'house11_nRMSE_percent': float(res['overall']['nRMSE_percent']),
    }

    if s == DEPLOYMENT_SEED:
        deployment_models = seed_models
        check_deployment_predictions_match_evaluation(val_meta, res['pred_pu'])

# -----------------------------------------------------------------------------
# House-11 summary: per seed and mean ± SD (sample SD, ddof=1) over seeds 42-44.
# -----------------------------------------------------------------------------
def mean_sd_table(rows):
    df = pd.DataFrame(rows)
    num = df.drop(columns=[c for c in ('Seed', 'Split') if c in df.columns]).astype(float)
    return pd.DataFrame([num.mean(), num.std(ddof=1)], index=['mean', 'sd'])


per_seed_df = pd.DataFrame(seed_rows)
planning_seed_df = pd.DataFrame(seed_planning_rows)
per_seed_df.to_csv(RUN_DIR / 'house11_point_metrics_per_seed.csv', index=False)
planning_seed_df.to_csv(RUN_DIR / 'house11_planning_metrics_per_seed.csv', index=False)
mean_sd_table(seed_rows).to_csv(RUN_DIR / 'house11_point_metrics_mean_sd.csv')
mean_sd_table(seed_planning_rows).to_csv(RUN_DIR / 'house11_planning_metrics_mean_sd.csv')
print(f'\nHOUSE 11 — {ARCH_LABEL} — {TASK} — per seed'); display(per_seed_df)
print('\nHOUSE 11 — mean ± SD over seeds'); display(mean_sd_table(seed_rows))

try:
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.figure.tight_layout(); ax.figure.savefig(RUN_DIR / 'optuna_history.png', dpi=160); plt.show()
except Exception as e:
    print('Could not create Optuna history plot:', e)
try:
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    ax.figure.tight_layout(); ax.figure.savefig(RUN_DIR / 'optuna_parameter_importance.png', dpi=160); plt.show()
except Exception as e:
    print('Could not create Optuna parameter-importance plot:', e)
plt.close('all')

nrmse_seeds = [seed_info[s]['house11_nRMSE_percent'] for s in FINAL_SEEDS]
summary = {
    'phase': 1,
    'architecture': ARCH_KEY,
    'architecture_label': ARCH_LABEL,
    'task': TASK,
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'train_houses': TRAIN_HOUSES, 'validation_house': VAL_HOUSES,
    'houses_loaded': PRETEST_HOUSES,
    'test_houses_accessed': False,
    'house_file_sha256': {str(h): v for h, v in house_file_sha256.items()},
    'future_scaler': {'mean': future_scaler.mean_.tolist(), 'scale': future_scaler.scale_.tolist()},
    'static_scaler': {'mean': static_scaler.mean_.tolist(), 'scale': static_scaler.scale_.tolist()},
    'schedule': (
        'same_day: origin_is_daylight & future_has_daylight; targets origin+1..origin+24; '
        f'Optuna stride {TUNING_STRIDE}, final stride {FINAL_STRIDE}'
        if TASK == 'same_day' else
        'next_day: first target = first 15-min daylight grid of target day; '
        'issue = first target - 96 - 1 steps; every 8 steps (2 h) while first target is daylight; '
        'targets origin+96+1..origin+96+24; same schedule for Optuna, final fit, validation'
    ),
    'target_offset_steps': TARGET_OFFSET_STEPS,
    'house11_validation_schedule_sha256': VAL_SCHEDULE_SHA256,
    'house11_validation_windows': int(len(val_final_origin)),
    'future_features': FUTURE_FEATURES, 'static_features': STATIC_FEATURES,
    'feature_layout': 'future_weather_24x8_flat (time-major) + static_3',
    'target': TARGET, 'target_normalization': 'PV / Array_Rated_Power_W',
    'xgboost_device': XGB_DEVICE,
    'xgboost_models_per_seed': HORIZON,
    'optuna_trials': OPTUNA_TRIALS,
    'optuna_tuning_horizons_steps': [j + 1 for j in TUNE_HORIZONS],
    'optuna_objective': 'House-11 daylight-masked MSE (per-unit) of raw predictions, pooled over all daylight points of all 24 horizons (same quantity as neural val_loss)',
    'best_hyperparameters': best,
    'optuna_best_value': float(study.best_value),
    'final_seeds': FINAL_SEEDS,
    'deployment_seed': DEPLOYMENT_SEED,
    'per_seed': {str(s): v for s, v in seed_info.items()},
    'house11_nRMSE_percent_mean': float(np.mean(nrmse_seeds)),
    'house11_nRMSE_percent_sd': float(np.std(nrmse_seeds, ddof=1)),
    'selection_metric_note': 'nRMSE_percent on post-processed House-11 predictions (clip >= 0, night = 0)',
    'primary_point_metrics': ['RMSE_W', 'MAE_W', 'nRMSE_percent', 'WAPE_percent', 'R2'],
    'secondary_point_metrics': ['sMAPE_percent'],
    'supplementary_point_metrics': ['MAPE_1pct_percent', 'N_MAPE_1pct_included', 'N_MAPE_1pct_excluded'],
    'pv_lookback_used': False, 'predicted_pv_feedback_used': False, 'past_weather_used': False,
    'house_id_used_as_feature': False, 'ac_inverter_features_used': False,
    'night_targets_masked': True,
}
with open(RUN_DIR / 'phase1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nPHASE 1 DONE — {ARCH_LABEL} — {TASK}')
print(f"House-11 nRMSE (mean ± SD over seeds): {summary['house11_nRMSE_percent_mean']:.4f} ± "
      f"{summary['house11_nRMSE_percent_sd']:.4f} %")
print('Outputs:', RUN_DIR)
