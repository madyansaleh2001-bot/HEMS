# =============================================================================
# COMBINED SAME-DAY + NEXT-DAY EXPERIMENT (Experiment A) — PHASE 1 — LSTM
# Google Colab / TensorFlow / Optuna — Strategy 2 split — NO PV LOOKBACK
#
# ONE execution of this file performs BOTH tasks automatically (no TASK switch):
#   1. SAME-DAY LSTM  — own Optuna study 'lstm_same_day', own hyperparameters,
#      seeds 42/43/44, House-11 metrics, saved models/results.
#   2. NEXT-DAY LSTM  — own Optuna study 'lstm_next_day', own hyperparameters,
#      seeds 42/43/44, House-11 metrics, saved models/results.
# Two independent task-specific models (Design B); nothing is shared between them
# except the data, the scalers fitted on Houses 1-10, and the evaluation code.
#
# ONE OPERATIONAL SCHEDULE RULE FOR BOTH TASKS (FUTURE_OFFSETS = 1..24):
#   For every target day: first target = first 15-min daylight grid point
#   (Solar_Zenith_rad < pi/2); first issue = first target - offset - 15 min;
#   further issues every cadence while the window's FIRST target is daylight.
#   The issue time itself does NOT need to be daylight.
#     SAME-DAY : targets = origin + FUTURE_OFFSETS          (t+15 ... t+360)
#                offset 0 days, cadence 15 min.
#                Sunrise 07:08 -> first daylight grid 07:15 -> first issue 07:00
#                -> 07:15 ... 13:00; issue 07:15 -> 07:30 ... 13:15; ...
#     NEXT-DAY : targets = origin + 96 + FUTURE_OFFSETS     (t+24h15 ... t+30h)
#                offset 1 day, cadence 2 h.
#                Tomorrow sunrise 07:08 -> first target 07:15 -> issue today 07:00
#                -> tomorrow 07:15 ... 13:00; issue 09:00 -> 09:15 ... 15:00; ...
#   Target positions after sunset are masked from loss/metrics and reported as 0 W.
#   Same-Day Optuna uses every 4th issue of that schedule (TUNING_STRIDE = 4, hourly)
#   to save compute; final training and ALL evaluation use the full 15-min schedule.
#   Next-Day uses its 2-hour schedule for Optuna, training and evaluation.
#
# PHASE 1 ISOLATION: only Houses 1-11 are opened. Houses 12-13 are not read,
# validated, transformed, scheduled or evaluated here; their paths are not even built.
# Houses 1-10 = training, House 11 = Optuna / early stopping / checkpoint / selection.
#
# Inputs: 24 x 8 target-time weather + 3 static PV-system parameters only.
# Target: PV_DC_Power_W / Array_Rated_Power_W (synthetic pvlib DC target in Experiment A).
# Optuna objective: House-11 daylight-masked MSE of the RAW per-unit output.
# Optuna storage: persistent SQLite on Drive, load_if_exists=True (resumable).
# Reporting: predictions clipped >= 0 and exactly 0 W at night.
# Final frozen configuration retrained with seeds 42, 43, 44 (mean ± SD); deployment seed 42.
# Metrics: Primary RMSE, MAE, nRMSE, WAPE, R2 | Secondary sMAPE |
#          Supplementary MAPE_1pct (+ N included / N excluded) | HEMS planning metrics.
# Resuming: a task whose phase1_summary.json exists is skipped; an unfinished task
# resumes its Optuna study and then repeats its final seed training.
# =============================================================================

import os, gc, json, math, random, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

OVERWRITE_PHASE1_RUN = False   # True only to deliberately redo an already completed task

# >>> ARCHITECTURE IDENTITY (architecture-specific) >>>
ARCH_KEY = 'lstm'
ARCH_LABEL = 'LSTM'
# <<< ARCHITECTURE IDENTITY <<<

SEED = 42                       # Optuna sampler + tuning runs (as in the references)
FINAL_SEEDS = [42, 43, 44]      # frozen final seeds
DEPLOYMENT_SEED = 42            # frozen deployment seed
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

GPU_DEVICES = tf.config.list_physical_devices('GPU')
print('TensorFlow GPU devices:', GPU_DEVICES)
if not GPU_DEVICES:
    print('WARNING: No GPU detected. In Colab choose Runtime -> Change runtime type -> T4 GPU.')

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

TASKS = ('same_day', 'next_day')   # both tasks run automatically, in this order

EXPERIMENT_DIR = DATA_DIR / 'combined_sameday_nextday_experiment'
PHASE1_DIR = EXPERIMENT_DIR / 'phase1'
SELECTION_FILE = EXPERIMENT_DIR / 'frozen_selection.json'

# Phase-1 lock: once the architecture decision is frozen, Phase 1 may not be re-run.
if SELECTION_FILE.exists():
    raise RuntimeError(
        f'{SELECTION_FILE} exists: the architecture decision is frozen. '
        'Phase 1 must not be re-run after selection.'
    )
PHASE1_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_HOUSES = list(range(1, 11))
VAL_HOUSES = [11]
TEST_HOUSES = [12, 13]   # listed for documentation only; never opened in Phase 1
PRETEST_HOUSES = TRAIN_HOUSES + VAL_HOUSES

HORIZON = 24
STEP_MIN = 15
STEPS_PER_DAY = (24 * 60) // STEP_MIN  # 96
DAY_AHEAD_OFFSET_STEPS = STEPS_PER_DAY
FUTURE_OFFSETS = np.arange(1, HORIZON + 1, dtype=np.int32)   # 1 ... 24
SAME_DAY_CADENCE_STEPS = 1                                    # every 15 min
DEPLOYMENT_UPDATE_HOURS = 2
DEPLOYMENT_CADENCE_STEPS = (DEPLOYMENT_UPDATE_HOURS * 60) // STEP_MIN  # 8 = every 2 h
TUNING_STRIDE = 4   # Same-Day Optuna only: every 4th operational issue (hourly)
OPTUNA_TRIALS = 20
TUNING_EPOCHS = 15
FINAL_EPOCHS = 60
PATIENCE = 8
MAPE_THRESHOLD_PU = 0.01  # supplementary MAPE only when actual PV >= 1% of rated power

# The only temporal difference between the tasks: offset (0 / 1 day) and refresh cadence.
TASK_TARGET_OFFSET_STEPS = {'same_day': 0, 'next_day': DAY_AHEAD_OFFSET_STEPS}
TASK_OPERATIONAL_CADENCE_STEPS = {'same_day': SAME_DAY_CADENCE_STEPS, 'next_day': DEPLOYMENT_CADENCE_STEPS}
TASK_TUNING_CADENCE_STEPS = {'same_day': SAME_DAY_CADENCE_STEPS * TUNING_STRIDE,
                              'next_day': DEPLOYMENT_CADENCE_STEPS}

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
print(f'{ARCH_LABEL} | tasks = {TASKS} | train/validation files found. '
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

# Fit scalers ONLY on training Houses 1-10 (one definition shared by both tasks).
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


# -----------------------------------------------------------------------------
# OPERATIONAL SCHEDULE — one rule for both tasks.
# -----------------------------------------------------------------------------
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


def build_task_index(house_ids, cadence_steps):
    """Operational schedule of the CURRENT task for several houses."""
    hs = []
    origins_all = []

    for h in house_ids:
        origins = operational_origins_for_house(int(h), TARGET_OFFSET_STEPS, cadence_steps)
        if len(origins) == 0:
            continue
        hs.append(np.full(len(origins), int(h), dtype=np.int16))
        origins_all.append(origins)

    if not origins_all:
        return np.empty(0, np.int16), np.empty(0, np.int32)

    return np.concatenate(hs), np.concatenate(origins_all)


def build_schedule_audit(house_ids, cadence_steps):
    """Human-readable timing audit (one row per retained forecast window)."""
    sample_house, sample_origin = build_task_index(house_ids, cadence_steps)
    rows = []
    for h in np.unique(sample_house):
        d = houses[int(h)]
        origins = sample_origin[sample_house == h].astype(np.int64)
        fidx = origins[:, None] + TARGET_OFFSET_STEPS + FUTURE_OFFSETS[None, :]
        rows.append(pd.DataFrame({
            'House': int(h),
            'Issue_Timestamp': d['timestamps'][origins],
            'Issue_Daylight': d['daylight'][origins],
            'First_Target_Timestamp': d['timestamps'][fidx[:, 0]],
            'Last_Target_Timestamp': d['timestamps'][fidx[:, -1]],
            'First_Target_Daylight': d['daylight'][fidx[:, 0]],
            'Daylight_Target_Count': d['daylight'][fidx].sum(axis=1).astype(int),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def schedule_sha256(sample_house, sample_origin):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sample_house, dtype=np.int16).tobytes())
    h.update(np.ascontiguousarray(sample_origin, dtype=np.int32).tobytes())
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Keras data pipeline — same packed target/mask format as the references.
# -----------------------------------------------------------------------------
class MultiHouseSequence(tf.keras.utils.Sequence):
    def __init__(self, house_ids, batch_size, cadence_steps, shuffle):
        super().__init__()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.sample_house, self.sample_origin = build_task_index(house_ids, cadence_steps)
        self.order = np.arange(len(self.sample_origin), dtype=np.int64)
        self.future_offsets = FUTURE_OFFSETS
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.order) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)

    def __getitem__(self, idx):
        sel = self.order[idx*self.batch_size:min((idx+1)*self.batch_size, len(self.order))]
        bh = self.sample_house[sel]
        bo = self.sample_origin[sel]
        b = len(sel)

        future = np.empty((b, HORIZON, len(FUTURE_FEATURES)), np.float32)
        static = np.empty((b, len(STATIC_FEATURES)), np.float32)
        target = np.empty((b, HORIZON), np.float32)
        mask = np.empty((b, HORIZON), np.float32)

        for h in np.unique(bh):
            pos = np.flatnonzero(bh == h)
            origin = bo[pos]
            d = houses[int(h)]
            fidx = origin[:, None] + TARGET_OFFSET_STEPS + self.future_offsets[None, :]
            future[pos] = d['future'][fidx]
            static[pos] = d['static']
            target[pos] = d['pv_pu'][fidx]
            mask[pos] = d['daylight'][fidx].astype(np.float32)

        # Same packed target/mask format used by the original workflow.
        return {'future_weather': future, 'static': static}, np.concatenate([target, mask], axis=1)

    def collect_meta(self):
        if self.shuffle:
            raise ValueError('collect_meta requires shuffle=False')
        n = len(self.sample_origin)
        y = np.empty((n, HORIZON), np.float32)
        mask = np.empty((n, HORIZON), bool)
        rated = np.empty(n, np.float32)
        ts = np.empty(n, dtype='datetime64[ns]')
        target_start_ts = np.empty(n, dtype='datetime64[ns]')
        for h in np.unique(self.sample_house):
            pos = np.flatnonzero(self.sample_house == h)
            origin = self.sample_origin[pos]
            d = houses[int(h)]
            fidx = origin[:, None] + TARGET_OFFSET_STEPS + self.future_offsets[None, :]
            y[pos] = d['pv_pu'][fidx]
            mask[pos] = d['daylight'][fidx]
            rated[pos] = d['rated']
            ts[pos] = d['timestamps'][origin]
            target_start_ts[pos] = d['timestamps'][origin + TARGET_OFFSET_STEPS + 1]
        return {
            'y_pu': y,
            'mask': mask,
            'rated': rated,
            'house': self.sample_house.copy(),
            'origin': self.sample_origin.copy(),
            'origin_ts': ts,
            'target_start_ts': target_start_ts,
        }


@tf.keras.utils.register_keras_serializable(package='PVForecast')
class MaskedMSE(tf.keras.losses.Loss):
    def __init__(self, horizon=HORIZON, name='masked_mse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.horizon = int(horizon)

    def call(self, y_true, y_pred):
        target = y_true[:, :self.horizon]
        mask = y_true[:, self.horizon:]

        squared_error = tf.square(y_pred - target) * mask

        # IMPORTANT:
        # Average over ALL valid daylight target points in the batch.
        # We do NOT first normalize each sample by its own daylight count.
        # Therefore a pre-sunrise sample with only 1-2 daylight horizons
        # cannot receive the same total weight as a midday sample with
        # 24 daylight horizons.
        numerator = tf.reduce_sum(squared_error)
        denominator = tf.reduce_sum(mask)

        return tf.math.divide_no_nan(
            numerator,
            denominator,
        )

    def get_config(self):
        c = super().get_config()
        c.update({'horizon': self.horizon})
        return c


# >>> ARCHITECTURE MODEL AND SEARCH SPACE (architecture-specific) >>>
# Same-Day LSTM reference architecture and Optuna ranges.
def suggest_params(trial):
    return {
        'future_units': trial.suggest_categorical('future_units', [32, 64, 96, 128]),
        'dense_units': trial.suggest_categorical('dense_units', [64, 128, 192, 256]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.4, step=0.1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
    }


def build_model(params):
    tf.keras.backend.clear_session()
    future_units = int(params['future_units'])
    dense_units = int(params['dense_units'])
    dropout = float(params['dropout'])

    future_in = keras.Input((HORIZON, len(FUTURE_FEATURES)), name='future_weather')
    b = layers.LSTM(future_units, name='future_weather_lstm')(future_in)
    b = layers.Dropout(dropout)(b)

    static_in = keras.Input((len(STATIC_FEATURES),), name='static')
    c = layers.Dense(32, activation='relu', name='static_dense')(static_in)

    x = layers.Concatenate(name='fusion')([b, c])
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(max(32, dense_units // 2), activation='relu')(x)
    out = layers.Dense(HORIZON, activation='linear', name='pv_pu_forecast')(x)

    model = keras.Model(
        {'future_weather': future_in, 'static': static_in},
        out,
        name=f'PV_6h_Strategy2_LSTM_NoPVLookback_{TASK}',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(params['learning_rate'])),
        loss=MaskedMSE(),
    )
    return model
# <<< ARCHITECTURE MODEL AND SEARCH SPACE <<<


class PruneCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get('val_loss')
        if v is not None:
            self.trial.report(float(v), epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def objective(trial):
    params = suggest_params(trial)
    batch = int(params['batch_size'])

    # Same-Day: every 4th operational issue (hourly). Next-Day: its 2-hour schedule.
    tr = MultiHouseSequence(TRAIN_HOUSES, batch, TUNE_CADENCE, True)
    va = MultiHouseSequence(VAL_HOUSES, batch, TUNE_CADENCE, False)

    model = build_model(params)
    hist = model.fit(
        tr,
        validation_data=va,
        epochs=TUNING_EPOCHS,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=4, restore_best_weights=True
            ),
            PruneCallback(trial),
        ],
    )
    # Raw (unclipped) daylight-masked MSE on House 11 = neural val_loss.
    best_score = float(np.min(hist.history['val_loss']))
    del model, tr, va
    gc.collect()
    return best_score


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


def mean_sd_table(rows):
    df = pd.DataFrame(rows)
    num = df.drop(columns=[c for c in ('Seed', 'Split') if c in df.columns]).astype(float)
    return pd.DataFrame([num.mean(), num.std(ddof=1)], index=['mean', 'sd'])


# -----------------------------------------------------------------------------
# Deployment helpers. Same feature layout as training:
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


def build_same_day_issue_schedule(today_weather):
    """
    Build the real deployment Same-Day issue schedule from today's 15-minute weather table.

    Required columns: Timestamp, Solar_Zenith_rad.
    Same rule as training: first target = first daylight 15-minute grid point,
    first issue = first target - 15 min, then every 15 min while the first target
    is daylight. The issue time itself need not be daylight.
    Example: sunrise 07:08 -> first daylight grid 07:15 -> first issue 07:00
    (07:15 ... 13:00), then 07:15 (07:30 ... 13:15), 07:30, ...
    """
    x = today_weather.copy()
    if 'Timestamp' not in x.columns or 'Solar_Zenith_rad' not in x.columns:
        raise ValueError('today_weather must contain Timestamp and Solar_Zenith_rad.')

    x['Timestamp'] = pd.to_datetime(x['Timestamp'], errors='raise')
    if x['Timestamp'].duplicated().any() or not x['Timestamp'].is_monotonic_increasing:
        raise ValueError('today_weather timestamps must be unique and increasing.')

    if len(x) > 1:
        dt = x['Timestamp'].diff().dropna().dt.total_seconds().to_numpy() / 60.0
        if not np.all(dt == STEP_MIN):
            raise ValueError(f'today_weather must be continuous {STEP_MIN}-minute data.')

    daylight = x['Solar_Zenith_rad'].to_numpy(float) < (np.pi / 2.0)
    daylight_pos = np.flatnonzero(daylight)
    if len(daylight_pos) == 0:
        return pd.DataFrame(columns=[
            'Issue_Timestamp',
            'First_Target_Timestamp',
            'Last_Target_Timestamp',
        ])

    target_start = pd.Timestamp(x['Timestamp'].iloc[int(daylight_pos[0])])
    ts_values = x['Timestamp'].to_numpy(dtype='datetime64[ns]')

    rows = []
    while True:
        match = np.flatnonzero(ts_values == np.datetime64(target_start))
        if len(match) == 0:
            break
        pos = int(match[0])
        if not daylight[pos]:
            break

        rows.append({
            'Issue_Timestamp': target_start - pd.Timedelta(minutes=STEP_MIN),
            'First_Target_Timestamp': target_start,
            'Last_Target_Timestamp': target_start + pd.Timedelta(minutes=(HORIZON - 1) * STEP_MIN),
        })
        target_start += pd.Timedelta(minutes=SAME_DAY_CADENCE_STEPS * STEP_MIN)

    return pd.DataFrame(rows)


deployment_model = None  # seed-42 model of the current task, set after final training


def predict_6h(future_weather_24, static_values, rated_power_w):
    """
    Predict one 24-step 6-hour window with the deployment (seed 42) model of the task.
    future_weather_24 = the 24 TARGET-time weather rows (t+15 ... t+360 for Same-Day,
    t+24h15m ... t+30h for Next-Day).
    """
    if deployment_model is None:
        raise RuntimeError('Deployment model is not loaded.')
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

    future_scaled = future_scaler.transform(raw_future).astype(np.float32)[None, :, :]
    static_scaled = static_scaler.transform(raw_static).astype(np.float32)
    pred_pu = deployment_model.predict(
        {'future_weather': future_scaled, 'static': static_scaled},
        verbose=0,
    )[0].astype(np.float32)
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
    """The deployment issue-schedule helper must reproduce the training schedule (House 11)."""
    d = houses[h]
    ts = pd.DatetimeIndex(d['timestamps'])
    _, origins = build_task_index([h], FINAL_CADENCE)
    issue_ts = ts[origins]
    # Training issues grouped by their target day.
    target_day = (issue_ts + pd.Timedelta(minutes=(TARGET_OFFSET_STEPS + 1) * STEP_MIN)).normalize()
    expected_by_day = pd.Series(issue_ts).groupby(target_day).apply(set).to_dict()
    weather = pd.DataFrame({'Timestamp': ts, 'Solar_Zenith_rad': d['future_raw'][:, FUTURE_FEATURES.index('Solar_Zenith_rad')]})
    days = ts.normalize()
    n_checked = 0
    # Skip the first and last two days of the file (file-edge truncation).
    for day in days.unique()[2:-2]:
        day_pos = np.flatnonzero(days == day)
        table = weather.iloc[day_pos[0]:day_pos[-1] + 1]
        if TASK == 'next_day':
            helper_issue = set(pd.DatetimeIndex(build_next_day_issue_schedule(table)['Issue_Timestamp_Today']))
        else:
            helper_issue = set(pd.DatetimeIndex(build_same_day_issue_schedule(table)['Issue_Timestamp']))
        if helper_issue != expected_by_day.get(day, set()):
            raise RuntimeError(f'{TASK}: deployment issue schedule differs from training on {day.date()}.')
        n_checked += 1
    print(f'{TASK}: deployment issue-schedule helper matches the training schedule on {n_checked} House-{h} days.')


def check_deployment_predictions_match_evaluation(meta, eval_pred_pu, n_windows=48):
    """predict_6h on raw weather rows must reproduce the evaluation pipeline outputs."""
    idx = np.unique(np.linspace(0, len(meta['origin']) - 1, n_windows).astype(int))
    max_diff = 0.0
    for i in idx:
        h = int(meta['house'][i])
        d = houses[h]
        fidx = meta['origin'][i] + TARGET_OFFSET_STEPS + FUTURE_OFFSETS
        wx = pd.DataFrame(d['future_raw'][fidx], columns=FUTURE_FEATURES)
        wx.insert(0, 'Timestamp', d['timestamps'][fidx])
        static_values = dict(zip(STATIC_FEATURES, d['static_raw'].tolist()))
        out, _ = predict_6h(wx, static_values, d['rated'])
        max_diff = max(max_diff, float(np.max(np.abs(out['PV_Predicted_pu'].to_numpy() - eval_pred_pu[i]))))
    if max_diff > 1e-4:
        raise RuntimeError(f'Deployment helper differs from evaluation pipeline (max |diff| = {max_diff:.2e} pu).')
    print(f'{TASK}: deployment helper reproduces evaluation outputs on {len(idx)} House-11 windows '
          f'(max |diff| = {max_diff:.2e} pu).')


def run_optuna_study():
    """One persistent, resumable Optuna study for (architecture, current task)."""
    study_name = f'{ARCH_KEY}_{TASK}'
    storage_url = f"sqlite:///{EXPERIMENT_DIR / 'optuna_studies.db'}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    # A trial left RUNNING by a disconnected session never finished: mark it FAIL so it is
    # repeated. Only finished (COMPLETE or PRUNED) trials count towards OPTUNA_TRIALS.
    for stale in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
        print(f'Optuna: interrupted trial {stale.number} marked FAIL and will be repeated.')
    n_finished = len(study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
    ))
    n_remaining = max(0, OPTUNA_TRIALS - n_finished)
    print(f'Optuna study {study_name!r} ({storage_url}): '
          f'{n_finished} finished trials, {n_remaining} remaining.')
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, gc_after_trial=True)
    return study, study_name, storage_url


def save_optuna_outputs(study):
    study.trials_dataframe().to_csv(RUN_DIR / 'optuna_trials.csv', index=False)
    with open(RUN_DIR / 'best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
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


def write_house11_tables(seed_rows, seed_planning_rows):
    per_seed_df = pd.DataFrame(seed_rows)
    pd.DataFrame(seed_rows).to_csv(RUN_DIR / 'house11_point_metrics_per_seed.csv', index=False)
    pd.DataFrame(seed_planning_rows).to_csv(RUN_DIR / 'house11_planning_metrics_per_seed.csv', index=False)
    mean_sd_table(seed_rows).to_csv(RUN_DIR / 'house11_point_metrics_mean_sd.csv')
    mean_sd_table(seed_planning_rows).to_csv(RUN_DIR / 'house11_planning_metrics_mean_sd.csv')
    print(f'\nHOUSE 11 — {ARCH_LABEL} — {TASK} — per seed'); display(per_seed_df)
    print('\nHOUSE 11 — mean ± SD over seeds'); display(mean_sd_table(seed_rows))


def write_summary(study, study_name, storage_url, best, seed_info, val_schedule_sha256, n_val_windows, extra):
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
            f'first target of each target day = its first 15-min daylight grid point; '
            f'first issue = first target - {TARGET_OFFSET_STEPS} - 1 steps; '
            f'issues every {FINAL_CADENCE} step(s) while the first target is daylight; '
            f'targets = origin + {TARGET_OFFSET_STEPS} + (1..24); '
            f'Optuna cadence {TUNE_CADENCE} step(s); final fit/evaluation cadence {FINAL_CADENCE} step(s)'
        ),
        'target_offset_steps': TARGET_OFFSET_STEPS,
        'operational_cadence_steps': FINAL_CADENCE,
        'optuna_cadence_steps': TUNE_CADENCE,
        'house11_validation_schedule_sha256': val_schedule_sha256,
        'house11_validation_windows': int(n_val_windows),
        'future_features': FUTURE_FEATURES, 'static_features': STATIC_FEATURES,
        'target': TARGET, 'target_normalization': 'PV / Array_Rated_Power_W',
        'optuna_trials': OPTUNA_TRIALS,
        'optuna_study_name': study_name,
        'optuna_storage': storage_url,
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
        **extra,
    }
    with open(RUN_DIR / 'phase1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nPHASE 1 TASK DONE — {ARCH_LABEL} — {TASK}')
    print(f"House-11 nRMSE (mean ± SD over seeds): {summary['house11_nRMSE_percent_mean']:.4f} ± "
          f"{summary['house11_nRMSE_percent_sd']:.4f} %")


def prepare_task_run():
    """Per-task outputs, schedule audit and schedule-helper check (before any training)."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(future_scaler, RUN_DIR / 'future_weather_scaler.joblib')
    joblib.dump(static_scaler, RUN_DIR / 'static_scaler.joblib')
    audit = build_schedule_audit(PRETEST_HOUSES, FINAL_CADENCE)
    audit.to_csv(RUN_DIR / 'operational_schedule_audit_houses_01_11.csv.gz', index=False)
    print(f'{TASK}: schedule audit (Houses 1-11): {len(audit):,} windows')
    print(audit[audit['House'] == VAL_HOUSES[0]].head(6).to_string(index=False))
    val_house, val_origin = build_task_index(VAL_HOUSES, FINAL_CADENCE)
    check_deployment_schedule_matches_training(VAL_HOUSES[0])
    return schedule_sha256(val_house, val_origin), len(val_origin)


def run_task():
    """Complete Phase-1 pipeline for the CURRENT task (TASK and task globals set by the loop)."""
    global deployment_model
    val_schedule_sha256, n_val_windows = prepare_task_run()

    study, study_name, storage_url = run_optuna_study()
    save_optuna_outputs(study)
    best = dict(study.best_params)
    batch = int(best['batch_size'])
    print(f'{TASK}: best params:', best)

    # Final frozen configuration retrained with seeds 42, 43, 44.
    # Early stopping / ReduceLROnPlateau / checkpoint on House-11 val_loss only.
    seed_rows, seed_planning_rows, seed_info = [], [], {}
    for s in FINAL_SEEDS:
        print(f'\n========== {ARCH_LABEL} | {TASK} | final seed {s} ==========')
        keras.utils.set_random_seed(s)

        train_seq = MultiHouseSequence(TRAIN_HOUSES, batch, FINAL_CADENCE, True)
        val_seq = MultiHouseSequence(VAL_HOUSES, batch, FINAL_CADENCE, False)
        if schedule_sha256(val_seq.sample_house, val_seq.sample_origin) != val_schedule_sha256:
            raise RuntimeError('Validation schedule changed between seeds.')
        print(f'Train samples: {len(train_seq.sample_origin):,} | Val samples: {len(val_seq.sample_origin):,}')

        model = build_model(best)
        trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
        seed_dir = MODEL_DIR / f'seed_{s}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_path = seed_dir / f'{ARCH_KEY}_{TASK}_seed{s}.keras'
        history = model.fit(
            train_seq, validation_data=val_seq, epochs=FINAL_EPOCHS, verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
                keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0),
            ],
        )
        best_epoch = int(np.argmin(history.history['val_loss']) + 1)
        pd.DataFrame(history.history).to_csv(seed_dir / 'training_history.csv', index_label='epoch0')
        model = keras.models.load_model(model_path, custom_objects={'MaskedMSE': MaskedMSE})

        plt.figure(figsize=(9, 5))
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation (House 11)')
        plt.xlabel('Epoch'); plt.ylabel('Masked MSE (per-unit)')
        plt.title(f'{ARCH_LABEL} — {TASK} — seed {s} — Training / Validation Loss')
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(seed_dir / 'training_validation_loss.png', dpi=160); plt.show(); plt.close('all')

        meta = val_seq.collect_meta()
        raw_pred_pu = model.predict(val_seq, verbose=0).astype(np.float32)
        raw_val_mse = float(np.sum(((raw_pred_pu - meta['y_pu']) ** 2) * meta['mask']) / np.sum(meta['mask']))
        res = evaluate_predictions(meta, raw_pred_pu, 'Validation_House_11')

        row = dict(res['overall']); row['Seed'] = s; seed_rows.append(row)
        prow = dict(res['planning_overall']); prow['Seed'] = s; seed_planning_rows.append(prow)
        res['by_horizon'].to_csv(seed_dir / 'validation_metrics_by_horizon.csv', index=False)
        res['window_metrics'].to_csv(seed_dir / 'validation_window_planning_metrics.csv.gz', index=False)
        np.savez_compressed(
            seed_dir / 'house11_predictions.npz',
            house=meta['house'], origin=meta['origin'],
            raw_pred_pu=raw_pred_pu, pred_pu=res['pred_pu'],
            y_pu=meta['y_pu'], mask=meta['mask'],
        )
        seed_info[s] = {
            'model_file': str(model_path.relative_to(RUN_DIR)),
            'best_epoch': best_epoch,
            'raw_masked_val_mse_pu': raw_val_mse,
            'house11_nRMSE_percent': float(res['overall']['nRMSE_percent']),
            'trainable_parameters': trainable_params,
        }

        if s == DEPLOYMENT_SEED:
            deployment_model = model
            check_deployment_predictions_match_evaluation(meta, res['pred_pu'])
        else:
            del model
        del train_seq, val_seq
        gc.collect()

    write_house11_tables(seed_rows, seed_planning_rows)
    # Deployment model = predefined seed 42, reloaded from disk.
    deployment_model = keras.models.load_model(
        MODEL_DIR / f'seed_{DEPLOYMENT_SEED}' / f'{ARCH_KEY}_{TASK}_seed{DEPLOYMENT_SEED}.keras',
        custom_objects={'MaskedMSE': MaskedMSE},
    )
    write_summary(study, study_name, storage_url, best, seed_info, val_schedule_sha256, n_val_windows,
                  {'feature_layout': 'future_weather (24, 8) + static (3,)'})


# =============================================================================
# MAIN — one execution performs SAME-DAY and then NEXT-DAY automatically.
# =============================================================================
for TASK in TASKS:
    # Task globals read by the functions above.
    TARGET_OFFSET_STEPS = TASK_TARGET_OFFSET_STEPS[TASK]
    FINAL_CADENCE = TASK_OPERATIONAL_CADENCE_STEPS[TASK]
    TUNE_CADENCE = TASK_TUNING_CADENCE_STEPS[TASK]
    RUN_DIR = PHASE1_DIR / f'{ARCH_KEY}_{TASK}'
    MODEL_DIR = RUN_DIR / 'models'

    print('\n' + '#' * 79)
    print(f'# {ARCH_LABEL} — {TASK.upper()} — offset {TARGET_OFFSET_STEPS} steps, '
          f'cadence {FINAL_CADENCE * STEP_MIN} min (Optuna {TUNE_CADENCE * STEP_MIN} min)')
    print('#' * 79)
    if (RUN_DIR / 'phase1_summary.json').exists() and not OVERWRITE_PHASE1_RUN:
        print(f'{RUN_DIR} already holds a completed {TASK} run — skipped (resume). '
              'Set OVERWRITE_PHASE1_RUN = True only to deliberately redo it.')
        continue
    run_task()
    gc.collect()

print(f'\nPHASE 1 DONE — {ARCH_LABEL} — same_day and next_day')
print('Outputs:', [str(PHASE1_DIR / f'{ARCH_KEY}_{t}') for t in TASKS])
