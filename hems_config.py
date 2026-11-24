"""
hems_config.py - every parameter of the Akkar House HEMS DRL project, in one place.

Each parameter is a named UPPER_CASE constant with a comment that states its source:
    [manual]            value from the Felicity IVEM4024-II manual
    [user]              value given by the user (master prompt, FSolar app, measurements)
    [default-calibrate] engineering default, to be calibrated later from logs or the user
    [default]           algorithm / software design default (not a physical fact)

Other modules never hard-code a number: they read it from a Config object built here with
make_config(**overrides). Overrides must name an existing parameter (typos raise an error).

Run in Colab:
    !python hems_config.py --selftest     # checks tags, consistency, overrides
    !python hems_config.py --show         # prints every parameter with its source
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import re
import sys

# =====================================================================================
# 1. Paths and input data
# =====================================================================================
DATA_ROOT = "/content/drive/MyDrive/Mutli-House2"  # [user] Colab Drive folder with all inputs
DATASET_FILE = "my_house_model_dataset.csv"  # [user] hourly PV truth + weather, inside DATA_ROOT
RESULTS_DIR = "/content/drive/MyDrive/Mutli-House2"  # [user] <RESULTS_DIR> holding cv_* and final_cv_* folders (searched recursively)
OUTPUT_DIR = "/content/drive/MyDrive/Mutli-House2/hems_drl"  # [default] all HEMS run folders are written below this folder
CV_FOLDERS = {"lstm": "cv_lstm_20261123-183844_84bacff2", "gru": "cv_gru_20261123-190326_33b4aa02", "cnn_lstm": "cv_cnn_lstm_20261123-191905_c2524c4d"}  # [user] fold forecasts (validation files of run_*_cv1..cv5_*)
BASELINE_RUNS = {"lstm": ["final_cv_lstm", "run_lstm_20261123-183844_84bacff2"], "gru": ["final_cv_gru", "run_gru_20261123-190326_33b4aa02"], "cnn_lstm": ["final_cv_cnn_lstm", "run_cnn_lstm_20261123-191905_c2524c4d"]}  # [user] normal runs (test forecasts, live models)
FORECAST_MODELS = ["lstm", "gru", "cnn_lstm"]  # [user] ensemble members; CNN excluded (highest CV error, least stable folds)
FORECAST_FILE_PATTERN = "{kind}_{horizon}_forecasts.csv.gz"  # [user] kind = validation | test_evaluation, horizon = same_day | next_day
SEARCH_MAX_DEPTH = 6  # [default] directory depth searched below RESULTS_DIR for the folders above
DATASET_TIME_COL = "Timestamp"  # [user] checked in the dataset header
DATASET_PV_COL = "PV_DC_Power_W"  # [user] simulated true PV (equals Actual_W of the forecast files)
DATASET_TEMP_COL = "Temperature"  # [user] outdoor temperature, deg C (dataset header)
DATASET_RH_COL = "Relative Humidity"  # [user] outdoor relative humidity, % (dataset header)
DATASET_GHI_COL = "GHI"  # [user] global horizontal irradiance, W/m2 (dataset header)
DATASET_ZENITH_COL = "Solar_Zenith_rad"  # [user] solar zenith (rad) at the timestamp (dataset header)
DATASET_CLOUD_COL = ""  # [user] the dataset has no cloud-cover column; empty = derive cloudiness from GHI / clear-sky GHI
DATA_TZ = "UTC"  # [user] dataset timestamps are UTC (checked: Solar_Zenith_rad matches the sun position in UTC)
LOCAL_TZ = "Asia/Beirut"  # [user] house clock (EET/EEST) for hours, nights, schedules and calendar months
HOUSE_LAT = 34.557545  # [user] house latitude
HOUSE_LON = 36.121263  # [user] house longitude
ARRAY_RATED_W = 2700.0  # [user] 6 x 450 W PV array, one series string

# =====================================================================================
# 2. Data splits, forecast use and seeds
# =====================================================================================
SPLIT_TRAIN = ["2024-03-15", "2025-02-28"]  # [user] train the agent (CV folds 1-4 validation forecasts)
SPLIT_SELECT = ["2025-03-01", "2025-05-26"]  # [user] checkpoint selection (CV fold 5 validation forecasts)
SPLIT_TEST = ["2025-05-27", "2025-12-31"]  # [user] final test (normal runs' test forecasts)
CV_TRAIN_FOLDS = [1, 2, 3, 4]  # [user] folds whose validation forecasts feed the training split
CV_SELECT_FOLD = 5  # [user] fold whose validation forecasts feed the selection split
SEED = 42  # [user] base seed for everything
TRAIN_SEEDS = [42, 43, 44]  # [user] 3 seeds per algorithm (42 main run, 43 and 44 for stability)
TEST_SCENARIO_SEEDS = [1000 + i for i in range(20)]  # [user] 20 fixed grid-scenario seeds, the same for every agent
SELECT_SCENARIO_SEED = 7000  # [default] base seed of the fixed selection episodes
FORECAST_UPSAMPLE = "hold"  # [user] hourly -> 15 min: "hold" (each hourly value over its 4 steps) or "linear"
SD_LEADS_H = [1, 2, 3, 4, 5, 6, 7, 8]  # [user] Same-Day horizon t+1..t+8 h
ND_LEADS_H = [24, 25, 26, 27, 28, 29, 30, 31]  # [user] Next-Day horizon t+24..t+31 h
FORECAST_MODE = "ensemble"  # [user] "ensemble" (normal), "none" (no-forecast agent), "perfect" (forecast = actual)
STRESS_EPISODE_FRAC = 0.0  # [user] optional stress episodes that scale the real forecast error up (0 = off)
STRESS_MAX_SCALE = 0.0  # [default-calibrate] 0 = use the winter-fold / all-fold MAE ratio measured from the training forecasts
CLOUD_FORECAST_SOURCE = "pv_forecast"  # [default] "pv_forecast" (honest: 1 - forecast PV / clear-sky PV) or "dataset_future" (future dataset weather = perfect cloud forecast)
CLEARSKY_QUANTILE = 0.95  # [default] quantile of past PV per (day-of-year window, hour) used as clear-sky PV
CLEARSKY_DOY_WINDOW = 7  # [default] +/- days in the clear-sky PV window
FORECAST_MIN_COVERAGE = 0.9  # [default] an episode may start only where >= 90 % of its issue hours have a forecast

# =====================================================================================
# 3. Inverter (Felicity IVEM4024-II) and FSolar settings
# =====================================================================================
INV_RATED_VA = 4000.0  # [manual] rated apparent power
INV_RATED_W = 4000.0  # [manual] rated active power
INV_SURGE_FACTOR = 2.0  # [manual] surge 2x for 5 s
INV_SURGE_S = 5.0  # [manual] surge duration
OVERLOAD_CURVE = [[1.05, 10.5], [1.20, 7.5], [1.50, 5.5], [2.00, 0.2]]  # [manual] cut-off: above x rated for longer than t seconds
BYPASS_MAX_A = 30.0  # [manual] line mode (bypass) maximum current
LOW_BATT_ALARM_V = [22.5, 22.0]  # [manual] low-battery alarm (informational; SOC rules protect the battery)
LOW_BATT_SHUTDOWN_V = [21.5, 21.0]  # [manual] low-battery shutdown (modelled as SOC = 0 %)
PV_MAX_INPUT_W = 6000.0  # [manual] PV input maximum power
PV_VOC_MAX_V = 500.0  # [manual] PV open-circuit voltage maximum
PV_MAX_A = 27.0  # [manual] PV input maximum current
OUTPUT_PRIORITY = "SUB"  # [user] FSolar program 04: solar first, then utility, then battery
CHARGE_PRIORITY = "SNU"  # [user] solar and utility charge the battery, also in daytime
MAX_CHARGE_A = 60.0  # [user] total charge current limit (PV + grid)
MAX_GRID_CHARGE_A = 30.0  # [user] grid charge current limit (about 0.8 kW)
DISCHARGE_CUTOFF_SOC = 0.0  # [user] inverter does not protect the battery; the safety controller does
LOW_SOC_TO_CHARGE = 10.0  # [user] inverter switching threshold (only used in SBU mode; not used with SUB)
HIGH_SOC_TO_DISCHARGE = 70.0  # [user] inverter switching threshold (only used in SBU mode; not used with SUB)
OVERLOAD_RESTART = False  # [user] disabled: an overload is a full blackout until a manual reset
INV_EFF = 0.93  # [user] inverter efficiency (PV/battery DC -> AC)
GRID_CHARGER_EFF = 0.93  # [default-calibrate] AC -> DC efficiency of grid charging (same power stage)
INVERTER_IDLE_W = 50.0  # [user] idle draw of the inverter (to confirm)
GRID_V = 230.0  # [user] nominal AC voltage used to convert currents to power

# =====================================================================================
# 4. Battery
# =====================================================================================
BATT_CAP_KWH = 5.0  # [user] lithium, 5 kWh
BATT_NOM_V = 24.0  # [user] nominal voltage
BATT_CHARGE_V = 26.7  # [default-calibrate] charge voltage used to turn currents into power (30 A -> about 0.8 kW, as stated)
ETA_CHARGE = 0.95  # [user] charge efficiency
ETA_DISCHARGE = 0.95  # [user] discharge efficiency
BATT_SOH = 1.0  # [default-calibrate] state of health (read from the inverter in operation)
BATT_SOH_RANGE = [0.92, 1.0]  # [default-calibrate] SOH drawn per training episode (domain randomization)
BATT_MAX_DISCHARGE_A = 200.0  # [default-calibrate] BMS discharge limit (not binding below 4 kW)
BATT_R_INT_OHM = 0.02  # [default-calibrate] internal resistance for the voltage feature
BATT_OCV_TABLE = [[0, 22.0], [5, 24.4], [10, 25.4], [20, 25.8], [40, 26.0], [60, 26.2], [80, 26.4], [95, 26.8], [100, 27.6]]  # [default-calibrate] 8S LiFePO4 rest voltage vs SOC %
LV_RESTART_SOC = 5.0  # [default-calibrate] after a low-voltage shutdown the inverter restarts once PV lifts SOC to this %

# =====================================================================================
# 5. Loads (230 V)
# =====================================================================================
PUMP_RUN_A = 10.0  # [user] well pump 1.5 hp, 10 A running (about 2.3 kVA)
PUMP_PF = 0.8  # [default-calibrate] pump power factor (energy meter)
PUMP_START_FACTOR = 2.5  # [default-calibrate] start current / running current (must stay under the 2x surge for the inverter to start it today)
PUMP_START_S = 1.0  # [default-calibrate] pump start transient duration
AC_RUN_A = 6.0  # [user] air conditioner running current
AC_RATED_A = 8.5  # [user] air conditioner rated current (used for conservative load predictions)
AC_START_A = 23.0  # [user] air conditioner start current
AC_START_S = 3.0  # [user] "a few seconds"
AC_INPUT_W = 1300.0  # [user] air conditioner input power
AC_COOLING_W = 3500.0  # [user] 12000 BTU/h removes about 3.5 kW of heat
FRIDGE_RUN_A = [0.5, 1.2]  # [user] LG inverter fridge running current range
FRIDGE_START_A = 3.0  # [user] fridge start current (upper value)
FRIDGE_AVG_W = 55.0  # [default-calibrate] average fridge power (about 1.3 kWh/day)
FREEZER_RUN_A = 0.7  # [default-calibrate] freezer running current (measured later)
FREEZER_AVG_W = 40.0  # [default-calibrate] average freezer power (measured later)
FREEZER_USER_ON = True  # [user] freezer is switched manually by the user (on by default in simulation)
HEATER_A = 20.0  # [user] water heater counted as 20 A
HEATER_W = 4600.0  # [user] water heater power (above the inverter's 4 kW: grid only)
HEATER_WINDOWS_H = [[6, 7], [18, 19]]  # [default-calibrate] programmed schedule (local hours), to define with the user
HEATER_FORCES_CONTACTOR = True  # [default] an active heater window with grid present acts like a user command (contactor held closed, cost not charged to the agent)
BASE_DAY_W = [30.0, 50.0]  # [user] base load in the day and at night (lighting off, router, TV standby)
BASE_EVENING_W = [150.0, 300.0]  # [user] base load in the evening
EVENING_HOURS = [17, 23]  # [default-calibrate] local hours counted as evening for the base load
BASE_STEP_NOISE = 0.10  # [default-calibrate] +/- relative noise of the base load per step
BASE_PF = 0.9  # [default-calibrate] power factor of the base load

# =====================================================================================
# 6. Safety controller rules (R1-R13) and emergencies
# =====================================================================================
SOC_RESTRICT = 35.0  # [user] R3: on battery or PV, SOC below 35 % -> no pump and no AC
SOC_CUT = 30.0  # [user] R4: on battery, SOC below 30 % -> fridge, freezer and AC disconnected
SOC_CUT_HYST = 3.0  # [user] R4: back on at SOC_CUT + 3 or when the grid is connected
INV_LIMIT_W = 3600.0  # [user] R6: 90 % of 4 kW; reject the action that adds load above it
BYPASS_LIMIT_FRAC = 0.9  # [default] R6 in line mode: reject the action that adds load above 90 % of the 30 A bypass
START_STAGGER_S = 30.0  # [user] R7: pump and AC starts at least 30 s apart
AC_MIN_OFF_MIN = 3.0  # [user] R8: compressor off at least 3 min before a restart
TANK_FULL = 95.0  # [user] R10: pump stops when the tank is full
PUMP_MAX_H = 10.0  # [user] R10: pump stops after 10 h of continuous running
PUMP_REST_MIN = 30.0  # [default-calibrate] R10: rest after a 10 h run before the pump may restart
STALE_S = 120.0  # [user] R12: data older than 2 min -> that load goes to its fail-safe state
RECOVERY_MIN = 10.0  # [user] return to the agent only after all readings are valid for 10 min
TRIP_OUTAGE_MIN = 30.0  # [user] an inverter trip = blackout, all loads off for 30 min (manual reset)
GRID_LOSS_REACTION_S = 2.0  # [default-calibrate] time for the Pi to shed load after the grid drops
WATCHDOG_S = 120.0  # [user] ESP32 local watchdog: no heartbeat for 2 min -> load to fail-safe on its own
FAILSAFE_STATE = {"pump": 0, "ac": 0, "fridge": 1}  # [user] fail-safe states (R12)

# =====================================================================================
# 7. Grid, tariff and user actions
# =====================================================================================
GRID_MAX_EVENTS_PER_DAY = 4.0  # [user] arrival rate drawn per episode from U(0, 4) events/day (Poisson arrivals)
GRID_EVENT_MIN = [60.0, 120.0]  # [user] each grid event lasts U(60, 120) minutes
GRID_RATE_BLOCK_DAYS = 3  # [default] in long test runs the arrival rate is redrawn every 3 days (same as an episode)
GRID_LAST_SEEN_INIT_H = [0.0, 48.0]  # [default-calibrate] hours since the grid was last seen at episode start
TARIFF_TIER_KWH = 100.0  # [user] block tariff threshold per calendar month
TARIFF_LOW = 0.10  # [user] $/kWh up to 100 kWh
TARIFF_HIGH = 0.27  # [user] $/kWh above 100 kWh
MONTH_KWH_INIT_PER_DAY = [0.0, 4.0]  # [default-calibrate] month-to-date grid kWh at episode start = U(range) x elapsed days
USER_MANUAL_GRID_PROB = 0.05  # [default-calibrate] probability that the user connects the grid by hand for a grid event
USER_MANUAL_GRID_MAX_H = 3.0  # [default-calibrate] the manual connection lasts until the event ends (at most this long)

# =====================================================================================
# 8. Water
# =====================================================================================
TANK_MIN = 10.0  # [user] below 10 % is a water outage
PUMP_FILL_PCT_PER_H = 12.0  # [default-calibrate] pump fill rate in % of the tank per hour (fit from tank logs)
WATER_DRAW_PCT_PER_DAY = 35.0  # [default-calibrate] household draw in % of the tank per day (fit from tank logs)
WATER_DRAW_RANDOM = [0.7, 1.3]  # [default-calibrate] per-episode factor on the daily draw
WATER_SUMMER_FACTOR = 1.3  # [default-calibrate] more water use in June-September (garden)
WATER_AWAY_FACTOR = 0.15  # [default-calibrate] draw while nobody is home, relative to normal
WATER_PROFILE = [[0, 6, 0.05], [6, 9, 0.35], [9, 17, 0.18], [17, 22, 0.37], [22, 24, 0.05]]  # [default-calibrate] share of the daily draw per local-hour band (morning and evening peaks)

# =====================================================================================
# 9. Comfort, thermal model and occupancy
# =====================================================================================
HUMIDEX_DAY = 31.0  # [user] AC needed above this humidex in the day
HUMIDEX_NIGHT = 30.0  # [user] AC needed above this humidex in sleeping hours
NIGHT_HOURS = [22, 7]  # [user] sleeping hours 22:00 to 07:00 (local)
LONG_DISCOMFORT_H = 2.0  # [user] extra penalty for each step after 2 h of continuous discomfort
THERM_UA_W_K = 150.0  # [default-calibrate] 1R1C: heat loss coefficient of the cooled zone (W/K)
THERM_C_J_K = 2.0e6  # [default-calibrate] 1R1C: heat capacity of the cooled zone (J/K)
SOLAR_GAIN_PER_PV_W = 0.15  # [default-calibrate] solar heat gain in W per W of PV DC power (irradiance proxy)
INTERNAL_GAIN_HOME_W = 200.0  # [default-calibrate] people and appliances when home
INTERNAL_GAIN_AWAY_W = 80.0  # [default-calibrate] appliances when away
AC_SHR = 0.75  # [default-calibrate] sensible heat ratio of the AC (rest removes moisture)
HUM_VENT_TAU_H = 3.0  # [default-calibrate] indoor vapour pressure relaxes to outdoor with this time constant
HUM_GAIN_HOME_HPA_H = 0.3  # [default-calibrate] moisture from occupants (hPa/h)
AC_COIL_DEW_C = 12.0  # [default-calibrate] AC coil dew point; the AC pulls indoor vapour pressure towards it
AC_DRY_TAU_H = 1.5  # [default-calibrate] time constant of AC dehumidification
INDOOR_INIT_OFFSET_C = [0.0, 3.0]  # [default-calibrate] start indoor T = outdoor daily mean + U(range)
OCC_OUTINGS_PER_DAY = 1.0  # [default-calibrate] random away periods per day (Poisson)
OCC_OUTING_H = [1.0, 5.0]  # [default-calibrate] duration of an away period
OCC_WORKDAY_PROB = 0.3  # [default-calibrate] probability that a day has an 08-16 absence
OCC_WORK_HOURS = [8, 16]  # [default-calibrate] local hours of that absence
OCC_AWAY_EPISODE_PROB = 0.05  # [default-calibrate] probability that nobody is home for the whole episode
OCC_UNKNOWN_PER_DAY = 0.5  # [default-calibrate] WiFi and GPS both silent: unknown periods per day
OCC_UNKNOWN_MIN = [15.0, 180.0]  # [default-calibrate] duration of an unknown period (last known state is kept)

# =====================================================================================
# 10. Generator (disabled, advisory)
# =====================================================================================
GEN_ENABLED = False  # [user] no generator now; the request is advisory only
GEN_SIZE_KVA = 0.0  # [user] set later if a generator is added
GEN_COST_PER_H = 0.0  # [user] fuel cost per hour, set later
GEN_LOOKAHEAD_H = 6.0  # [user] request judged in hindsight over the next 6 h of the same episode
GEN_COMFORT_VIOLATION_H = 2.0  # [user] a comfort violation longer than 2 h while home makes a request right

# =====================================================================================
# 11. Reward (order: safety > water > fridge > comfort > cost)
# =====================================================================================
W_TRIP = -500.0  # [user] inverter trip (blackout), per event
W_REJECT = -2.0  # [user] rejected action, per step
W_WATER_OUTAGE = -30.0  # [user] tank below TANK_MIN, per step
W_FRIDGE_CUT = -20.0  # [user] fridge or freezer cut, per step (counted once if both are cut)
W_DISCOMFORT = -1.0  # [user] per deg C of humidex above the threshold, per step, when someone is home
W_LONG_DISCOMFORT = -2.0  # [user] extra per step after 2 h of continuous discomfort
W_LOW_SOC = -0.1  # [user] per % of SOC below LOW_SOC_MARGIN, per step
LOW_SOC_MARGIN = 40.0  # [user] low-battery margin threshold
W_COST = -10.0  # [user] per $ of grid energy at the marginal tariff
W_WASTED_PV = -0.5  # [user] per kWh of curtailed PV while the tank is not full
W_SWITCH = -0.2  # [user] per contactor, pump or AC toggle
W_GEN_RIGHT_PER_H = 5.0  # [user] generator request right in hindsight, per request hour
W_GEN_WRONG_BASE_PER_H = -2.0  # [user] generator request wrong in hindsight: -(GEN_COST_PER_H x 10) - 2 per request hour
GEN_WRONG_COST_MULT = 10.0  # [user] multiplier of GEN_COST_PER_H in the wrong-request penalty

# =====================================================================================
# 12. Environment and episodes
# =====================================================================================
STEP_MIN = 15  # [user] simulation step
EPISODE_DAYS = 3  # [user] 3-day episodes so the battery carries over nights
START_SOC_RANGE = [40.0, 90.0]  # [user] random start SOC
START_TANK_RANGE = [30.0, 90.0]  # [user] random start tank level
HARD_EPISODE_FRAC = 0.10  # [user] episodes that start in hard conditions on purpose
EMERGENCY_EPISODE_FRAC = 0.02  # [user] episodes with a simulated emergency (rule-based control for a random time)
EMERGENCY_TYPES = {"comm_loss": 0.3, "inverter_fault": 0.2, "pi_restart": 0.2, "sensor_stale": 0.15, "mcb_trip": 0.15}  # [default-calibrate] relative frequency of emergency types
EMERGENCY_DURATION_MIN = [15.0, 360.0]  # [default-calibrate] duration of a simulated emergency
USER_CONFIRM_DELAY_MIN = [15.0, 120.0]  # [default-calibrate] time until the user confirms after an inverter fault
USE_ACTION_MASK = False  # [user] ablation flag: mask invalid actions instead of rejecting and penalizing them
MAX_HOURS_FEATURE = 168.0  # [default] cap of "hours since" features (one week)

# =====================================================================================
# 13. Algorithms, training and evaluation
# =====================================================================================
ALGOS = ["ppo", "sac", "ddqn", "a2c"]  # [user] PPO, SAC-Discrete, DDQN, A2C
TRAIN_STEPS = 2_000_000  # [user] environment steps per training
EVAL_EVERY = 100_000  # [default] steps between checkpoint evaluations on the selection split
HIDDEN = [256, 256]  # [user] same MLP for all: 2 x 256, ReLU
GAMMA = 0.99  # [default] discount per 15-min step (horizon about 25 h)
REWARD_SCALE = 0.05  # [default] reward multiplier inside the learners only (ranking unchanged)
OBS_NORM_STEPS = 20_000  # [default] steps of training-split rollouts used to fit the state normalization
OBS_CLIP = 10.0  # [default] clip of normalized features
DEVICE = "auto"  # [default] "auto" = cuda if available else cpu
TORCH_THREADS = 0  # [default] 0 = leave PyTorch default
PPO = {"n_envs": 8, "n_steps": 256, "lr": 3e-4, "epochs": 10, "minibatch": 256, "clip": 0.2, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5}  # [default] PPO hyperparameters
A2C = {"n_envs": 8, "n_steps": 16, "lr": 3e-4, "gae_lambda": 1.0, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5}  # [default] A2C hyperparameters
DDQN = {"n_envs": 4, "lr": 1e-4, "buffer": 300_000, "batch": 256, "learning_starts": 20_000, "train_every": 1, "grad_steps": 1, "target_tau": 0.005, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.2, "max_grad_norm": 10.0}  # [default] DDQN hyperparameters (train_every in vector steps)
SAC = {"n_envs": 4, "lr": 3e-4, "buffer": 300_000, "batch": 256, "learning_starts": 20_000, "train_every": 1, "grad_steps": 1, "tau": 0.005, "target_entropy_ratio": 0.5, "max_grad_norm": 10.0}  # [default] SAC-Discrete hyperparameters
SAVE_REPLAY_BUFFER = False  # [default] save the replay buffer on checkpoints (large; resume refills it otherwise)
SELECT_EPISODES_MAX = 29  # [default] fixed 3-day selection episodes tiled over the selection period
TEST_INIT_SOC = 60.0  # [default] start SOC of each continuous test run
TEST_INIT_TANK = 60.0  # [default] start tank level of each continuous test run

# =====================================================================================
# 14. Rule-based baseline (and emergency controller)
# =====================================================================================
RB_CONTACTOR_SOC = 50.0  # [user] contactor closed whenever the grid is present and SOC is below 50 %
RB_AC_MIN_SOC = 40.0  # [default-calibrate] "SOC allows": AC on battery/PV only above this SOC
RB_PUMP_START_TANK = 85.0  # [default-calibrate] pump only while the tank is below this level
RB_PV_SURPLUS_FRAC = 0.9  # [default-calibrate] PV surplus must cover this fraction of the pump power
RB_FULL_SOC = 90.0  # [default-calibrate] battery nearly full: PV would be curtailed, so a smaller surplus is enough to pump
RB_PV_SURPLUS_FRAC_FULL = 0.5  # [default-calibrate] surplus fraction of the pump power needed when SOC >= RB_FULL_SOC
RB_CONTACTOR_HYST = 5.0  # [default] once closed, the contactor stays closed until SOC >= RB_CONTACTOR_SOC + 5 (no chattering)
RB_TANK_GRID = 40.0  # [default-calibrate] contactor also closed while the grid is present and the tank is below 40 % (pump on the grid)
RB_AC_HYST = 1.0  # [default] AC keeps running until the humidex is 1 deg C below the threshold

PARAM_SOURCES = ("manual", "user", "default-calibrate", "default")


# =====================================================================================
# Config object
# =====================================================================================
def _param_names():
    mod = sys.modules[__name__]
    return [k for k in vars(mod) if re.fullmatch(r"[A-Z][A-Z0-9_]*", k) and k != "PARAM_SOURCES"]


class Config:
    """Read-only bag of parameters. Access as cfg.SOC_CUT; cfg.to_dict() for saving."""

    def __init__(self, values: dict):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f"unknown parameter {name!r}") from None

    def __setattr__(self, name, value):
        raise AttributeError("Config is read-only; use cfg.replace(...) or make_config(...)")

    def replace(self, **overrides) -> "Config":
        values = dict(self._values)
        for k, v in overrides.items():
            if k not in values:
                raise KeyError(f"unknown parameter {k!r}")
            values[k] = v
        return Config(values)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._values, default=str))

    def save(self, path: str, extra: dict | None = None):
        d = {"parameters": self.to_dict(), "sources": param_sources()}
        if extra:
            d["run"] = extra
        with open(path, "w") as f:
            json.dump(d, f, indent=1, default=str)

    # derived quantities used in several modules --------------------------------------
    @property
    def steps_per_hour(self):
        return 60 // self._values["STEP_MIN"]

    @property
    def dt_h(self):
        return self._values["STEP_MIN"] / 60.0

    @property
    def episode_steps(self):
        return int(self._values["EPISODE_DAYS"] * 24 * 60 // self._values["STEP_MIN"])


def make_config(**overrides) -> Config:
    mod = sys.modules[__name__]
    values = {k: json.loads(json.dumps(getattr(mod, k))) for k in _param_names()}
    for k, v in overrides.items():
        if k not in values:
            raise KeyError(f"unknown parameter {k!r}")
        values[k] = v
    return Config(values)


def config_from_file(path: str) -> Config:
    """Rebuild the Config saved in a run folder (settings.json)."""
    with open(path) as f:
        d = json.load(f)
    params = d.get("parameters", d)
    base = make_config()
    known = {k: v for k, v in params.items() if k in base._values}
    return base.replace(**known)


def param_sources() -> dict:
    """Parameter name -> source tag, parsed from this file's comments."""
    src = {}
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^([A-Z][A-Z0-9_]*)\s*=.*#\s*\[([a-z\-]+)\]", line)
            if m:
                src[m.group(1)] = m.group(2)
    return src


# =====================================================================================
# Helpers shared by all programs
# =====================================================================================
def set_global_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def new_run_dir(cfg: Config, kind: str, name: str, base: str | None = None) -> str:
    """Create OUTPUT_DIR/<kind>/<name>_<timestamp>/ and save the settings used."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = base if base is not None else cfg.OUTPUT_DIR
    path = os.path.join(root, kind, f"{name}_{stamp}")
    i = 1
    while os.path.exists(path):
        path = os.path.join(root, kind, f"{name}_{stamp}_{i}")
        i += 1
    os.makedirs(path)
    cfg.save(os.path.join(path, "settings.json"), extra={"kind": kind, "name": name, "created": stamp, "argv": sys.argv})
    return path


_SYNTHETIC_SPLITS = {"SPLIT_TRAIN": ["2024-06-01", "2024-06-30"], "SPLIT_SELECT": ["2024-07-01", "2024-07-09"],
                    "SPLIT_TEST": ["2024-07-10", "2024-07-24"]}


def synthetic_overrides(base_dir: str) -> dict:
    """Overrides for a dry run on generated stand-in data (no Drive data needed)."""
    base_dir = os.path.abspath(base_dir)
    d = dict(_SYNTHETIC_SPLITS)
    d.update(DATA_ROOT=os.path.join(base_dir, "data"), RESULTS_DIR=os.path.join(base_dir, "data"),
             OUTPUT_DIR=os.path.join(base_dir, "out"), TEST_SCENARIO_SEEDS=[1000, 1001, 1002],
             SELECT_EPISODES_MAX=3, OBS_NORM_STEPS=2000, TRAIN_STEPS=20_000, EVAL_EVERY=10_000)
    ddqn = dict(DDQN)
    ddqn.update(buffer=50_000, learning_starts=1000)
    sac = dict(SAC)
    sac.update(buffer=50_000, learning_starts=1000)
    d.update(DDQN=ddqn, SAC=sac)
    return d


def parse_set_args(pairs) -> dict:
    """["KEY=VALUE", ...] -> {KEY: value}; VALUE is parsed as JSON when possible."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--set expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override parameters, e.g. --set TRAIN_STEPS=500000")
    ap.add_argument("--synthetic", default=None, metavar="DIR", help="dry run on generated stand-in data in DIR (no Drive data needed)")


def config_from_args(a) -> Config:
    over = {}
    if getattr(a, "synthetic", None):
        over.update(synthetic_overrides(a.synthetic))
    over.update(parse_set_args(getattr(a, "set", [])))
    return make_config(**over)


def humidex(t_c, rh_pct):
    """Humidex from temperature (deg C) and relative humidity (%). Works on floats and numpy arrays."""
    import math
    try:
        import numpy as np
        if isinstance(t_c, np.ndarray) or isinstance(rh_pct, np.ndarray):
            e = np.asarray(rh_pct) / 100.0 * 6.112 * np.exp(17.67 * np.asarray(t_c) / (np.asarray(t_c) + 243.5))
            return np.asarray(t_c) + 0.5555 * (e - 10.0)
    except ImportError:
        pass
    e = rh_pct / 100.0 * 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))
    return t_c + 0.5555 * (e - 10.0)


class SelfTest:
    """Tiny check collector used by every self-test section."""

    def __init__(self, name):
        self.name, self.passed, self.failed = name, [], []

    def check(self, cond, label, detail=""):
        (self.passed if cond else self.failed).append(label + (f" ({detail})" if detail and not cond else ""))
        print(("  PASS  " if cond else "  FAIL  ") + label + (f"  [{detail}]" if detail else ""))
        return bool(cond)

    def summary(self, out_dir: str | None = None) -> bool:
        ok = not self.failed
        print(f"\n{self.name}: {len(self.passed)} passed, {len(self.failed)} failed -> {'ALL PASS' if ok else 'FAILED'}")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "selftest_result.json"), "w") as f:
                json.dump({"name": self.name, "passed": self.passed, "failed": self.failed, "ok": ok}, f, indent=1)
        return ok


# =====================================================================================
# Self-test
# =====================================================================================
def selftest(out_root: str | None = None) -> bool:
    T = SelfTest("hems_config")
    names = _param_names()
    sources = param_sources()
    missing = [n for n in names if n not in sources]
    T.check(not missing, "every parameter has a source tag", str(missing))
    bad = [n for n, s in sources.items() if s not in PARAM_SOURCES]
    T.check(not bad, "source tags are manual/user/default-calibrate/default", str(bad))
    cfg = make_config()
    T.check(cfg.SEED == 42 and cfg.TRAIN_SEEDS == [42, 43, 44], "base seed 42, seeds 42/43/44")
    T.check(abs(cfg.INV_LIMIT_W - 0.9 * cfg.INV_RATED_W) < 1e-9, "INV_LIMIT_W = 90 % of 4 kW")
    T.check(cfg.SOC_CUT < cfg.SOC_CUT + cfg.SOC_CUT_HYST <= cfg.SOC_RESTRICT < cfg.LOW_SOC_MARGIN, "SOC_CUT < cut+hyst <= SOC_RESTRICT < LOW_SOC_MARGIN")
    T.check(cfg.TANK_MIN < cfg.RB_PUMP_START_TANK < cfg.TANK_FULL <= 100, "TANK_MIN < RB start < TANK_FULL")
    d = lambda s: _dt.date.fromisoformat(s)
    T.check(d(cfg.SPLIT_TRAIN[0]) < d(cfg.SPLIT_TRAIN[1]) < d(cfg.SPLIT_SELECT[0]) < d(cfg.SPLIT_SELECT[1]) < d(cfg.SPLIT_TEST[0]) < d(cfg.SPLIT_TEST[1]),
            "train < select < test, no overlap")
    T.check(len(cfg.TEST_SCENARIO_SEEDS) == 20 and len(set(cfg.TEST_SCENARIO_SEEDS)) == 20, "20 distinct test scenario seeds")
    T.check(cfg.episode_steps == 288 and cfg.steps_per_hour == 4, "3-day episodes of 288 steps of 15 min")
    T.check(cfg.FORECAST_MODELS == ["lstm", "gru", "cnn_lstm"], "ensemble = LSTM, GRU, CNN-LSTM (CNN excluded)")
    # Worked check of the reward order (Section 6 of the master prompt)
    disc = 4 * (cfg.W_DISCOMFORT * 2.0)
    ac_cost = 1.3 * cfg.TARIFF_HIGH * cfg.W_COST
    fridge = 4 * cfg.W_FRIDGE_CUT
    water = 4 * cfg.W_WATER_OUTAGE
    T.check(abs(disc - (-8)) < 1e-9 and abs(ac_cost - (-3.51)) < 1e-9 and fridge == -80 and water == -120,
            "worked check numbers -8, -3.5, -80, -120", f"{disc} {ac_cost} {fridge} {water}")
    T.check(cfg.W_TRIP < water < fridge < disc < ac_cost < 0, "trip < water < fridge < comfort < cost")
    T.check(abs(humidex(25, 50) - 28.2) < 0.06 and abs(humidex(25, 70) - 31.8) < 0.06 and abs(humidex(27, 50) - 31.3) < 0.06
            and abs(humidex(28, 30) - 28.7) < 0.06 and abs(humidex(28, 60) - 35.0) < 0.06, "humidex reference table")
    # overrides
    c2 = make_config(SOC_CUT=25.0)
    T.check(c2.SOC_CUT == 25.0 and cfg.SOC_CUT == 30.0, "override changes only the new config")
    try:
        make_config(SOC_CUTT=1)
        T.check(False, "unknown override raises")
    except KeyError:
        T.check(True, "unknown override raises")
    try:
        cfg.SOC_CUT = 1
        T.check(False, "config is read-only")
    except AttributeError:
        T.check(True, "config is read-only")
    T.check(json.loads(json.dumps(cfg.to_dict()))["SOC_CUT"] == 30.0, "config is JSON serializable")
    out = None
    if out_root:
        out = new_run_dir(cfg, "selftest", "hems_config", base=out_root)
        T.check(os.path.exists(os.path.join(out, "settings.json")), "run folder with settings.json")
        c3 = config_from_file(os.path.join(out, "settings.json"))
        T.check(c3.to_dict() == cfg.to_dict(), "settings.json round trip")
    return T.summary(out)


def _default_selftest_root():
    cfg = make_config()
    if os.path.isdir(os.path.dirname(cfg.OUTPUT_DIR)):
        return cfg.OUTPUT_DIR
    return os.path.join(os.getcwd(), "hems_drl_local")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default=None, help="root folder for the self-test run folder")
    a = ap.parse_args()
    if a.show:
        src = param_sources()
        for k, v in make_config().to_dict().items():
            print(f"{k:28s} [{src.get(k, '?'):17s}] {v}")
    if a.selftest or not a.show:
        sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
