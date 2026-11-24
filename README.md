# DRL energy agent for the Akkar House HEMS

A simulation of the off-grid house in Akkar (15-minute steps), a rule-based safety controller (R1–R13) that is
the only path to the actuators, and four DRL agents (PPO, SAC-Discrete, DDQN, A2C) in one PyTorch framework.
The repository also has training with checkpoint selection, a test over 20 fixed grid scenarios, an 11-scenario
test suite and a results report.

| File | Content |
|---|---|
| `hems_config.py` | Every parameter, each tagged with its source: `[manual]`, `[user]`, `[default-calibrate]`, `[default]` |
| `hems_safety.py` | Safety controller (R1–R13), actuator bus, emergency / safe modes, alerts and logs, rule-based controller |
| `hems_forecast_feed.py` | Finds and loads the forecasts, inner-joins LSTM/GRU/CNN-LSTM, builds ensemble, spread and the forecast features |
| `hems_env.py` | Environment, `inverter_model()`, overload curve, grid, water, thermal, occupancy, reward |
| `hems_agents.py` | PPO, SAC-Discrete, DDQN, A2C (same 2×256 ReLU MLP) and the rule-based baseline; `agent.act(state)` |
| `hems_train.py` | Training with checkpoint selection on spring 2025 (fold 5); resumes after a disconnect |
| `hems_evaluate.py` | Test 2025-05-27..2025-12-31 (20 grid seeds) and the 11-scenario suite |
| `hems_report.py` | Tables, plots, acceptance checks, chosen agent |

Every program has a self-test (`--selftest`) that builds its own stand-in data, so it runs without your Drive files.
Every run writes its own folder with `settings.json` (all parameters used, with their sources).

## Running in Google Colab

### 0. Setup (each new Colab session)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Put the eight `.py` files in a folder on your Drive, e.g. `/content/drive/MyDrive/Mutli-House2/hems_code`, then:

```python
%cd /content/drive/MyDrive/Mutli-House2/hems_code
```

Colab already has PyTorch, numpy, pandas and matplotlib. A GPU is not needed: the networks are small and the
simulation runs on the CPU, so a CPU runtime is enough. Results go to `/content/drive/MyDrive/Mutli-House2/hems_drl`
(`OUTPUT_DIR` in `hems_config.py`).

### 1. Self-tests (about 3 minutes in total)

```python
!for m in hems_config hems_safety hems_forecast_feed hems_env hems_agents hems_train hems_evaluate hems_report; do python $m.py --selftest || break; done
```

Every program prints `ALL PASS` at the end. The results are saved in `hems_drl/selftest/`, and the report reads them for the acceptance check.

### 2. Build the forecast feeds from your data

```python
!python hems_forecast_feed.py --build
```

This step finds `cv_lstm_20261123-183844_84bacff2`, `cv_gru_…`, `cv_cnn_lstm_…` and the `final_cv_*` baseline runs
anywhere below `RESULTS_DIR` (default: `/content/drive/MyDrive/Mutli-House2`). It then checks that the dataset timestamps
are UTC, and prints the forecast coverage and error for each split. Check that train, select and test all show a coverage
close to 1.0. If a folder is reported missing, set `RESULTS_DIR` with `--set RESULTS_DIR="/content/drive/MyDrive/..."`.
The feeds are cached in `hems_drl/feeds/`.

### 3. Train (12 runs: 4 algorithms × seeds 42, 43, 44)

```python
!python hems_train.py --plan            # lists the commands
!python hems_train.py --algo ppo --seed 42
```

Repeat with `--algo sac|ddqn|a2c` and `--seed 43|44`, or run them all one after the other with
`!python hems_train.py --all`. If Colab disconnects, run the same command again: the unfinished run resumes
from its `last.pt`. Each run keeps `best.pt`, the checkpoint with the best score on the selection period
(2025-03-01..2025-05-26). The test is never used for selection.

Then train the two comparison agents with the algorithm that scored best on the selection period (the
`select_reward` column in each run's `log_select.csv`):

```python
!python hems_train.py --algo ppo --seed 42 --forecast none      # no-forecast agent
!python hems_train.py --algo ppo --seed 42 --forecast perfect   # perfect-forecast agent (upper bound)
```

For a shorter first try, add `--steps 300000`.

### 4. Test and scenario suite

```python
!python hems_evaluate.py
```

This evaluates every finished run plus the rule-based controller over the whole test period, once per grid seed,
and runs all 11 scenarios for each of them.

### 5. Report

```python
!python hems_report.py
```

The report goes to `hems_drl/report/report_<time>/`: `report.md`, CSV tables, plots and the acceptance checks.

### Dry run without your data

Every step also runs on generated stand-in data, which is useful for checking the pipeline end to end:

```python
!python hems_train.py --synthetic /content/hems_dry --algo ppo --seed 42 --steps 20000
!python hems_evaluate.py --synthetic /content/hems_dry
!python hems_report.py --synthetic /content/hems_dry
```

## Design notes

- **Safety path.** The agent only proposes an action. `SafetyController.decide()` checks R3–R13 against the
  predicted inverter load and executes the nearest safe action (fewest switch changes, preferring to switch
  things off, water before comfort). A rejection is logged and penalized, and `last_rejected` appears in the next
  state. Actuator writes need a token that only `hems_safety.py` holds. The bus has no inverter channel, and any
  inverter command raises an error (R1). A manual user command, or the water-heater program while the grid is
  present, holds the contactor closed. It is never blocked and never penalized, and the cost of that energy is
  logged but kept out of the reward.
- **Emergencies.** Communication loss, a Pi restart or an inverter fault hands control to the rule-based
  controller, and the agent runs in shadow mode. The step where an emergency starts is marked `truncated`, so
  value estimates bootstrap. Emergency steps carry no reward and `info["agent_in_control"] = False`, so no learner
  stores them; a self-test checks this for all four algorithms. A single stale sensor or an MCB trip only takes
  that load to its fail-safe state. Note: the episode itself ends only when `info["episode_end"]` is true.
- **Forecasts.** For every issue hour the feed builds the latest available forecast for the next 50 h from all
  issues made at or before that hour. Same-Day covers t+1..t+8, and the Next-Day issues of the previous hours fill
  t+9..t+31. A self-test checks there is no look-ahead. Night hours are 0 W with 0 spread.
- **Cloud cover.** The dataset has no cloud-cover column. "Cloud now" is estimated as 1 − GHI / clear-sky GHI.
  "Cloud forecast" is derived from the honest PV forecast (1 − forecast PV / clear-sky PV), so future weather does
  not leak into the state. Set `CLOUD_FORECAST_SOURCE="dataset_future"` to use the dataset's future weather
  instead, which acts as a perfect cloud forecast.
- **Outdoor temperature and humidity** for the next hours come from the dataset, as the master prompt asks.
  This works like a perfect short-range weather forecast.
- **Energy tomorrow** in the morning: forecasts reach t+31 h, so before about 13:00 part of tomorrow is not yet
  covered. Those hours use the best forecast for the same hour today.
- **Inverter model** (`inverter_model()` in `hems_env.py`) follows SUB / SNU. With the grid connected, loads take PV
  first, then the grid (30 A bypass), and the battery charges from PV plus up to 30 A from the grid (60 A in total).
  Off-grid, loads take PV and then the battery, within 4 kW. There is no export ever: surplus PV is curtailed. The
  inverter's 10 % / 70 % switching thresholds only matter in SBU mode and are not used. A trip (manual cut-off curve,
  including motor starts and the grid-loss transient) is a 30-minute blackout and the largest penalty.
- **Generator request.** It changes no power flow. It is judged after 6 h, or earlier as soon as a fridge cut, a dry
  tank or more than 2 h of discomfort happens. The reward is paid at that time.
- **Timestamps.** The dataset is UTC (checked against `Solar_Zenith_rad`). Hours, nights, schedules and calendar
  months use `Asia/Beirut` local time, and the DST switch at midnight is handled.
- **Rule-based baseline.** It uses the master prompt's rules, plus small hysteresis bands so it does not toggle every
  15 minutes (`RB_*` parameters). It never requests the generator.

## Values to calibrate later

These are the master prompt's list. Each has its own parameter in `hems_config.py`:
`HUMIDEX_DAY/NIGHT`, `GRID_MAX_EVENTS_PER_DAY` and `GRID_EVENT_MIN` (PZEM logs), `PUMP_FILL_PCT_PER_H` and
`WATER_DRAW_*` (tank logs), `BASE_*_W` (inverter load logs), `FREEZER_*` (meter), `HEATER_WINDOWS_H` (user),
`INVERTER_IDLE_W`, the thermal parameters `THERM_*`, `SOLAR_GAIN_*`, `AC_SHR`, `HUM_*` (indoor sensor logs), and
`GEN_SIZE_KVA` / `GEN_COST_PER_H` (only if a generator is added).
