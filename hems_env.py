"""
hems_env.py - 15-minute simulation of the Akkar house (Gymnasium-style API, no gymnasium dependency).

    env = HEMSEnv(cfg, split_data, mode="train")
    obs, info = env.reset(seed)
    obs, reward, terminated, truncated, info = env.step(action)     # action in 0..15

Models
  inverter_model()   replaceable power-flow function: SUB output priority, SNU charging, 60 A / 30 A charge limits,
                     30 A bypass, 4 kW inverter, no export ever (PV is curtailed instead). Validate later against logs.
  overload_trips()   manual cut-off curve (105 %/10.5 s, 120 %/7.5 s, 150 %/5.5 s, 200 %/0.2 s) incl. motor starts
                     and the grid-loss transient; a trip = blackout for TRIP_OUTAGE_MIN and the largest penalty.
  grid               Poisson arrivals at any hour, rate drawn per episode from U(0, 4)/day, events 60-120 min.
  water              tank %, pump fill rate, household draw with morning/evening peaks, TANK_MIN outage.
  thermal            1R1C indoor temperature + vapour pressure, AC removes ~3.5 kW and dries the air; humidex.
  occupancy          random away periods, WiFi/GPS "unknown" periods keep the last known state.
  battery            5 kWh, 0.95/0.95 efficiencies, inverter 0.93, idle 50 W, SOH, low-voltage shutdown at 0 %.
  emergencies        comm loss, stale sensor, inverter fault, MCB trip, Pi restart: rule-based control, agent in
                     shadow mode, the step where an emergency starts is truncated (bootstrap), emergency steps carry
                     no reward and info["agent_in_control"] = False so learners never store them.

Every action goes through hems_safety.SafetyController; the environment cannot write the actuators itself.
Note: truncated=True also marks the start of an emergency; the episode itself ends only when
info["episode_end"] is True (VecEnv resets on that flag).

Run in Colab:
    !python hems_env.py --selftest
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np

import hems_forecast_feed as ff
from hems_config import Config, SelfTest, humidex, make_config, new_run_dir, synthetic_overrides
from hems_safety import (ActuatorBus, Measurements, SafetyContext, SafetyController, decode_action, encode_action,
                         rule_based_action, valid_action_mask, N_ACTIONS)

# =====================================================================================
# State features (raw units; normalization is done by the agents from training statistics)
# =====================================================================================
FEATURE_GROUPS = {
    "time": ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos", "season_winter", "season_spring",
             "season_summer", "season_autumn", "is_day", "h_to_sunrise", "h_to_sunset"],
    "battery": ["soc_pct", "batt_power_w", "soc_chg_15m", "soc_chg_1h", "batt_voltage_v", "soh", "h_to_soc35"],
    "pv_forecast": ["pv_now_w", "pv_last_hour_w"] + list(ff.FC_NAMES) + ["cloud_now"],
    "grid": ["grid_present", "contactor_closed", "h_since_grid_arrived", "h_since_grid_seen", "grid_kwh_month", "tariff_high"],
    "water": ["tank_pct", "tank_chg_per_h", "pump_on", "pump_run_h", "pump_h_today"],
    "comfort": ["t_in", "rh_in", "humidex_in", "humidex_thr", "humidex_excess", "t_out", "rh_out", "t_out_1_3h", "t_out_4_8h",
                "rh_out_1_3h", "rh_out_4_8h", "ac_on", "ac_run_h", "occ_home", "occ_away", "occ_unknown", "occ_age_h",
                "occ_effective_home", "discomfort_h"],
    "loads": ["fridge_on", "freezer_on", "heater_on", "base_load_w", "total_load_w", "inverter_load_pct"],
    "safety": ["last_rejected", "safe_mode", "emergency_mode", "inverter_fault", "pump_unavailable", "ac_unavailable",
               "user_override_contactor", "fridge_cut_active", "lv_shutdown"],
}
FEATURE_NAMES = [f for g in FEATURE_GROUPS.values() for f in g]
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)
FC_IDX = np.array([IDX[n] for n in ff.FC_NAMES] + [IDX["cloud_now"]])  # zeroed for the no-forecast agent

REWARD_TERMS = ["trip", "reject", "water", "fridge", "discomfort", "long_discomfort", "low_soc", "cost", "wasted_pv", "switch", "generator"]


def obs_to_dict(obs) -> dict:
    return {n: float(obs[i]) for i, n in enumerate(FEATURE_NAMES)}


# =====================================================================================
# Physics
# =====================================================================================
def e_sat_hpa(t_c):
    return 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))


def inverter_model(pv_avail_w, load_w, load_va, grid_ok, contactor_closed, soc_pct, cap_wh, dt_h, cfg: Config,
                   output_off=False, lv_shutdown=False) -> dict:
    """Power flows over one step. SUB: loads take PV first, then the grid (bypass, max 30 A); the battery charges
    from PV surplus plus up to 30 A from the grid (60 A total). Contactor open or no grid: loads take PV, then the
    battery, within the 4 kW limit. Never exports: surplus PV is curtailed. Battery DC power is + when charging."""
    eta_i = cfg.INV_EFF
    pv = min(max(pv_avail_w, 0.0), cfg.PV_MAX_INPUT_W)
    max_chg = cfg.MAX_CHARGE_A * cfg.BATT_CHARGE_V
    max_grid = cfg.MAX_GRID_CHARGE_A * cfg.BATT_CHARGE_V
    e = soc_pct / 100.0 * cap_wh
    head_w = max(0.0, cap_wh - e) / (cfg.ETA_CHARGE * dt_h)
    avail_w = max(0.0, e) * cfg.ETA_DISCHARGE / dt_h
    r = {"mode": "battery", "pv_to_load_w": 0.0, "pv_to_batt_w": 0.0, "curtailed_w": 0.0, "grid_to_load_w": 0.0,
         "grid_to_batt_ac_w": 0.0, "grid_import_w": 0.0, "batt_dc_w": 0.0, "soc_delta_wh": 0.0, "served": 1.0,
         "inv_out_va": 0.0, "inverter_load_pct": 0.0, "bypass_a": 0.0, "export_w": 0.0, "lv_trip": False}
    if output_off:
        r.update(mode="off", served=0.0, curtailed_w=pv)
        return r
    if grid_ok and contactor_closed:
        inv_ac = min(load_w, pv * eta_i, cfg.INV_RATED_W)
        grid_load = load_w - inv_ac
        surplus = pv - inv_ac / eta_i
        pv_b = min(surplus, max_chg, head_w)
        g_b = max(0.0, min(max_grid, max_chg - pv_b, head_w - pv_b))
        g_b_ac = g_b / cfg.GRID_CHARGER_EFF
        r.update(mode="line", pv_to_load_w=inv_ac / eta_i, pv_to_batt_w=pv_b, curtailed_w=surplus - pv_b, grid_to_load_w=grid_load,
                 grid_to_batt_ac_w=g_b_ac, grid_import_w=grid_load + g_b_ac, batt_dc_w=pv_b + g_b,
                 soc_delta_wh=(pv_b + g_b) * cfg.ETA_CHARGE * dt_h, inv_out_va=load_va,
                 inverter_load_pct=100.0 * load_va / cfg.INV_RATED_VA,
                 bypass_a=(load_va * (grid_load / load_w if load_w > 0 else 0.0) + g_b_ac) / cfg.GRID_V)
        return r
    if lv_shutdown:
        pv_b = min(pv, max_chg, head_w)
        r.update(mode="lv_off", served=0.0, pv_to_batt_w=pv_b, curtailed_w=pv - pv_b, batt_dc_w=pv_b,
                 soc_delta_wh=pv_b * cfg.ETA_CHARGE * dt_h)
        return r
    pv_ac = min(load_w, pv * eta_i)
    need_dc = (load_w - pv_ac) / eta_i
    if need_dc <= avail_w + 1e-9:
        f = 1.0
        out = need_dc
    else:  # the battery empties within the step -> low-voltage shutdown for the rest of it
        f = avail_w / need_dc if need_dc > 0 else 1.0
        out = avail_w
    surplus = pv - pv_ac / eta_i if need_dc <= 0 else 0.0
    pv_b = min(surplus, max_chg, head_w)
    if f < 1.0:  # after the shutdown, PV charges the battery
        pv_b = min(pv, max_chg) * (1.0 - f)
    r.update(pv_to_load_w=pv_ac / eta_i * f, pv_to_batt_w=pv_b, curtailed_w=max(0.0, pv - pv_ac / eta_i * f - pv_b),
             batt_dc_w=pv_b - out * f, soc_delta_wh=(pv_b * cfg.ETA_CHARGE - out * f / cfg.ETA_DISCHARGE) * dt_h,
             served=f, inv_out_va=load_va, inverter_load_pct=100.0 * load_va / cfg.INV_RATED_VA, lv_trip=f < 1.0)
    return r


def overload_trips(segments, cfg: Config):
    """segments: consecutive (VA, seconds). Trip if the load stays above x*rated longer than the manual allows."""
    for ratio, tmax in cfg.OVERLOAD_CURVE:
        run = 0.0
        lim = ratio * cfg.INV_RATED_VA
        for va, sec in segments:
            if va > lim:
                run += sec
                if run > tmax + 1e-9:
                    return True, f">{ratio * 100:.0f}% for {run:.1f}s"
            else:
                run = 0.0
    return False, ""


def battery_voltage(soc_pct, batt_dc_w, cfg: Config):
    tab = np.array(cfg.BATT_OCV_TABLE, dtype=float)
    ocv = float(np.interp(soc_pct, tab[:, 0], tab[:, 1]))
    return ocv + cfg.BATT_R_INT_OHM * batt_dc_w / max(ocv, 1.0)


def tariff_cost(month_kwh_before, kwh, cfg: Config):
    """$ for kwh at the block tariff given the month-to-date energy."""
    low = max(0.0, min(kwh, cfg.TARIFF_TIER_KWH - month_kwh_before))
    return low * cfg.TARIFF_LOW + (kwh - low) * cfg.TARIFF_HIGH


def thermal_step(t_in, e_in, t_out, e_out, solar_w, home, ac_frac, cfg: Config, dt_s):
    ua, c = cfg.THERM_UA_W_K, cfg.THERM_C_J_K
    q = solar_w + (cfg.INTERNAL_GAIN_HOME_W if home else cfg.INTERNAL_GAIN_AWAY_W) - cfg.AC_COOLING_W * cfg.AC_SHR * ac_frac
    t_eq = t_out + q / ua
    t_new = t_eq + (t_in - t_eq) * math.exp(-dt_s * ua / c)
    kv = 1.0 / cfg.HUM_VENT_TAU_H
    ka = ac_frac / cfg.AC_DRY_TAU_H
    g = cfg.HUM_GAIN_HOME_HPA_H if home else 0.0
    e_coil = e_sat_hpa(cfg.AC_COIL_DEW_C)
    e_eq = (kv * e_out + g + ka * e_coil) / (kv + ka)
    e_new = e_eq + (e_in - e_eq) * math.exp(-(kv + ka) * dt_s / 3600.0)
    e_new = min(max(e_new, 1.0), e_sat_hpa(t_new))
    return t_new, e_new


def compute_reward(ev: dict, cfg: Config):
    """Sum of penalties; sizes enforce safety > water > fridge > comfort > cost. ev holds one step's events."""
    t = {
        "trip": cfg.W_TRIP * float(ev.get("trip", 0)),
        "reject": cfg.W_REJECT * float(ev.get("rejected", 0)),
        "water": cfg.W_WATER_OUTAGE * float(ev.get("water_outage", 0)),
        "fridge": cfg.W_FRIDGE_CUT * float(ev.get("fridge_cut", 0)),
        "discomfort": cfg.W_DISCOMFORT * float(ev.get("discomfort_c", 0)),
        "long_discomfort": cfg.W_LONG_DISCOMFORT * float(ev.get("long_discomfort", 0)),
        "low_soc": cfg.W_LOW_SOC * max(0.0, cfg.LOW_SOC_MARGIN - float(ev.get("soc_pct", 100))),
        "cost": cfg.W_COST * float(ev.get("cost_usd", 0)),
        "wasted_pv": cfg.W_WASTED_PV * float(ev.get("wasted_pv_kwh", 0)),
        "switch": cfg.W_SWITCH * float(ev.get("toggles", 0)),
        "generator": float(ev.get("generator", 0)),
    }
    return sum(t.values()), t


# =====================================================================================
# Exogenous processes (pre-generated at reset: identical for every agent with the same seed)
# =====================================================================================
def generate_exogenous(cfg: Config, data, s0: int, n: int, rng: np.random.Generator, opts: dict) -> dict:
    L = n + 1
    sl = slice(s0, s0 + L)
    hour = data.hour_local[sl].astype(float)
    day_id = data.day_id_local[sl]
    month = data.month_local[sl]
    spd = 96
    block = max(1, int(cfg.GRID_RATE_BLOCK_DAYS * spd))
    # grid
    grid = np.zeros(L, dtype=bool)
    events = []
    if opts.get("grid_events") is not None:
        for a, b in opts["grid_events"]:
            events.append((int(a), int(b) - int(a)))
    else:
        for b0 in range(0, L, block):
            bl = min(block, L - b0)
            lam = opts.get("grid_rate", rng.uniform(0.0, cfg.GRID_MAX_EVENTS_PER_DAY))
            k = rng.poisson(lam * bl / spd)
            for _ in range(k):
                st = b0 + int(rng.integers(0, bl))
                dur = int(round(rng.uniform(*cfg.GRID_EVENT_MIN) / cfg.STEP_MIN))
                events.append((st, max(1, dur)))
    for st, dur in events:
        grid[max(0, st): max(0, st + dur)] = True
    user_grid = np.zeros(L, dtype=bool)
    if opts.get("user_grid") is not None:
        for a, b in opts["user_grid"]:
            user_grid[a:b] = True
    elif opts.get("user_events", True):
        for st, dur in events:
            if rng.random() < cfg.USER_MANUAL_GRID_PROB:
                user_grid[st: st + min(dur, int(cfg.USER_MANUAL_GRID_MAX_H * 4))] = True
    user_grid &= grid
    # occupancy (True = home)
    if opts.get("occupancy") is not None:
        occ = np.asarray(opts["occupancy"], dtype=bool)
        occ = np.resize(occ, L)
    else:
        occ = np.ones(L, dtype=bool)
        if rng.random() < opts.get("away_episode_prob", cfg.OCC_AWAY_EPISODE_PROB):
            occ[:] = False
        else:
            for d in np.unique(day_id):
                m = day_id == d
                if rng.random() < cfg.OCC_WORKDAY_PROB:
                    occ[m & (hour >= cfg.OCC_WORK_HOURS[0]) & (hour < cfg.OCC_WORK_HOURS[1])] = False
                for _ in range(rng.poisson(cfg.OCC_OUTINGS_PER_DAY)):
                    h0 = rng.uniform(7.0, 21.0)
                    occ[m & (hour >= h0) & (hour < h0 + rng.uniform(*cfg.OCC_OUTING_H))] = False
    occ_obs = np.where(occ, 0, 1).astype(np.int8)
    if opts.get("occ_unknown") is not None:
        for a, b in opts["occ_unknown"]:
            occ_obs[a:b] = 2
    elif opts.get("occ_unknown_random", True):
        for _ in range(rng.poisson(cfg.OCC_UNKNOWN_PER_DAY * L / spd)):
            st = int(rng.integers(0, L))
            occ_obs[st: st + int(rng.uniform(*cfg.OCC_UNKNOWN_MIN) / cfg.STEP_MIN)] = 2
    # loads and water draw
    base = np.zeros(L)
    draw = np.zeros(L)
    for b0 in range(0, L, block):
        b1 = min(L, b0 + block)
        day_lvl = rng.uniform(*cfg.BASE_DAY_W)
        eve_lvl = rng.uniform(*cfg.BASE_EVENING_W)
        hh = hour[b0:b1]
        eve = (hh >= cfg.EVENING_HOURS[0]) & (hh < cfg.EVENING_HOURS[1]) & occ[b0:b1]
        base[b0:b1] = np.where(eve, eve_lvl, day_lvl)
        dpct = cfg.WATER_DRAW_PCT_PER_DAY * rng.uniform(*cfg.WATER_DRAW_RANDOM) * opts.get("water_draw_scale", 1.0)
        share = np.zeros(b1 - b0)
        for h0, h1, sh in cfg.WATER_PROFILE:
            share[(hh >= h0) & (hh < h1)] = sh / ((h1 - h0) * 4.0)
        summer = np.isin(month[b0:b1], [6, 7, 8, 9])
        draw[b0:b1] = dpct * share * np.where(summer, cfg.WATER_SUMMER_FACTOR, 1.0)
    if opts.get("base_w") is not None:
        base[:] = opts["base_w"]
    base *= rng.uniform(1 - cfg.BASE_STEP_NOISE, 1 + cfg.BASE_STEP_NOISE, L)
    draw *= np.where(occ, 1.0, cfg.WATER_AWAY_FACTOR) * rng.uniform(0.5, 1.5, L)
    fridge = cfg.FRIDGE_AVG_W * rng.uniform(0.6, 1.4, L)
    freezer = cfg.FREEZER_AVG_W * rng.uniform(0.6, 1.4, L)
    windows = opts.get("heater_windows", cfg.HEATER_WINDOWS_H)
    heater = np.zeros(L, dtype=bool)
    for h0, h1 in windows:
        heater |= (hour >= h0) & (hour < h1)
    # faults
    comm = np.ones(L, dtype=bool)
    ages = {k: np.zeros(L) for k in ("tank", "indoor", "pump_meter", "ac_meter", "fridge_meter")}
    inv_fault = np.zeros(L, dtype=np.int16)
    confirm = np.ones(L, dtype=bool)
    pi_restart = np.zeros(L, dtype=bool)
    mcb = {"pump": np.zeros(L, dtype=bool), "ac": np.zeros(L, dtype=bool)}
    reset = {"pump": np.zeros(L, dtype=bool), "ac": np.zeros(L, dtype=bool)}
    for f in opts.get("faults", []):
        st, du = int(f["start"]), max(1, int(f["duration"]))
        typ = f["type"]
        if typ == "comm_loss":
            comm[st: st + du] = False
        elif typ == "sensor_stale":
            ages[f.get("target", "tank")][st: st + du] = cfg.STALE_S * 2.5
        elif typ == "inverter_fault":
            inv_fault[st: st + du] = int(f.get("code", 9))
            cd = int(f.get("confirm_delay", 1))
            confirm[st: st + du + cd] = False
        elif typ == "pi_restart":
            comm[st: st + du] = False
            if st + du < L:
                pi_restart[st + du] = True
        elif typ == "mcb_trip":
            tgt = f.get("target", "pump")
            mcb[tgt][st: st + du] = True
            if st + du < L:
                reset[tgt][st + du] = True
    return dict(grid=grid, user_grid=user_grid, occ=occ, occ_obs=occ_obs, base=base, draw=draw, fridge=fridge, freezer=freezer,
                heater=heater, comm=comm, ages=ages, inv_fault=inv_fault, confirm=confirm, pi_restart=pi_restart, mcb=mcb,
                reset=reset, events=events)


def random_emergency(cfg: Config, n: int, rng: np.random.Generator) -> dict:
    types = list(cfg.EMERGENCY_TYPES)
    p = np.array([cfg.EMERGENCY_TYPES[t] for t in types], dtype=float)
    typ = types[int(rng.choice(len(types), p=p / p.sum()))]
    dur = int(round(rng.uniform(*cfg.EMERGENCY_DURATION_MIN) / cfg.STEP_MIN))
    st = int(rng.integers(4, max(5, n - dur - 4)))
    f = {"type": typ, "start": st, "duration": max(1, dur)}
    if typ == "inverter_fault":
        f["confirm_delay"] = int(round(rng.uniform(*cfg.USER_CONFIRM_DELAY_MIN) / cfg.STEP_MIN))
    if typ == "sensor_stale":
        f["target"] = ["tank", "indoor", "fridge_meter"][int(rng.integers(3))]
    if typ == "mcb_trip":
        f["target"] = ["pump", "ac"][int(rng.integers(2))]
    return f


# =====================================================================================
# Environment
# =====================================================================================
class HEMSEnv:
    metadata = {"step_minutes": 15}

    def __init__(self, cfg: Config, data, mode: str = "train", forecast_mode: str | None = None, seed: int | None = None,
                 keep_log: bool = False, want_mask: bool | None = None, track_monthly: bool = False):
        self.cfg = cfg
        self.track_monthly = track_monthly
        self._ym = data.day_id_local.astype("datetime64[D]").astype("datetime64[M]")
        self.data = data
        self.mode = mode
        self.forecast_mode = forecast_mode or cfg.FORECAST_MODE
        assert self.forecast_mode in ("ensemble", "none", "perfect")
        self.keep_log = keep_log
        self.want_mask = cfg.USE_ACTION_MASK if want_mask is None else want_mask
        self.n_actions = N_ACTIONS
        self.n_features = N_FEATURES
        self.feature_names = FEATURE_NAMES
        self.observation_shape = (N_FEATURES,)
        self._seed_rng = np.random.default_rng(cfg.SEED if seed is None else seed)
        self.dt_h = cfg.dt_h
        self.dt_s = cfg.STEP_MIN * 60.0
        self.cap_nom_wh = cfg.BATT_CAP_KWH * 1000.0
        self._valid = np.flatnonzero(data.valid_start & (np.arange(data.n) % 4 == 0))
        if len(self._valid) == 0:
            self._valid = np.arange(0, max(1, data.n - cfg.episode_steps - 1), 4)
        self.k = 0
        self.n_steps = 0

    # ---------------------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        cfg = self.cfg
        opts = dict(options or {})
        if seed is None:
            seed = int(self._seed_rng.integers(2 ** 31))
        rng = np.random.default_rng(seed)
        self.rng = rng
        train = self.mode == "train"
        hard = opts.get("hard")
        if hard is None and train and rng.random() < cfg.HARD_EPISODE_FRAC:
            hard = ["low_soc_tank_nogrid", "cloudy_winter", "long_no_grid"][int(rng.integers(3))]
        self.hard = hard
        n = int(opts.get("steps", cfg.episode_steps))
        if "start" in opts:
            s0 = int(opts["start"])
        elif hard == "cloudy_winter":
            s0 = self._cloudy_start(rng, n)
        else:
            s0 = int(rng.choice(self._valid))
        n = min(n, self.data.n - s0 - 1)
        self.s0, self.n_steps, self.k = s0, n, 0
        if hard == "long_no_grid":
            opts.setdefault("grid_rate", rng.uniform(0.0, 0.3))
        faults = list(opts.get("faults", []))
        if opts.get("emergency", train and rng.random() < cfg.EMERGENCY_EPISODE_FRAC):
            faults.append(random_emergency(cfg, n, rng))
        opts["faults"] = faults
        self.faults = faults
        self.x = generate_exogenous(cfg, self.data, s0, n, rng, opts)
        if hard == "low_soc_tank_nogrid":
            self.x["grid"][: min(n + 1, 96)] = False
            self.x["user_grid"][: min(n + 1, 96)] = False
        self.opts = opts
        # forecast mode / stress
        k_fc = {"ensemble": 1.0, "perfect": 0.0, "none": 0.0}[self.forecast_mode]
        if self.forecast_mode == "ensemble" and train and rng.random() < cfg.STRESS_EPISODE_FRAC:
            kmax = cfg.STRESS_MAX_SCALE or max(1.0, float(self.data.meta.get("winter_error_ratio", 1.0) or 1.0))
            k_fc = rng.uniform(1.0, max(1.0, kmax))
        self.k_fc = float(opts.get("stress_k", k_fc))
        self.fc_override = opts.get("fc_override", {})
        self.fc_transform = opts.get("fc_transform", {})  # feature -> (multiplier, offset), for scenario tests
        self.pv_scale = opts.get("pv_scale", 1.0)
        self.t_out_offset = float(opts.get("t_out_offset", 0.0))
        self.rh_out_override = opts.get("rh_out")
        # battery and house state
        lo, hi = (30.0, 40.0) if hard == "low_soc_tank_nogrid" else tuple(cfg.START_SOC_RANGE)
        self.soc = float(opts.get("soc", rng.uniform(lo, hi)))
        lo, hi = (12.0, 25.0) if hard == "low_soc_tank_nogrid" else tuple(cfg.START_TANK_RANGE)
        self.tank = float(opts.get("tank", rng.uniform(lo, hi)))
        self.soh = float(opts.get("soh", rng.uniform(*cfg.BATT_SOH_RANGE) if train else cfg.BATT_SOH))
        self.cap_wh = self.cap_nom_wh * self.soh
        s = s0
        t_day = float(np.mean(self.data.t_out[max(0, s - 96): s + 1])) + self.t_out_offset
        self.t_in = float(opts.get("t_in", t_day + rng.uniform(*cfg.INDOOR_INIT_OFFSET_C)))
        e_out = self._rh_out(0) / 100.0 * e_sat_hpa(self._t_out(0))
        self.e_in = min(e_sat_hpa(self.t_in) * 0.95, e_out * rng.uniform(0.9, 1.1))
        if "rh_in" in opts:
            self.e_in = opts["rh_in"] / 100.0 * e_sat_hpa(self.t_in)
        self.contactor = int(opts.get("contactor", 0))
        self.pump_on = 0
        self.ac_on = int(opts.get("ac_on", 0))
        self.heater_on = 0
        self.pump_run_min = 0.0
        self.pump_rest_left_min = 0.0
        self.pump_today_h = 0.0
        self.ac_run_min = 0.0
        self.ac_off_min = 1e6
        self.freezer_user_on = bool(opts.get("freezer_on", cfg.FREEZER_USER_ON))
        self.trip_left = 0
        self.trip_confirm = True
        self.lv_shutdown = False
        self.batt_dc_w = 0.0
        self.soc_hist = [self.soc] * 5
        self.tank_hist = [self.tank] * 5
        self.pv_hist = [float(self.data.pv_w[max(0, s - i)]) * self._pv_scale(0) for i in range(4, 0, -1)]
        self.last_total_w = 0.0
        self.last_total_va = 0.0
        self.last_base_w = float(self.x["base"][0])
        self.last_inv_pct = 0.0
        self.last_rejected = 0
        self.last_line = False
        self.prev_relays = {"contactor": 0, "pump": 0, "ac": 0, "fridge": 1, "freezer": int(self.freezer_user_on), "heater": 0}
        self.grid_arrived_k = 0 if self.x["grid"][0] else None
        self.grid_last_seen_h = float(opts.get("grid_last_seen_h", rng.uniform(*cfg.GRID_LAST_SEEN_INIT_H)))
        self.month = int(self.data.month_local[s])
        dom = int(np.datetime64(self.data.t_utc[s], "D").astype(object).day)
        self.month_kwh = float(opts.get("month_kwh", rng.uniform(*cfg.MONTH_KWH_INIT_PER_DAY) * (dom - 1)))
        self.day_id = int(self.data.day_id_local[s])
        self.occ_last_known = int(self.x["occ_obs"][0]) if self.x["occ_obs"][0] != 2 else int(not self.x["occ"][0])
        self.occ_age_steps = 0
        self.disc_streak = 0
        self.gen_pending = []
        self.gen_reward_pending = 0.0
        self.time_s = 0.0
        self.bus = ActuatorBus()
        self.safety = SafetyController(cfg, self.bus, start_mode="agent", keep_log=self.keep_log)
        self.safety.update_mode(self._measurements(0))
        self.metrics = {k: 0.0 for k in ("steps", "agent_steps", "emergency_steps", "trips", "export_kwh", "water_outage_min",
                                         "fridge_cut_min", "discomfort_ch", "grid_kwh", "grid_kwh_agent", "grid_kwh_user", "bill_usd",
                                         "bill_usd_agent", "pv_avail_kwh", "pv_curtailed_kwh", "rejected", "toggles",
                                         "gen_requests", "gen_right", "gen_judged", "lv_shutdown_min", "reward", "unserved_kwh",
                                         "user_override_steps", "pump_kwh", "ac_kwh", "heater_kwh")}
        self.log = []
        self.monthly = {}
        self._cur_ym = self._ym[s0]
        self._ym_snap = dict(self.metrics)
        obs = self._obs()
        info = {"agent_in_control": self.safety.mode == "agent", "segment_start": True, "episode_end": False,
                "start_utc": str(self.data.t_utc[s]), "hard": hard, "faults": faults}
        if self.want_mask:
            info["action_mask"] = self.action_mask()
        return obs, info

    def _cloudy_start(self, rng, n):
        v = self._valid
        months = self.data.month_local[v]
        cand = v[np.isin(months, [12, 1, 2])]
        if len(cand) < 10:
            cand = v
        cs = np.concatenate([[0.0], np.cumsum(self.data.pv_w)])
        e = cs[np.minimum(cand + n, len(cs) - 1)] - cs[cand]
        low = cand[e <= np.quantile(e, 0.2)]
        return int(rng.choice(low if len(low) else cand))

    # --- exogenous accessors --------------------------------------------------------
    def _pv_scale(self, k):
        ps = self.pv_scale
        return float(ps[min(k, len(ps) - 1)]) if isinstance(ps, (list, np.ndarray)) else float(ps)

    def _t_out(self, k):
        return float(self.data.t_out[self.s0 + k]) + self.t_out_offset

    def _rh_out(self, k):
        if self.rh_out_override is not None:
            return float(self.rh_out_override)
        return float(self.data.rh_out[self.s0 + k])

    def _measurements(self, k) -> Measurements:
        x = self.x
        ages = {key: float(v[k]) for key, v in x["ages"].items()}
        fault = int(x["inv_fault"][k]) or (7 if self.trip_left > 0 else 0)
        return Measurements(time_s=k * self.dt_s, comm_ok=bool(x["comm"][k]), sensor_age_s=ages, inverter_fault_code=fault,
                            pi_restarted=bool(x["pi_restart"][k]), user_confirmed=bool(x["confirm"][k]) and self.trip_confirm,
                            pump_meter_w=getattr(self, "_pump_meter", 0.0), ac_meter_w=getattr(self, "_ac_meter", 0.0),
                            pump_cmd_last=self.prev_relays["pump"], ac_cmd_last=self.prev_relays["ac"],
                            power_available=getattr(self, "_power_ok", True),
                            user_reset={"pump": bool(x["reset"]["pump"][k]), "ac": bool(x["reset"]["ac"][k])})

    def _user_fixed(self, k) -> dict:
        x = self.x
        if x["user_grid"][k] or (self.cfg.HEATER_FORCES_CONTACTOR and x["heater"][k] and x["grid"][k]):
            return {"contactor": 1}
        return {}

    def _context(self, k) -> SafetyContext:
        x = self.x
        return SafetyContext(
            grid_present=bool(x["grid"][k]), soc=self.soc, tank=self.tank, contactor_on=self.contactor, pump_on=self.pump_on,
            ac_on=self.ac_on, pump_run_s=self.pump_run_min * 60.0, pump_rest_left_s=self.pump_rest_left_min * 60.0,
            ac_off_s=self.ac_off_min * 60.0, heater_request=bool(x["heater"][k]), base_va=self.last_base_w / self.cfg.BASE_PF,
            freezer_user_on=self.freezer_user_on, blackout=bool(self.trip_left > 0 or x["inv_fault"][k] > 0),
            user_fixed=self._user_fixed(k), dt_s=self.dt_s)

    def action_mask(self) -> np.ndarray:
        return valid_action_mask(self._context(self.k), self.cfg)

    # ---------------------------------------------------------------------------------
    def step(self, action):
        cfg, x, k = self.cfg, self.x, self.k
        s = self.s0 + k
        dt_h, dt_s = self.dt_h, self.dt_s
        mode = self.safety.mode
        in_control = mode == "agent"
        a_agent = int(action)
        soc0, tank0 = self.soc, self.tank
        obs_now = getattr(self, "_last_obs", None)
        if in_control:
            proposal = decode_action(a_agent)
        else:
            proposal = decode_action(rule_based_action(obs_to_dict(obs_now), cfg))
        grid = bool(x["grid"][k])
        ctx = self._context(k)
        dec = self.safety.decide(proposal, ctx, t=k * dt_s)
        rel = self.bus.read()
        c_ex, p_ex, a_ex = rel["contactor"], rel["pump"], rel["ac"]
        g_ex = dec.executed[3]
        line = grid and bool(c_ex)
        output_off = self.trip_left > 0 or x["inv_fault"][k] > 0
        if line:
            self.lv_shutdown = False
        # loads this step
        mcb_p, mcb_a = bool(x["mcb"]["pump"][k]), bool(x["mcb"]["ac"][k])
        pump_w_nom = cfg.PUMP_RUN_A * cfg.GRID_V * cfg.PUMP_PF
        fill_step = cfg.PUMP_FILL_PCT_PER_H * dt_h
        frac_p = 0.0
        if p_ex and not mcb_p:
            frac_p = 1.0 if self.tank + fill_step <= cfg.TANK_FULL else max(0.0, (cfg.TANK_FULL - self.tank) / fill_step)
        base_w = float(x["base"][k])
        fridge_w = float(x["fridge"][k]) * (1.0 + 0.02 * (self.t_in - 25.0)) if rel["fridge"] else 0.0
        freezer_w = float(x["freezer"][k]) if rel["freezer"] else 0.0
        heater_w = cfg.HEATER_W if rel["heater"] else 0.0
        ac_w = cfg.AC_INPUT_W if (a_ex and not mcb_a) else 0.0
        pump_w = pump_w_nom * frac_p
        load_w = base_w + fridge_w + freezer_w + heater_w + ac_w + pump_w + cfg.INVERTER_IDLE_W
        run_va = (base_w / cfg.BASE_PF + (cfg.FRIDGE_RUN_A[1] * cfg.GRID_V if rel["fridge"] else 0.0)
                  + (cfg.FREEZER_RUN_A * cfg.GRID_V if rel["freezer"] else 0.0) + heater_w + cfg.INVERTER_IDLE_W
                  + (cfg.AC_RUN_A * cfg.GRID_V if ac_w > 0 else 0.0) + (cfg.PUMP_RUN_A * cfg.GRID_V if frac_p > 0 else 0.0))
        # overload check (battery mode: inverter carries everything; line mode: 30 A bypass)
        trip, why = False, ""
        if not output_off:
            if line:
                pass
            else:
                segs = []
                if self.last_line and not grid and self.prev_relays["contactor"]:
                    segs.append((self.last_total_va, cfg.GRID_LOSS_REACTION_S))  # grid drop before load shedding
                started_p = frac_p > 0 and not self.pump_on
                started_a = ac_w > 0 and not self.ac_on
                if started_p:
                    segs.append((run_va - cfg.PUMP_RUN_A * cfg.GRID_V + cfg.PUMP_START_FACTOR * cfg.PUMP_RUN_A * cfg.GRID_V, cfg.PUMP_START_S))
                if started_a:
                    segs.append((run_va - cfg.AC_RUN_A * cfg.GRID_V + cfg.AC_START_A * cfg.GRID_V, cfg.AC_START_S))
                if rel["fridge"] and not self.prev_relays["fridge"]:
                    segs.append((run_va - cfg.FRIDGE_RUN_A[1] * cfg.GRID_V + cfg.FRIDGE_START_A * cfg.GRID_V, 2.0))
                segs.append((run_va, dt_s))
                if not self.lv_shutdown:
                    trip, why = overload_trips(segs, cfg)
        pv_avail = float(self.data.pv_w[s]) * self._pv_scale(k)
        if not trip:
            fl = inverter_model(pv_avail, load_w, run_va, grid, c_ex, self.soc, self.cap_wh, dt_h, cfg,
                                output_off=output_off, lv_shutdown=self.lv_shutdown and not line)
            if fl["mode"] == "line" and fl["bypass_a"] > cfg.BYPASS_MAX_A:
                trip, why = True, f"bypass {fl['bypass_a']:.1f} A > {cfg.BYPASS_MAX_A} A"
        if trip:
            self.trip_left = int(math.ceil(cfg.TRIP_OUTAGE_MIN / cfg.STEP_MIN))
            fl = inverter_model(pv_avail, load_w, run_va, grid, c_ex, self.soc, self.cap_wh, dt_h, cfg, output_off=True)
        served = fl["served"]
        if fl["lv_trip"]:
            self.lv_shutdown = True
        # battery
        self.soc = float(np.clip(self.soc + 100.0 * fl["soc_delta_wh"] / self.cap_wh, 0.0, 100.0))
        if self.lv_shutdown and not line and self.soc >= cfg.LV_RESTART_SOC and not fl["lv_trip"]:
            self.lv_shutdown = False
        self.batt_dc_w = fl["batt_dc_w"]
        # water
        pumped = fill_step * frac_p * served
        self.tank = float(np.clip(self.tank + pumped - float(x["draw"][k]), 0.0, 100.0))
        # comfort
        home = bool(x["occ"][k])
        ac_frac = (1.0 if ac_w > 0 else 0.0) * served
        e_out = self._rh_out(k) / 100.0 * e_sat_hpa(self._t_out(k))
        self.t_in, self.e_in = thermal_step(self.t_in, self.e_in, self._t_out(k), e_out, cfg.SOLAR_GAIN_PER_PV_W * pv_avail,
                                            home, ac_frac, cfg, dt_s)
        rh_in = 100.0 * self.e_in / e_sat_hpa(self.t_in)
        hx = humidex(self.t_in, rh_in)
        hour_next = float(self.data.hour_local[s + 1])
        thr = self._threshold(hour_next)
        occ_obs = int(x["occ_obs"][k])
        eff_home = occ_obs == 0 or (occ_obs == 2 and self.occ_last_known == 0)
        excess = max(0.0, hx - thr)
        disc = excess if eff_home else 0.0
        self.disc_streak = self.disc_streak + 1 if disc > 0 else 0
        long_disc = self.disc_streak > cfg.LONG_DISCOMFORT_H * 4
        # grid energy, tariff (manual user / heater-program energy is logged but not charged to the agent)
        mon = int(self.data.month_local[s])
        if mon != self.month:
            self.month, self.month_kwh = mon, 0.0
        g_kwh = fl["grid_import_w"] * dt_h / 1000.0
        heater_kwh = heater_w * dt_h / 1000.0 * served if line else 0.0
        user_kwh = g_kwh if dec.user_override else min(g_kwh, heater_kwh)
        cost = tariff_cost(self.month_kwh, g_kwh, cfg)
        cost_agent = cost * ((g_kwh - user_kwh) / g_kwh) if g_kwh > 0 else 0.0
        self.month_kwh += g_kwh
        curtailed_kwh = fl["curtailed_w"] * dt_h / 1000.0
        wasted = curtailed_kwh if (self.tank < cfg.TANK_FULL and not output_off and not trip) else 0.0
        # fridge / freezer cut (R4, low-voltage shutdown or trip)
        fridge_cut_frac = 0.0
        if not rel["fridge"] or (self.freezer_user_on and not rel["freezer"]):
            fridge_cut_frac = 1.0
        elif served < 1.0:
            fridge_cut_frac = 1.0 - served
        # toggles caused by the agent's decisions
        toggles = 0
        for ch, now in (("contactor", c_ex), ("pump", p_ex), ("ac", a_ex)):
            if now != self.prev_relays[ch] and ch not in ctx.user_fixed:
                toggles += 1
        # generator advisory (judged in hindsight over GEN_LOOKAHEAD_H)
        fridge_event = fridge_cut_frac > 0
        tank_event = self.tank < cfg.TANK_MIN
        comfort_event = eff_home and self.disc_streak > cfg.GEN_COMFORT_VIOLATION_H * 4
        if in_control and g_ex:
            self.gen_pending.append((k, k + int(cfg.GEN_LOOKAHEAD_H * 4)))
            self.metrics["gen_requests"] += 1
        gen_r = 0.0
        right = cfg.W_GEN_RIGHT_PER_H * dt_h
        wrong = (cfg.W_GEN_WRONG_BASE_PER_H - cfg.GEN_COST_PER_H * cfg.GEN_WRONG_COST_MULT) * dt_h
        keep = []
        for kr, dl in self.gen_pending:
            if fridge_event or tank_event or comfort_event:
                gen_r += right
                self.metrics["gen_right"] += 1
                self.metrics["gen_judged"] += 1
            elif k + 1 >= dl:
                gen_r += wrong
                self.metrics["gen_judged"] += 1
            else:
                keep.append((kr, dl))
        self.gen_pending = keep
        episode_end = k + 1 >= self.n_steps
        if episode_end and self.gen_pending:  # window cut by the episode end: judged on what was seen
            gen_r += wrong * len(self.gen_pending)
            self.metrics["gen_judged"] += len(self.gen_pending)
            self.gen_pending = []
        self.gen_reward_pending += gen_r
        ev = {"trip": trip, "rejected": dec.rejected, "water_outage": tank_event, "fridge_cut": fridge_cut_frac,
              "discomfort_c": disc, "long_discomfort": long_disc and disc > 0, "soc_pct": self.soc, "cost_usd": cost_agent,
              "wasted_pv_kwh": wasted, "toggles": toggles, "generator": self.gen_reward_pending if in_control else 0.0}
        total, terms = compute_reward(ev, cfg)
        if in_control:
            self.gen_reward_pending = 0.0
            reward = total
        else:
            reward = 0.0  # emergency steps never enter learning
        # metrics
        M = self.metrics
        if self.track_monthly and self._ym[s] != self._cur_ym:
            self._close_month()
            self._cur_ym = self._ym[s]
        M["steps"] += 1
        M["agent_steps"] += in_control
        M["emergency_steps"] += not in_control
        M["trips"] += trip
        M["export_kwh"] += fl["export_w"] * dt_h / 1000.0
        M["water_outage_min"] += cfg.STEP_MIN * tank_event
        M["fridge_cut_min"] += cfg.STEP_MIN * fridge_cut_frac
        M["discomfort_ch"] += disc * dt_h
        M["grid_kwh"] += g_kwh
        M["grid_kwh_user"] += user_kwh
        M["grid_kwh_agent"] += g_kwh - user_kwh
        M["bill_usd"] += cost
        M["bill_usd_agent"] += cost_agent
        M["pv_avail_kwh"] += pv_avail * dt_h / 1000.0
        M["pv_curtailed_kwh"] += curtailed_kwh
        M["rejected"] += dec.rejected
        M["toggles"] += toggles
        M["lv_shutdown_min"] += cfg.STEP_MIN * (1.0 - served) * (fl["mode"] in ("battery", "lv_off"))
        M["unserved_kwh"] += load_w * (1.0 - served) * dt_h / 1000.0
        M["reward"] += reward
        M["user_override_steps"] += bool(ctx.user_fixed)
        M["pump_kwh"] += pump_w * served * dt_h / 1000.0
        M["ac_kwh"] += ac_w * served * dt_h / 1000.0
        M["heater_kwh"] += heater_kwh
        # timers and histories
        if p_ex and frac_p > 0:
            self.pump_run_min += cfg.STEP_MIN * frac_p
            self.pump_today_h += dt_h * frac_p
        new_pump = int(p_ex and frac_p >= 1.0)  # stops by itself (R10 at the ESP32) when the tank reaches full
        if self.pump_on and not new_pump:
            if self.pump_run_min >= cfg.PUMP_MAX_H * 60 - 1e-6:
                self.pump_rest_left_min = cfg.PUMP_REST_MIN
            self.pump_run_min = 0.0
        elif not self.pump_on and not new_pump:
            self.pump_rest_left_min = max(0.0, self.pump_rest_left_min - cfg.STEP_MIN)
        if p_ex and self.pump_run_min >= cfg.PUMP_MAX_H * 60 - 1e-6:
            new_pump = 0
            self.pump_rest_left_min = cfg.PUMP_REST_MIN
            self.pump_run_min = 0.0
        self.pump_on = new_pump
        if a_ex:
            self.ac_run_min = self.ac_run_min + cfg.STEP_MIN if self.ac_on else cfg.STEP_MIN
            self.ac_off_min = 0.0
        else:
            self.ac_off_min = (self.ac_off_min + cfg.STEP_MIN) if not self.ac_on else cfg.STEP_MIN
            self.ac_run_min = 0.0
        self.ac_on = int(a_ex)
        self.contactor = int(c_ex)
        self.heater_on = int(rel["heater"])
        self._pump_meter = pump_w * served if p_ex else 0.0
        self._ac_meter = ac_w * served if a_ex else 0.0
        self._power_ok = served > 0
        self.prev_relays = dict(rel)
        self.last_line = line and served > 0
        self.last_total_w = load_w * served
        self.last_total_va = run_va
        self.last_base_w = base_w
        self.last_inv_pct = fl["inverter_load_pct"] if served > 0 else 0.0
        self.last_rejected = int(dec.rejected)
        self.soc_hist = (self.soc_hist + [self.soc])[-5:]
        self.tank_hist = (self.tank_hist + [self.tank])[-5:]
        self.pv_hist = (self.pv_hist + [pv_avail])[-4:]
        if self.trip_left > 0:
            self.trip_left -= 1
        if grid:
            self.grid_last_seen_h = 0.0
        else:
            self.grid_last_seen_h += dt_h
        nxt = k + 1
        if x["grid"][min(nxt, self.n_steps)] and not grid:
            self.grid_arrived_k = nxt
        if occ_obs != 2:
            self.occ_last_known, self.occ_age_steps = occ_obs, 0
        else:
            self.occ_age_steps += 1
        day_next = int(self.data.day_id_local[s + 1])
        if day_next != self.day_id:
            self.day_id, self.pump_today_h = day_next, 0.0
        self.k = nxt
        self.time_s = nxt * dt_s
        # next control mode (emergencies) and truncation
        if not episode_end:
            next_mode = self.safety.update_mode(self._measurements(nxt))
        else:
            next_mode = self.safety.mode
        emergency_start = in_control and next_mode != "agent" and not episode_end
        truncated = episode_end or emergency_start
        obs = self._obs()
        info = {"agent_in_control": in_control, "mode": mode, "next_mode": next_mode, "emergency_start": emergency_start,
                "segment_start": (not in_control) and next_mode == "agent", "episode_end": episode_end,
                "proposed": proposal, "executed": dec.executed, "rejected": dec.rejected, "violations": dec.violations,
                "terms": terms, "reward_total": total, "trip": trip, "trip_reason": why, "served": served,
                "grid_kwh": g_kwh, "grid_kwh_user": user_kwh, "cost_usd": cost, "cost_usd_agent": cost_agent,
                "shadow_action": None if in_control else a_agent, "alerts": dec.alerts}
        if episode_end:
            if self.track_monthly:
                self._close_month()
            info["episode_metrics"] = self.episode_metrics()
        if self.want_mask:
            info["action_mask"] = self.action_mask()
        if self.keep_log:
            self.log.append({"k": k, "t_utc": str(self.data.t_utc[s]), "hour_local": float(self.data.hour_local[s]), "mode": mode,
                             "agent_action": a_agent, "proposed": encode_action(*proposal), "executed": encode_action(*dec.executed),
                             "contactor": c_ex, "pump": p_ex, "ac": a_ex, "gen": g_ex, "fridge": rel["fridge"], "heater": rel["heater"],
                             "rejected": int(dec.rejected), "violations": ",".join(dec.violations), "grid": int(grid),
                             "user_grid": int(bool(ctx.user_fixed)), "soc0": soc0, "tank0": tank0, "soc": self.soc, "tank": self.tank, "pv_w": pv_avail,
                             "load_w": load_w, "grid_w": fl["grid_import_w"], "batt_dc_w": fl["batt_dc_w"], "curtailed_w": fl["curtailed_w"],
                             "served": served, "t_in": self.t_in, "rh_in": rh_in, "humidex": hx, "thr": thr, "home": int(home),
                             "occ_obs": occ_obs, "eff_home": int(eff_home), "reward": reward, "trip": int(trip), "lv": int(self.lv_shutdown),
                             **{f"r_{n}": v for n, v in terms.items()}})
        return obs, float(reward), False, bool(truncated), info

    def _threshold(self, hour):
        n0, n1 = self.cfg.NIGHT_HOURS
        night = hour >= n0 or hour < n1
        return self.cfg.HUMIDEX_NIGHT if night else self.cfg.HUMIDEX_DAY

    # ---------------------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        cfg, d, x, k = self.cfg, self.data, self.x, self.k
        s = self.s0 + k
        o = np.zeros(N_FEATURES, dtype=np.float32)
        I = IDX
        h = float(d.hour_local[s])
        doy = float(d.doy_local[s])
        mon = int(d.month_local[s])
        o[I["hour_sin"]], o[I["hour_cos"]] = math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)
        o[I["doy_sin"]], o[I["doy_cos"]] = math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25)
        o[I["month_sin"]], o[I["month_cos"]] = math.sin(2 * math.pi * (mon - 1) / 12), math.cos(2 * math.pi * (mon - 1) / 12)
        season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(mon, 3)
        o[I["season_winter"] + season] = 1.0
        o[I["is_day"]] = float(d.is_day[s])
        o[I["h_to_sunrise"]] = d.min_to_sunrise[s] / 60.0
        o[I["h_to_sunset"]] = d.min_to_sunset[s] / 60.0
        # battery
        o[I["soc_pct"]] = self.soc
        o[I["batt_power_w"]] = self.batt_dc_w
        o[I["soc_chg_15m"]] = self.soc_hist[-1] - self.soc_hist[-2]
        o[I["soc_chg_1h"]] = self.soc_hist[-1] - self.soc_hist[0]
        o[I["batt_voltage_v"]] = battery_voltage(self.soc, self.batt_dc_w, cfg)
        o[I["soh"]] = self.soh
        net = -self.batt_dc_w
        o[I["h_to_soc35"]] = min(48.0, max(0.0, (self.soc - cfg.SOC_RESTRICT) / 100.0 * self.cap_wh / net)) if net > 1 else 48.0
        # PV and forecast
        o[I["pv_now_w"]] = self.pv_hist[-1]
        o[I["pv_last_hour_w"]] = float(np.mean(self.pv_hist))
        if self.forecast_mode != "none":
            fe, fp = d.fc_ens[s], d.fc_perf[s]
            fc = fp + self.k_fc * (fe - fp) if self.k_fc != 0.0 else fp
            o[FC_IDX[:-1]] = fc
            o[I["cloud_now"]] = d.cloud_now[s]
            for name, (mul, add) in self.fc_transform.items():
                o[I[name]] = o[I[name]] * mul + add
            for name, v in self.fc_override.items():
                o[I[name]] = v[min(k, len(v) - 1)] if isinstance(v, (list, np.ndarray)) else v
        # grid
        grid = bool(x["grid"][k])
        o[I["grid_present"]] = grid
        o[I["contactor_closed"]] = self.contactor
        o[I["h_since_grid_arrived"]] = min(cfg.MAX_HOURS_FEATURE, (k - self.grid_arrived_k) * self.dt_h) if (grid and self.grid_arrived_k is not None) else 0.0
        o[I["h_since_grid_seen"]] = 0.0 if grid else min(cfg.MAX_HOURS_FEATURE, self.grid_last_seen_h)
        o[I["grid_kwh_month"]] = self.month_kwh
        o[I["tariff_high"]] = float(self.month_kwh >= cfg.TARIFF_TIER_KWH)
        # water
        o[I["tank_pct"]] = self.tank
        o[I["tank_chg_per_h"]] = self.tank_hist[-1] - self.tank_hist[0]
        o[I["pump_on"]] = self.pump_on
        o[I["pump_run_h"]] = self.pump_run_min / 60.0
        o[I["pump_h_today"]] = self.pump_today_h
        # comfort
        rh_in = 100.0 * self.e_in / e_sat_hpa(self.t_in)
        hx = humidex(self.t_in, rh_in)
        thr = self._threshold(h)
        o[I["t_in"]], o[I["rh_in"]], o[I["humidex_in"]], o[I["humidex_thr"]] = self.t_in, rh_in, hx, thr
        o[I["humidex_excess"]] = max(0.0, hx - thr)
        o[I["t_out"]] = self._t_out(k)
        o[I["rh_out"]] = self._rh_out(k)
        n = d.n - 1
        o[I["t_out_1_3h"]] = float(np.mean(d.t_out[min(s + 4, n): min(s + 12, n) + 1])) + self.t_out_offset
        o[I["t_out_4_8h"]] = float(np.mean(d.t_out[min(s + 16, n): min(s + 32, n) + 1])) + self.t_out_offset
        if self.rh_out_override is None:
            o[I["rh_out_1_3h"]] = float(np.mean(d.rh_out[min(s + 4, n): min(s + 12, n) + 1]))
            o[I["rh_out_4_8h"]] = float(np.mean(d.rh_out[min(s + 16, n): min(s + 32, n) + 1]))
        else:
            o[I["rh_out_1_3h"]] = o[I["rh_out_4_8h"]] = float(self.rh_out_override)
        o[I["ac_on"]] = self.ac_on
        o[I["ac_run_h"]] = self.ac_run_min / 60.0
        occ_obs = int(x["occ_obs"][k])
        o[I["occ_home"] + occ_obs] = 1.0
        o[I["occ_age_h"]] = self.occ_age_steps * self.dt_h
        o[I["occ_effective_home"]] = float(occ_obs == 0 or (occ_obs == 2 and self.occ_last_known == 0))
        o[I["discomfort_h"]] = self.disc_streak * self.dt_h
        # loads
        o[I["fridge_on"]] = self.prev_relays["fridge"]
        o[I["freezer_on"]] = self.prev_relays["freezer"]
        o[I["heater_on"]] = self.heater_on
        o[I["base_load_w"]] = self.last_base_w
        o[I["total_load_w"]] = self.last_total_w
        o[I["inverter_load_pct"]] = self.last_inv_pct
        # safety feedback
        sc = self.safety
        o[I["last_rejected"]] = self.last_rejected
        o[I["safe_mode"]] = float(sc.mode == "safe")
        o[I["emergency_mode"]] = float(sc.mode == "emergency")
        o[I["inverter_fault"]] = float(x["inv_fault"][k] > 0 or self.trip_left > 0)
        o[I["pump_unavailable"]] = float(sc.unavailable["pump"] or sc.stale["pump"])
        o[I["ac_unavailable"]] = float(sc.unavailable["ac"] or sc.stale["ac"])
        o[I["user_override_contactor"]] = float("contactor" in self._user_fixed(k))
        o[I["fridge_cut_active"]] = float(sc.r4_cut_active)
        o[I["lv_shutdown"]] = float(self.lv_shutdown)
        self._last_obs = o
        return o.copy()

    def _close_month(self):
        M = {k: self.metrics[k] - self._ym_snap.get(k, 0.0) for k in self.metrics}
        if M["steps"] > 0:
            self.monthly[str(self._cur_ym)] = derive_metrics(M, self.dt_h)
        self._ym_snap = dict(self.metrics)

    def episode_metrics(self) -> dict:
        return derive_metrics(dict(self.metrics), self.dt_h)


def derive_metrics(M: dict, dt_h: float) -> dict:
    """Add per-day rates and percentages to accumulated metrics."""
    days = max(M["steps"] * dt_h / 24.0, 1e-9)
    M["days"] = days
    M["pv_used_pct"] = 100.0 * (1.0 - M["pv_curtailed_kwh"] / M["pv_avail_kwh"]) if M["pv_avail_kwh"] > 0 else 100.0
    M["rejected_per_day"] = M["rejected"] / days
    M["toggles_per_day"] = M["toggles"] / days
    M["gen_right_pct"] = 100.0 * M["gen_right"] / M["gen_judged"] if M["gen_judged"] > 0 else float("nan")
    return M


# =====================================================================================
# Synchronous vector environment
# =====================================================================================
class VecEnv:
    """Runs several HEMSEnv copies; resets an env only when its episode ends (not at emergency truncation)."""

    def __init__(self, envs, seed: int):
        self.envs = envs
        self.n = len(envs)
        self.rng = np.random.default_rng(seed)
        self.obs = None
        self.completed = []  # metrics of finished episodes (training logs)

    def reset(self):
        out = [e.reset(int(self.rng.integers(2 ** 31))) for e in self.envs]
        self.obs = np.stack([o for o, _ in out])
        return self.obs.copy(), [i for _, i in out]

    def step(self, actions):
        obs, rews, terms, truncs, infos = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, te, tr, info = e.step(int(a))
            if info["episode_end"]:
                info["final_obs"] = o
                self.completed.append(info["episode_metrics"])
                o, i2 = e.reset(int(self.rng.integers(2 ** 31)))
                info["reset_info"] = i2
            obs.append(o)
            rews.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)
        self.obs = np.stack(obs)
        return self.obs.copy(), np.array(rews, dtype=np.float32), np.array(terms), np.array(truncs), infos

    def action_masks(self):
        return np.stack([e.action_mask() for e in self.envs])


def run_episodes(policy, cfg: Config, data, specs, forecast_mode: str = "ensemble", keep_log: bool = False,
                 mode: str = "eval", track_monthly: bool = False):
    """Run several fixed episodes in lock-step with batched decisions.
    policy(obs_batch, masks_or_None) -> actions. specs: [{"seed": int, "options": {...}}]."""
    envs = [HEMSEnv(cfg, data, mode=mode, forecast_mode=forecast_mode, seed=sp["seed"], keep_log=keep_log,
                    track_monthly=track_monthly) for sp in specs]
    obs = [e.reset(seed=sp["seed"], options=sp.get("options"))[0] for e, sp in zip(envs, specs)]
    active = list(range(len(envs)))
    results = [None] * len(envs)
    while active:
        ob = np.stack([obs[i] for i in active])
        masks = np.stack([envs[i].action_mask() for i in active]) if cfg.USE_ACTION_MASK else None
        acts = policy(ob, masks)
        still = []
        for j, i in enumerate(active):
            o, r, te, tr, info = envs[i].step(int(acts[j]))
            obs[i] = o
            if info["episode_end"]:
                results[i] = {"metrics": info["episode_metrics"], "monthly": dict(envs[i].monthly), "log": envs[i].log,
                              "alerts": list(envs[i].safety.alerts), "spec": specs[i], "start_utc": str(data.t_utc[envs[i].s0])}
            else:
                still.append(i)
        active = still
    return results


# =====================================================================================
# Self-test
# =====================================================================================
def selftest(out_root: str | None = None) -> bool:
    T = SelfTest("hems_env")
    cfg0 = make_config()
    dt = cfg0.dt_h
    cap = cfg0.BATT_CAP_KWH * 1000
    # humidex table from the master prompt
    tab = [(25, 50, 28.2), (25, 70, 31.8), (27, 50, 31.3), (28, 30, 28.7), (28, 60, 35.0)]
    T.check(all(abs(humidex(t, r) - h) < 0.06 for t, r, h in tab), "humidex reference table (5 rows)")
    # inverter model
    f = inverter_model(0, 1000, 1100, False, 0, 60, cap, dt, cfg0)
    T.check(abs(f["batt_dc_w"] + 1000 / 0.93) < 1e-6 and abs(f["soc_delta_wh"] + 1000 / 0.93 / 0.95 * dt) < 1e-6, "battery mode: load from battery through 0.93 x 0.95")
    f = inverter_model(2500, 300, 330, False, 0, 60, cap, dt, cfg0)
    T.check(abs(f["pv_to_batt_w"] - 60 * 26.7) < 1e-6 and f["curtailed_w"] > 0 and f["grid_import_w"] == 0, "battery mode: PV charging limited to 60 A, surplus curtailed")
    f = inverter_model(0, 1000, 1100, True, 1, 60, cap, dt, cfg0)
    T.check(abs(f["grid_to_load_w"] - 1000) < 1e-6 and abs(f["batt_dc_w"] - 30 * 26.7) < 1e-6 and abs(f["grid_import_w"] - (1000 + 801 / 0.93)) < 1e-6,
            "line mode: loads from grid, battery charges 30 A (about 0.8 kW) from grid")
    f = inverter_model(3000, 500, 550, True, 1, 60, cap, dt, cfg0)
    T.check(abs(f["pv_to_load_w"] - 500 / 0.93) < 1e-6 and abs(f["batt_dc_w"] - 1602) < 1e-6 and f["grid_import_w"] == 0, "line mode: PV first, PV + grid charge within 60 A total")
    f = inverter_model(3000, 500, 550, True, 1, 100, cap, dt, cfg0)
    T.check(f["batt_dc_w"] == 0 and abs(f["curtailed_w"] - (3000 - 500 / 0.93)) < 1e-6, "full battery: no charging, PV curtailed, no export")
    f = inverter_model(0, 1000, 1100, False, 0, 1, cap, dt, cfg0)
    T.check(f["served"] < 1 and f["lv_trip"], "empty battery: low-voltage shutdown within the step")
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(3000):
        g, c = bool(rng.random() < .5), int(rng.random() < .5)
        pv, lw, soc = rng.uniform(0, 3000), rng.uniform(0, 4000), rng.uniform(0, 100)
        f = inverter_model(pv, lw, lw * 1.1, g, c, soc, cap, dt, cfg0)
        ok &= f["grid_import_w"] >= -1e-9 and f["export_w"] == 0 and f["curtailed_w"] >= -1e-6
        ok &= f["pv_to_load_w"] + f["pv_to_batt_w"] + f["curtailed_w"] <= pv + 1e-6
        ok &= abs(f["pv_to_load_w"] + f["pv_to_batt_w"] + f["curtailed_w"] - pv) < 1e-6 or f["served"] < 1
        if f["mode"] == "line":
            ok &= f["pv_to_batt_w"] + f["grid_to_batt_ac_w"] * cfg0.GRID_CHARGER_EFF <= 60 * 26.7 + 1e-6
            ok &= abs(f["pv_to_load_w"] * 0.93 + f["grid_to_load_w"] - lw) < 1e-6
    T.check(ok, "3000 random cases: no export, PV balance, 60 A limit, loads served")
    # overload curve
    T.check(not overload_trips([(4100, 900)], cfg0)[0] and overload_trips([(4300, 900)], cfg0)[0], "steady load: 102 % ok, 107 % trips")
    T.check(not overload_trips([(6000, 3)], cfg0)[0] and overload_trips([(6100, 6)], cfg0)[0], "above 150 % trips after 5.5 s")
    T.check(overload_trips([(8100, 0.3)], cfg0)[0] and not overload_trips([(8100, 0.1)], cfg0)[0], "above 200 % trips after 0.2 s")
    others = 333 + 276 + 161 + 50
    T.check(not overload_trips([(others + 23 * 230, 3), (others + 6 * 230, 900)], cfg0)[0], "AC start (23 A, 3 s) on battery does not trip")
    T.check(overload_trips([(others + 2300 + 23 * 230, 3)], cfg0)[0], "AC start while the pump runs on battery would trip (R5/R7 prevent it)")
    T.check(abs(tariff_cost(99, 2, cfg0) - (0.10 + 0.27)) < 1e-9, "block tariff split across 100 kWh")
    # reward priority order with small fixed scenarios (1 hour each = 4 steps)
    def hour(ev):
        return sum(compute_reward(ev, cfg0)[0] for _ in range(4))
    base = {"soc_pct": 60.0}
    r_trip = compute_reward({**base, "trip": 1}, cfg0)[0] + 3 * compute_reward(base, cfg0)[0]
    r_water = hour({**base, "water_outage": 1})
    r_fridge = hour({**base, "fridge_cut": 1})
    r_comfort = hour({**base, "discomfort_c": 2.0})
    r_cost = hour({**base, "cost_usd": 1.3 / 4 * cfg0.TARIFF_HIGH})
    T.check(r_trip < r_water < r_fridge < r_comfort < r_cost < 0, "reward ranks trip < water < fridge < comfort < cost",
            f"{r_trip:.1f} {r_water:.1f} {r_fridge:.1f} {r_comfort:.1f} {r_cost:.2f}")
    T.check(abs(r_comfort + 8) < 1e-9 and abs(r_cost + 3.51) < 1e-9 and r_fridge == -80 and r_water == -120, "worked check: -8, -3.5, -80, -120")

    # environment on synthetic data
    tmp = tempfile.mkdtemp(prefix="hems_env_")
    try:
        cfg = make_config(**synthetic_overrides(tmp))
        ff.make_synthetic_inputs(cfg, seed=2)
        data = ff.build_split(cfg, "train")
        env = HEMSEnv(cfg, data, mode="train", seed=5)
        o, info = env.reset(seed=11)
        T.check(o.shape == (N_FEATURES,) and np.all(np.isfinite(o)), f"reset returns {N_FEATURES} finite features")
        steps, trunc_at = 0, None
        terms_ok = True
        while True:
            o, r, te, tr, info = env.step(int(env.rng.integers(16)))
            steps += 1
            terms_ok &= set(REWARD_TERMS) <= set(info["terms"]) and np.all(np.isfinite(o))
            if info["episode_end"]:
                trunc_at = steps
                break
        T.check(trunc_at == 288 and tr and not te, "3-day episode = 288 steps, truncated (not terminated) at the end")
        T.check(terms_ok, "info carries every reward term each step")
        # determinism and exogenous independence from actions
        def roll(seed, policy):
            e = HEMSEnv(cfg, data, mode="train", seed=1)
            e.reset(seed=seed)
            obs, rew = [], []
            for i in range(100):
                o, r, *_ = e.step(policy(i))
                obs.append(o)
                rew.append(r)
            return np.array(obs), np.array(rew), e.x["grid"].copy()
        a1 = roll(3, lambda i: (i * 7) % 16)
        a2 = roll(3, lambda i: (i * 7) % 16)
        a3 = roll(3, lambda i: 0)
        T.check(np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1]), "same seed + same actions -> identical run")
        T.check(np.array_equal(a1[2], a3[2]), "grid scenario does not depend on the agent's actions")
        # only the safety controller writes the actuators
        e = HEMSEnv(cfg, data, mode="eval", seed=1)
        e.reset(seed=4)
        for i in range(50):
            e.step(i % 16)
        T.check(e.bus.write_count == 50 and e.safety.counts["steps"] == 50, "every step goes through the safety controller (1 write per step)")
        here = os.path.dirname(os.path.abspath(__file__))
        leaks = []
        for fn in ("hems_env.py", "hems_agents.py", "hems_train.py", "hems_evaluate.py", "hems_report.py"):
            p = os.path.join(here, fn)
            if os.path.exists(p):
                src = open(p, encoding="utf-8").read()
                src = src[: src.find("def selftest")] if "def selftest" in src else src
                if re.search(r"_SAFETY_TOKEN|\._write\(|_state\[", src):
                    leaks.append(fn)
        T.check(not leaks, "no module other than hems_safety touches the actuator token or bus state", str(leaks))
        try:
            e.bus._write({"pump": 1}, None)
            T.check(False, "env cannot write the actuators directly")
        except PermissionError:
            T.check(True, "env cannot write the actuators directly")
        # random actions over many episodes: zero trips, zero export
        tot = {"trips": 0, "export_kwh": 0.0, "rejected": 0}
        e = HEMSEnv(cfg, data, mode="train", seed=9)
        for ep in range(12):
            e.reset(seed=100 + ep)
            for i in range(288):
                _, _, _, _, info = e.step(int(e.rng.integers(16)))
                if info["episode_end"]:
                    break
            m = e.episode_metrics()
            for kk in tot:
                tot[kk] += m[kk]
        T.check(tot["trips"] == 0 and tot["export_kwh"] == 0 and tot["rejected"] > 0, "random policy, 12 episodes: 0 trips, 0 export (rejections happen)", str(tot))
        # grid model statistics
        ev_per_day, durs, nogrid, hours = [], [], 0, []
        for sd in range(300):
            ee = HEMSEnv(cfg, data, mode="eval", seed=sd)
            ee.reset(seed=sd, options={"user_events": False})
            evs = ee.x["events"]
            ev_per_day.append(len(evs) / 3.0)
            durs += [d for _, d in evs]
            nogrid += int(not ee.x["grid"].any())
            hours += [float(data.hour_local[ee.s0 + min(s_, 288)]) for s_, _ in evs]
        hh = np.histogram(hours, bins=4, range=(0, 24))[0] / max(len(hours), 1)
        T.check(1.6 < np.mean(ev_per_day) < 2.4 and min(durs) >= 4 and max(durs) <= 8, "grid: about 2 events/day on average, 60-120 min",
                f"{np.mean(ev_per_day):.2f}/day")
        T.check(0.03 < nogrid / 300 < 0.16, "grid: some episodes have no grid at all", f"{nogrid / 300:.3f}")
        T.check(np.all(np.abs(hh - 0.25) < 0.06), "grid arrivals spread over all hours (no schedule)", str(np.round(hh, 3)))
        # emergency semantics
        e = HEMSEnv(cfg, data, mode="eval", seed=1)
        e.reset(seed=7, options={"faults": [{"type": "comm_loss", "start": 10, "duration": 3}], "user_events": False})
        recs = []
        for i in range(20):
            o, r, te, tr, info = e.step(15)
            recs.append((info["agent_in_control"], tr, r, info["emergency_start"], info["segment_start"], info["shadow_action"]))
        T.check(recs[9][1] and recs[9][3] and recs[9][0], "the step where an emergency starts is truncated (bootstrap), not terminated")
        T.check(all(not a and r == 0.0 and sh == 15 for a, _, r, _, _, sh in recs[10:14]), "emergency steps: rule-based control, reward 0, agent in shadow mode")
        T.check(any(ss for *_, ss, _ in recs[10:16]) and recs[16][0], "after recovery a new segment starts and the agent is back")
        e = HEMSEnv(cfg, data, mode="eval", seed=1)
        e.reset(seed=7, options={"faults": [{"type": "inverter_fault", "start": 5, "duration": 2, "confirm_delay": 3}], "user_events": False})
        modes = [e.step(15)[4]["mode"] for _ in range(14)]
        T.check(modes[5:10] == ["safe"] * 5 and modes[-1] == "agent", "inverter fault: safe mode until the user confirms", str(modes))
        # water: pump fills at the configured rate and stops at full
        e = HEMSEnv(cfg, data, mode="eval", seed=1)
        e.reset(seed=3, options={"tank": 50.0, "soc": 90.0, "grid_events": [(0, 300)], "water_draw_scale": 0.0, "user_events": False,
                                 "heater_windows": []})
        for i in range(30):
            o, r, te, tr, info = e.step(encode_action(1, 1, 0, 0))
        T.check(abs(e.tank - cfg.TANK_FULL) < 1e-6 and o[IDX["pump_on"]] == 0, "pump fills at PUMP_FILL_PCT_PER_H and stops at TANK_FULL", f"{e.tank:.2f}")
        # comfort: AC on the grid lowers the humidex on a hot summer afternoon
        def hot(ac):
            ee = HEMSEnv(cfg, data, mode="eval", seed=1)
            idx = int(np.flatnonzero((data.hour_local >= 13) & (data.hour_local < 13.1))[5])
            ee.reset(seed=3, options={"start": idx, "t_in": 29.0, "rh_in": 60.0, "grid_events": [(0, 40)], "soc": 90.0, "occupancy": [1] * 50,
                                      "occ_unknown": [], "user_events": False, "heater_windows": []})
            for i in range(8):
                o, *_ = ee.step(encode_action(1, 0, ac, 0))
            return o[IDX["humidex_in"]]
        T.check(hot(1) < hot(0) - 3, "AC lowers the indoor humidex", f"{hot(1):.1f} vs {hot(0):.1f}")
        # fridge cut below 30 % on battery -> penalty; user grid -> no penalty, cost excluded
        e = HEMSEnv(cfg, data, mode="eval", seed=1)
        night = int(np.flatnonzero((data.hour_local >= 23) & (data.hour_local < 23.1))[2])
        e.reset(seed=3, options={"start": night, "soc": 29.0, "grid_events": [], "user_events": False})
        o, r, te, tr, info = e.step(0)
        T.check(info["terms"]["fridge"] == cfg.W_FRIDGE_CUT and o[IDX["fridge_on"]] == 0, "R4 cut gives the fridge penalty")
        e.reset(seed=3, options={"start": night, "soc": 50.0, "grid_events": [(0, 8)], "user_grid": [(0, 8)], "user_events": False})
        o, r, te, tr, info = e.step(encode_action(0, 0, 0, 0))
        T.check(info["executed"][0] == 1 and not info["rejected"] and info["terms"]["cost"] == 0 and info["grid_kwh"] > 0,
                "user connects the grid by hand: agent cannot open it, no penalty, cost left out of the reward")
        # generator advisory judged in hindsight
        e.reset(seed=3, options={"start": night, "soc": 31.0, "grid_events": [], "user_events": False})
        rs = [e.step(encode_action(0, 0, 0, 1))[4]["terms"]["generator"] for _ in range(8)]
        T.check(sum(rs) > 0, "generator request before a fridge cut is right in hindsight (+)")
        e.reset(seed=3, options={"start": night, "soc": 95.0, "grid_events": [], "user_events": False, "occupancy": [0] * 400})
        e.step(encode_action(0, 0, 0, 1))
        rs = [e.step(0)[4]["terms"]["generator"] for _ in range(30)]
        T.check(abs(sum(rs) - cfg.W_GEN_WRONG_BASE_PER_H * 0.25) < 1e-9, "needless generator request is wrong in hindsight (-2 per request hour)")
        # speed
        import time
        e = HEMSEnv(cfg, data, mode="train", seed=1)
        e.reset(seed=1)
        t0 = time.time()
        nst = 0
        while time.time() - t0 < 2.0:
            _, _, _, _, info = e.step(int(e.rng.integers(16)))
            nst += 1
            if info["episode_end"]:
                e.reset()
        T.check(nst / 2.0 > 500, f"speed: {nst / 2.0:.0f} env steps/s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = new_run_dir(cfg0, "selftest", "hems_env", base=out_root) if out_root else None
    return T.summary(out)


if __name__ == "__main__":
    from hems_config import _default_selftest_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
