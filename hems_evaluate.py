"""
hems_evaluate.py - final test and the fixed scenario test suite.

Test: every trained agent (best.pt chosen on the selection period), the rule-based controller, the no-forecast and
the perfect-forecast agents run the whole test period 2025-05-27..2025-12-31 once per grid-scenario seed
(TEST_SCENARIO_SEEDS, 20 fixed seeds, identical for every agent; battery and tank carry over the whole period).
Metrics per month and over the test: inverter trips, exported energy, water outage, fridge/freezer cut, discomfort
while home, grid energy and bill, PV used, rejected actions per day, switching per day, generator requests right.

Scenario suite (11 fixed, seeded situations with pass criteria): night pumping on the grid, low SOC water, summer night
battery, water vs comfort vs cost, water heater, cloudy/sunny tomorrow, occupancy, faults (R11-R13), manual grid,
grid at unusual hours. Each trained agent and the rule-based controller run all of them.

Outputs in OUTPUT_DIR/eval/eval_<time>/: test_overall.csv, test_monthly.csv, scenarios.csv, scenario_details.json,
selection.csv, trajectories/*.csv, summary.json (+ settings.json).

Run in Colab (after training):
    !python hems_evaluate.py                    # all finished runs + rule-based controller
    !python hems_evaluate.py --runs <run_dir> <run_dir>
    !python hems_evaluate.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import hems_forecast_feed as ff
from hems_agents import RuleBasedAgent, load_agent
from hems_config import Config, SelfTest, add_common_args, config_from_args, make_config, new_run_dir, synthetic_overrides
from hems_env import HEMSEnv, run_episodes
from hems_safety import encode_action

METRIC_COLS = ["trips", "export_kwh", "water_outage_min", "fridge_cut_min", "discomfort_ch", "grid_kwh", "grid_kwh_agent", "bill_usd",
               "bill_usd_agent", "pv_used_pct", "rejected_per_day", "toggles_per_day", "gen_requests", "gen_right_pct",
               "lv_shutdown_min", "reward", "days"]


# =====================================================================================
# Agents to evaluate
# =====================================================================================
def finished_runs(cfg: Config, run_dirs=None) -> list:
    dirs = run_dirs or sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, "train", "*")))
    out = []
    for d in dirs:
        dj = os.path.join(d, "done.json")
        if os.path.exists(dj) and os.path.exists(os.path.join(d, "best.pt")):
            info = json.load(open(dj))
            info["run_dir"] = d
            out.append(info)
    return out


def agent_label(info) -> str:
    fm = info["forecast_mode"]
    return f"{info['algo']}_s{info['seed']}" + ("" if fm == "ensemble" else f"_{fm}")


class _Policy:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, obs, masks):
        return self.agent.act_batch(obs, masks, deterministic=True)


class FixedPolicy:
    """Probe policy that always proposes the same action (used to test the safety mechanisms)."""

    def __init__(self, action):
        self.action = int(action)

    def act_batch(self, obs, masks=None, deterministic=True):
        return np.full(len(obs), self.action)

    def act(self, obs, mask=None, deterministic=True):
        return self.action


# =====================================================================================
# Test period
# =====================================================================================
def test_specs(cfg: Config, data) -> list:
    return [{"seed": int(s), "options": {"start": 0, "steps": data.n - 1, "soc": cfg.TEST_INIT_SOC, "tank": cfg.TEST_INIT_TANK}}
            for s in cfg.TEST_SCENARIO_SEEDS]


def evaluate_test(cfg: Config, agent, label: str, forecast_mode: str, data):
    res = run_episodes(_Policy(agent), cfg, data, test_specs(cfg, data), forecast_mode=forecast_mode, track_monthly=True)
    overall, monthly = [], []
    for r in res:
        sd = r["spec"]["seed"]
        overall.append({"agent": label, "scenario_seed": sd, **{k: r["metrics"].get(k) for k in METRIC_COLS}})
        for ym, m in r["monthly"].items():
            monthly.append({"agent": label, "scenario_seed": sd, "month": ym, **{k: m.get(k) for k in METRIC_COLS}})
    return pd.DataFrame(overall), pd.DataFrame(monthly)


def trajectory(cfg: Config, agent, forecast_mode, data, days=3, seed=None):
    """One 3-day window in the hottest part of the test period, logged step by step (for plots)."""
    n = int(days * 96)
    hot = int(np.argmax(np.convolve(data.t_out, np.ones(n) / n, mode="valid")[: max(1, data.n - n - 1)]))
    start = (hot // 96) * 96
    sp = [{"seed": seed or cfg.TEST_SCENARIO_SEEDS[0], "options": {"start": start, "steps": n, "soc": 60.0, "tank": 60.0}}]
    r = run_episodes(_Policy(agent), cfg, data, sp, forecast_mode=forecast_mode, keep_log=True)[0]
    return pd.DataFrame(r["log"])


# =====================================================================================
# Scenario suite
# =====================================================================================
def _pick_start(data, hour: float, steps: int, months=(6, 7, 8), how="hot"):
    h = data.hour_local
    cand = np.flatnonzero((np.abs(h - hour) < 1e-6) & np.isin(data.month_local, months))
    cand = cand[cand + steps + 2 < data.n]
    if len(cand) == 0:
        cand = np.flatnonzero(np.abs(h - hour) < 1e-6)
        cand = cand[cand + steps + 2 < data.n]
    if how == "hot":
        score = np.array([data.t_out[c: c + max(steps, 8)].mean() for c in cand])
    elif how == "sunny":
        score = np.array([data.pv_w[c: c + max(steps, 8)].sum() for c in cand])
    else:
        return int(cand[len(cand) // 2])
    return int(cand[int(np.argmax(score))])


def _sim(agent, cfg, data, start, steps, opts=None, seed=5000, forecast_mode="ensemble"):
    o = {"start": start, "steps": steps, "user_events": False, "occ_unknown": [], "heater_windows": []}
    o.update(opts or {})
    env = HEMSEnv(cfg, data, mode="eval", forecast_mode=forecast_mode, seed=seed, keep_log=True)
    obs, _ = env.reset(seed=seed, options=o)
    for _ in range(steps):
        a = agent.act_batch(obs[None], env.action_mask()[None] if cfg.USE_ACTION_MASK else None, deterministic=True)[0]
        obs, r, te, tr, info = env.step(int(a))
        if info["episode_end"]:
            break
    return pd.DataFrame(env.log), env


def _home(n):
    return [1] * (n + 1)


def run_scenarios(cfg: Config, agent, data, forecast_mode="ensemble") -> list:
    R = []
    pump_w = cfg.PUMP_RUN_A * cfg.GRID_V * cfg.PUMP_PF

    def add(sid, name, passed, **details):
        R.append({"scenario": sid, "name": name, "passed": bool(passed), "details": details})

    # 1 Night, tank below 25 %, the grid arrives -> contactor closed and pumping on the grid
    st = _pick_start(data, 23.0, 16)
    L, _ = _sim(agent, cfg, data, st, 16, {"soc": 60.0, "tank": 20.0, "grid_events": [(4, 12)], "occupancy": _home(16)}, seed=5001, forecast_mode=forecast_mode)
    g = L.iloc[4:12]
    closed = bool(L["contactor"].iloc[4:6].max() == 1)
    pumped = int(((g["pump"] == 1) & (g["contactor"] == 1) & (g["grid"] == 1)).sum())
    add(1, "night, tank < 25 %, grid arrives -> contactor closed, pump on the grid", closed and (pumped >= 2 or L["tank"].max() >= cfg.TANK_FULL - 0.5),
        contactor_closed_at_arrival=closed, pump_steps_on_grid=pumped, tank_end=float(L["tank"].iloc[-1]))

    # 2 SOC below 35 %, water needed, no grid -> no pump on battery; pumps on PV surplus or when the grid comes
    st = _pick_start(data, 7.0, 40, how="sunny")
    L, _ = _sim(agent, cfg, data, st, 40, {"soc": 33.0, "tank": 20.0, "grid_events": [(24, 32)], "occupancy": _home(40)}, seed=5002, forecast_mode=forecast_mode)
    line = (L["grid"] == 1) & (L["contactor"] == 1)
    bad = int(((L["pump"] == 1) & ~line & (L["soc0"] < cfg.SOC_RESTRICT)).sum())
    good = int(((L["pump"] == 1) & (line | ((L["soc0"] >= cfg.SOC_RESTRICT) & (L["pv_w"] >= pump_w)))).sum())
    add(2, "SOC < 35 %, no grid -> no pump on battery; pumps on PV surplus or grid", bad == 0 and good >= 1, pump_on_battery_below_35=bad, pump_steps_ok=good)

    # 3 Summer night, battery projected to empty before 7 AM
    st = _pick_start(data, 21.0, 40)
    hot = {"soc": 55.0, "tank": 70.0, "occupancy": _home(40), "t_out_offset": 2.0}
    L, _ = _sim(agent, cfg, data, st, 40, {**hot, "grid_events": [(10, 18)]}, seed=5003, forecast_mode=forecast_mode)
    closed = bool(L["contactor"].iloc[10:12].max() == 1)
    cut = int((L["fridge"] == 0).sum())
    add(3, "summer night, battery short (grid comes) -> contactor closed at first arrival, AC limited, fridge kept", closed and cut == 0,
        contactor_closed_at_arrival=closed, fridge_cut_steps=cut, ac_steps_on_battery=int(((L["ac"] == 1) & (L["grid"] == 0)).sum()),
        soc_end=float(L["soc"].iloc[-1]))
    L, _ = _sim(agent, cfg, data, st, 40, {**hot, "grid_events": []}, seed=5003, forecast_mode=forecast_mode)
    first_cut = int(np.argmax((L["fridge"] == 0).values)) if (L["fridge"] == 0).any() else len(L)
    gen_before = bool(L["gen"].iloc[:max(first_cut, 1)].max() == 1) if first_cut > 0 else False
    needed = (L["fridge"] == 0).any() or (L["r_long_discomfort"] < 0).any()
    add("3b", "summer night, no grid comes -> generator advisory before the fridge cut", gen_before or not needed,
        generator_before_cut=gen_before, first_cut_step=first_cut)

    # 4 Tank 30-40 %, AC needed, battery will not last -> grid; water before comfort before cost; generator advisory
    st = _pick_start(data, 20.0, 40)
    base4 = {"soc": 50.0, "tank": 35.0, "occupancy": _home(40), "t_out_offset": 2.0}
    L, _ = _sim(agent, cfg, data, st, 40, {**base4, "grid_events": [(8, 14)]}, seed=5004, forecast_mode=forecast_mode)
    g = L.iloc[8:14]
    closed_frac = float((g["contactor"] == 1).mean())
    pump_grid = int(((g["pump"] == 1) & (g["contactor"] == 1)).sum())
    hot_on_grid = bool((g["humidex"] > g["thr"]).any() or (L["humidex"].iloc[7] > L["thr"].iloc[7]))
    ac_grid = int(((g["ac"] == 1) & (g["contactor"] == 1)).sum())
    add(4, "tank 30-40 %, AC needed, battery short -> grid; water first, then comfort", closed_frac >= 0.8 and pump_grid >= 1 and (ac_grid >= 1 or not hot_on_grid),
        contactor_closed_frac=closed_frac, pump_steps_on_grid=pump_grid, ac_steps_on_grid=ac_grid)
    L, _ = _sim(agent, cfg, data, st, 40, {**base4, "soc": 40.0, "grid_events": []}, seed=5004, forecast_mode=forecast_mode)
    first_cut = int(np.argmax((L["fridge"] == 0).values)) if (L["fridge"] == 0).any() else len(L)
    gen_before = bool(L["gen"].iloc[:max(first_cut, 1)].max() == 1) if first_cut > 0 else False
    add("4b", "no grid and a fridge cut is coming -> generator advisory", gen_before or first_cut == len(L), generator_before_cut=gen_before, first_cut_step=first_cut)

    # 5 Water heater on the grid -> no pump, no AC; grid drops -> heater off at once (agent and an adversarial probe)
    st = _pick_start(data, 5.75, 8)
    ok5, det5 = True, {}
    for nm, pol in (("agent", agent), ("probe_all_on", FixedPolicy(encode_action(1, 1, 1, 0)))):
        L, env = _sim(pol, cfg, data, st, 8, {"soc": 80.0, "tank": 50.0, "grid_events": [(0, 4)], "heater_windows": cfg.HEATER_WINDOWS_H}, seed=5005, forecast_mode=forecast_mode)
        hrows = L[L["heater"] == 1]
        ok = len(hrows) > 0 and int(hrows["pump"].sum() + hrows["ac"].sum()) == 0 and int(L["heater"].iloc[4]) == 0 and int(L["trip"].sum()) == 0
        ok5 &= ok
        det5[nm] = {"heater_steps": int(len(hrows)), "pump_or_ac_with_heater": int(hrows["pump"].sum() + hrows["ac"].sum()),
                    "heater_after_grid_drop": int(L["heater"].iloc[4]), "trips": int(L["trip"].sum())}
    add(5, "water heater on the grid -> no pump/AC; grid drops -> heater off at once", ok5, **det5)

    # 6 / 7 Cloudy vs sunny tomorrow (same evening, only the Next-Day forecast differs)
    st = _pick_start(data, 15.0, 64, how="sunny")
    nd = [f"nd_ens_{k}" for k in range(1, 9)]
    ns = [f"nd_spr_{k}" for k in range(1, 9)]
    cloudy = {**{f: (0.2, 0.0) for f in nd}, **{f: (1.0, 250.0) for f in ns}, "e_tom_kwh": (0.2, 0.0), "e_tom_spr_kwh": (1.0, 3.0), "cloud_fc_tom": (0.0, 0.85)}
    sunny = {**{f: (1.0, 0.0) for f in nd}, **{f: (0.5, 0.0) for f in ns}, "e_tom_kwh": (1.0, 0.0), "e_tom_spr_kwh": (0.5, 0.0), "cloud_fc_tom": (0.0, 0.05)}
    common = {"soc": 70.0, "tank": 80.0, "occupancy": _home(64), "grid_events": [(12, 18)]}
    Lc, _ = _sim(agent, cfg, data, st, 64, {**common, "fc_transform": cloudy}, seed=5006, forecast_mode=forecast_mode)
    Ls, _ = _sim(agent, cfg, data, st, 64, {**common, "fc_transform": sunny}, seed=5006, forecast_mode=forecast_mode)
    gc = Lc.iloc[12:18]
    closed_c = float((gc["contactor"] == 1).mean())
    soc_c, soc_s = float(Lc["soc"].iloc[-1]), float(Ls["soc"].iloc[-1])
    dis = lambda L: float((-L["batt_dc_w"].clip(upper=0)).sum() * cfg.dt_h / 1000.0)
    gkwh = lambda L: float((L["grid_w"]).sum() * cfg.dt_h / 1000.0)
    add(6, "cloudy tomorrow -> save battery today, take grid charging", closed_c >= 0.8 and soc_c >= soc_s - 1.0,
        contactor_closed_frac_cloudy=closed_c, soc_end_cloudy=soc_c, soc_end_sunny=soc_s)
    add(7, "sunny tomorrow -> may use more battery tonight", dis(Ls) >= dis(Lc) - 0.1,
        battery_kwh_sunny=dis(Ls), battery_kwh_cloudy=dis(Lc), grid_kwh_sunny=gkwh(Ls), grid_kwh_cloudy=gkwh(Lc))

    # 8 Nobody home -> no AC; occupancy unknown -> last known state kept
    st = _pick_start(data, 13.0, 16)
    hotday = {"soc": 90.0, "tank": 80.0, "grid_events": [(0, 17)], "t_in": 29.5, "rh_in": 60.0}
    La, _ = _sim(agent, cfg, data, st, 16, {**hotday, "occupancy": [0] * 17}, seed=5008, forecast_mode=forecast_mode)
    Lb, _ = _sim(agent, cfg, data, st, 16, {**hotday, "occupancy": _home(16), "occ_unknown": [(4, 17)]}, seed=5008, forecast_mode=forecast_mode)
    Lc8, _ = _sim(agent, cfg, data, st, 16, {**hotday, "occupancy": [0] * 17, "occ_unknown": [(4, 17)]}, seed=5008, forecast_mode=forecast_mode)
    b = Lb.iloc[4:16]
    c8 = Lc8.iloc[4:16]
    ok8 = int(La["ac"].sum()) == 0 and bool((b["eff_home"] == 1).all()) and int(b["ac"].sum()) >= 1 and bool((c8["eff_home"] == 0).all()) and int(c8["ac"].sum()) == 0
    add(8, "nobody home -> no AC; unknown occupancy keeps the last known state", ok8, ac_steps_away=int(La["ac"].sum()),
        unknown_after_home_eff_home=bool((b["eff_home"] == 1).all()), ac_steps_unknown_after_home=int(b["ac"].sum()),
        unknown_after_away_eff_home=bool((c8["eff_home"] == 0).all()), ac_steps_unknown_after_away=int(c8["ac"].sum()))

    # 9 Faults: MQTT loss, stale sensor, inverter fault code, MCB trip, total power loss
    st = _pick_start(data, 11.0, 14)
    fb = {"soc": 70.0, "tank": 40.0, "grid_events": [(0, 15)], "occupancy": _home(14), "t_in": 29.0, "rh_in": 60.0}
    sub = {}
    L, env = _sim(agent, cfg, data, st, 14, {**fb, "faults": [{"type": "comm_loss", "start": 4, "duration": 4}]}, seed=5009, forecast_mode=forecast_mode)
    e = L.iloc[4:8]
    sub["mqtt_loss"] = bool((e["mode"] != "agent").all() and e["pump"].sum() == 0 and e["ac"].sum() == 0 and (e["fridge"] == 1).all()
                            and (L["mode"].iloc[9:] == "agent").all() and any(a["code"] == "EMERGENCY" for a in env.safety.alerts))
    L, env = _sim(agent, cfg, data, st, 14, {**fb, "faults": [{"type": "sensor_stale", "target": "tank", "start": 4, "duration": 4}]}, seed=5009, forecast_mode=forecast_mode)
    e = L.iloc[4:8]
    sub["stale_sensor"] = bool(e["pump"].sum() == 0 and (e["mode"] == "agent").all() and any(a["code"] == "R12" for a in env.safety.alerts))
    L, env = _sim(agent, cfg, data, st, 14, {**fb, "faults": [{"type": "inverter_fault", "start": 4, "duration": 2, "confirm_delay": 2}]}, seed=5009, forecast_mode=forecast_mode)
    e = L.iloc[4:8]
    sub["inverter_fault"] = bool((e["mode"] == "safe").all() and e["pump"].sum() == 0 and e["ac"].sum() == 0 and (L["mode"].iloc[10:] == "agent").all()
                                 and any(a["code"] == "R11" for a in env.safety.alerts))
    L, env = _sim(FixedPolicy(encode_action(1, 1, 0, 0)), cfg, data, st, 14, {**fb, "tank": 20.0, "faults": [{"type": "mcb_trip", "target": "pump", "start": 2, "duration": 6}]},
                  seed=5009, forecast_mode=forecast_mode)
    sub["mcb_trip"] = bool(any(a["code"] == "R13" for a in env.safety.alerts) and L["pump"].iloc[4:8].sum() == 0 and L["pump"].iloc[9:].sum() >= 1)
    L, env = _sim(agent, cfg, data, st, 14, {**fb, "faults": [{"type": "pi_restart", "start": 4, "duration": 2}]}, seed=5009, forecast_mode=forecast_mode)
    sub["power_loss_restart"] = bool((L["mode"].iloc[4:8] != "agent").all() and L["pump"].iloc[4:6].sum() == 0 and L["ac"].iloc[4:6].sum() == 0
                                     and (L["mode"].iloc[8:] == "agent").all())
    add(9, "faults -> fail-safe behaviour (R11-R13), rule-based control, return after recovery", all(sub.values()), **sub)

    # 10 The user connects the grid by hand -> the agent cannot open it, no penalty
    st = _pick_start(data, 10.0, 12, how="any")
    ok10, det10 = True, {}
    for nm, pol in (("agent", agent), ("probe_open", FixedPolicy(0))):
        L, _ = _sim(pol, cfg, data, st, 12, {"soc": 60.0, "tank": 60.0, "grid_events": [(0, 12)], "user_grid": [(0, 8)]}, seed=5010, forecast_mode=forecast_mode)
        u = L.iloc[0:8]
        ok = bool((u["contactor"] == 1).all() and (u["r_cost"] == 0).all())
        if nm == "probe_open":  # proposing to open the contactor against the user is never penalized
            ok = ok and bool((u["r_reject"] == 0).all())
        ok10 &= ok
        det10[nm] = {"contactor_closed_all": bool((u["contactor"] == 1).all()), "reject_penalty": float(u["r_reject"].sum()), "cost_penalty": float(u["r_cost"].sum())}
    add(10, "manual grid connection -> agent cannot open it, no penalty", ok10, **det10)

    # 11 Grid at unusual hours and long periods without grid -> no behaviour tied to the clock
    closed = {}
    for H in (21, 22, 23, 0, 1, 2, 3, 4):  # dark hours: charging is worth the same whatever the clock says
        st = _pick_start(data, float((H - 1) % 24), 10, how="any")
        L, _ = _sim(agent, cfg, data, st, 10, {"soc": 40.0, "tank": 50.0, "grid_events": [(4, 10)], "occupancy": _home(10)}, seed=5011, forecast_mode=forecast_mode)
        closed[H] = bool(L["contactor"].iloc[4:6].max() == 1)
    st = _pick_start(data, 6.0, 192, how="sunny")
    L, _ = _sim(agent, cfg, data, st, 192, {"soc": 80.0, "tank": 60.0, "grid_events": [], "occupancy": _home(192)}, seed=5012, forecast_mode=forecast_mode)
    long_ok = int(L["trip"].sum()) == 0 and int((L["tank"] < cfg.TANK_MIN).sum()) == 0
    add(11, "grid at unusual hours / 48 h without grid -> no clock-tied behaviour", sum(closed.values()) >= 7 and long_ok,
        closed_at_arrival_by_hour=closed, long_no_grid_trips=int(L["trip"].sum()), long_no_grid_water_outage_steps=int((L["tank"] < cfg.TANK_MIN).sum()))
    return R


def scenario_summary(results) -> dict:
    by = {}
    for r in results:
        sid = str(r["scenario"]).rstrip("b")
        by[sid] = by.get(sid, True) and r["passed"]
    return {"passed": sum(by.values()), "total": len(by), "all_pass": all(by.values()), "by_scenario": by}


# =====================================================================================
# Main evaluation
# =====================================================================================
def evaluate_all(cfg: Config, run_dirs=None, out_dir=None, include_rule=True, verbose=True) -> str:
    out = out_dir or new_run_dir(cfg, "eval", "eval")
    os.makedirs(os.path.join(out, "trajectories"), exist_ok=True)
    data = ff.load_or_build(cfg, "test")
    runs = finished_runs(cfg, run_dirs)
    agents = []
    if include_rule:
        agents.append(("rule_based", RuleBasedAgent(cfg), "ensemble", {"algo": "rule", "seed": None, "forecast_mode": "ensemble"}))
    for info in runs:
        ag = load_agent(os.path.join(info["run_dir"], "best.pt"), cfg)
        agents.append((agent_label(info), ag, info["forecast_mode"], info))
    overall, monthly, scen, details, sel = [], [], [], {}, []
    for label, ag, fm, info in agents:
        if verbose:
            print(f"test: {label} ({len(cfg.TEST_SCENARIO_SEEDS)} grid scenarios) ...", flush=True)
        o, m = evaluate_test(cfg, ag, label, fm, data)
        for df in (o, m):
            df.insert(1, "algo", info["algo"])
            df.insert(2, "seed", info["seed"])
            df.insert(3, "forecast_mode", fm)
        overall.append(o)
        monthly.append(m)
        trajectory(cfg, ag, fm, data).to_csv(os.path.join(out, "trajectories", f"{label}.csv"), index=False)
        res = run_scenarios(cfg, ag, data, fm)
        details[label] = res
        summ = scenario_summary(res)
        for r in res:
            scen.append({"agent": label, "algo": info["algo"], "seed": info["seed"], "forecast_mode": fm, "scenario": str(r["scenario"]),
                         "name": r["name"], "passed": r["passed"]})
        if verbose:
            print(f"   trips {o['trips'].sum():.0f}, export {o['export_kwh'].sum():.2f} kWh, bill ${o['bill_usd_agent'].mean():.1f}, "
                  f"discomfort {o['discomfort_ch'].mean():.1f} C-h, scenarios {summ['passed']}/{summ['total']}", flush=True)
        if info.get("best"):
            sel.append({"agent": label, "algo": info["algo"], "seed": info["seed"], "forecast_mode": fm, "run_dir": info["run_dir"],
                        **{k: v for k, v in info["best"].items() if k.startswith("select_") or k == "steps"}})
    pd.concat(overall).to_csv(os.path.join(out, "test_overall.csv"), index=False)
    pd.concat(monthly).to_csv(os.path.join(out, "test_monthly.csv"), index=False)
    pd.DataFrame(scen).to_csv(os.path.join(out, "scenarios.csv"), index=False)
    json.dump(details, open(os.path.join(out, "scenario_details.json"), "w"), indent=1, default=str)
    pd.DataFrame(sel).to_csv(os.path.join(out, "selection.csv"), index=False)
    ov = pd.concat(overall)
    chosen = None
    if sel:
        sdf = pd.DataFrame(sel)
        sdf = sdf[sdf["forecast_mode"] == "ensemble"]
        if len(sdf):
            chosen = str(sdf.sort_values("select_reward", ascending=False)["agent"].iloc[0])
    summary = {"test_split": cfg.SPLIT_TEST, "agents": [a[0] for a in agents], "chosen_agent": chosen,
               "zero_trips": bool(ov["trips"].sum() == 0), "zero_export": bool(ov["export_kwh"].sum() == 0),
               "chosen_agent_scenarios": scenario_summary(details[chosen]) if chosen else None,
               "rule_based_scenarios": scenario_summary(details["rule_based"]) if include_rule else None}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    if verbose:
        print(json.dumps(summary, indent=1, default=str))
        print("evaluation saved in", out)
    return out


def selftest(out_root: str | None = None) -> bool:
    T = SelfTest("hems_evaluate")
    tmp = tempfile.mkdtemp(prefix="hems_ev_")
    try:
        cfg = make_config(**{**synthetic_overrides(tmp), "DEVICE": "cpu"})
        ff.make_synthetic_inputs(cfg, seed=6)
        data = ff.load_or_build(cfg, "test", verbose=False)
        rb = RuleBasedAgent(cfg)
        o, m = evaluate_test(cfg, rb, "rule_based", "ensemble", data)
        T.check(len(o) == len(cfg.TEST_SCENARIO_SEEDS) and abs(o["days"].iloc[0] - (data.n - 1) / 96) < 1e-6, "test: one continuous run per grid seed over the whole period")
        T.check(o["trips"].sum() == 0 and o["export_kwh"].sum() == 0, "rule-based controller: zero trips, zero export on the test")
        T.check(set(m["month"]) == {"2024-07"} and abs(m.groupby("scenario_seed")["grid_kwh"].sum().values - o["grid_kwh"].values).max() < 1e-6,
                "monthly metrics add up to the totals")
        res = run_scenarios(cfg, rb, data)
        s = scenario_summary(res)
        T.check(s["total"] == 11, "scenario suite has 11 scenarios", str(s["by_scenario"]))
        for sid in ("5", "9", "10"):
            T.check(s["by_scenario"][sid], f"rule-enforced scenario {sid} passes with the rule-based controller")
        # adversarial probes cannot break the safety-enforced scenarios
        for a in range(16):
            r = run_scenarios(cfg, FixedPolicy(a), data)
            sb = scenario_summary(r)["by_scenario"]
            ok = sb["5"] and sb["10"]
            if not ok:
                break
        T.check(ok, "every fixed action (0..15): heater and manual-grid scenarios still pass (safety controller)")
        # full pipeline with a short training
        from hems_train import train
        c2 = cfg.replace(TRAIN_STEPS=1024, EVAL_EVERY=1024, OBS_NORM_STEPS=500, SELECT_EPISODES_MAX=1, PPO={**cfg.PPO, "n_steps": 64})
        train(c2, "ppo", 42, verbose=False)
        out = evaluate_all(c2, verbose=False)
        files = [f for f in ("test_overall.csv", "test_monthly.csv", "scenarios.csv", "summary.json", "selection.csv", "settings.json")
                 if os.path.exists(os.path.join(out, f))]
        T.check(len(files) == 6 and len(os.listdir(os.path.join(out, "trajectories"))) == 2, "evaluation folder with all outputs", str(files))
        summ = json.load(open(os.path.join(out, "summary.json")))
        T.check(summ["chosen_agent"] == "ppo_s42" and summ["zero_trips"] and summ["zero_export"], "chosen agent from the selection period; zero trips/export")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = new_run_dir(make_config(), "selftest", "hems_evaluate", base=out_root) if out_root else None
    return T.summary(out)


if __name__ == "__main__":
    from hems_config import _default_selftest_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs", nargs="*", default=None, help="training run folders (default: all finished runs)")
    ap.add_argument("--no-rule", action="store_true")
    ap.add_argument("--out", default=None)
    add_common_args(ap)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
    cfg = config_from_args(a)
    ff.ensure_inputs(cfg, bool(a.synthetic))
    evaluate_all(cfg, a.runs, include_rule=not a.no_rule)
