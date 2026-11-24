"""
hems_train.py - trains PPO, SAC-Discrete, DDQN and A2C under identical conditions, with checkpoint selection.

  1. Train on 2024-03-15..2025-02-28 (CV folds 1-4 validation forecasts), 3-day episodes, random grid,
     HARD_EPISODE_FRAC hard episodes, EMERGENCY_EPISODE_FRAC simulated emergencies (never learned from).
  2. Every EVAL_EVERY steps evaluate on 2025-03-01..2025-05-26 (fold 5) with a fixed set of 3-day episodes
     and grid scenarios; keep the best checkpoint (best.pt). The deployed model is chosen here, never on the test.
  3. The test (2025-05-27..2025-12-31) is run once by hems_evaluate.py.

Each run writes OUTPUT_DIR/train/<algo>_s<seed>_<forecast>_<time>/ with settings.json, obs_norm.json,
log_train.csv, log_select.csv, best.pt, last.pt (full state for resuming) and done.json.
If Colab disconnects, run the same command again: an unfinished run with the same algo/seed/forecast resumes.

Run in Colab:
    !python hems_train.py --selftest
    !python hems_train.py --algo ppo --seed 42                  # one training (repeat for sac, ddqn, a2c and seeds 43, 44)
    !python hems_train.py --all                                 # 4 algorithms x 3 seeds, one after the other
    !python hems_train.py --algo ppo --seed 42 --forecast none     # no-forecast agent (comparison)
    !python hems_train.py --algo ppo --seed 42 --forecast perfect  # perfect-forecast agent (upper bound)
    !python hems_train.py --plan                                # prints the commands and the time estimate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import hems_forecast_feed as ff
from hems_agents import ObsNormalizer, fit_normalizer, load_agent, make_agent
from hems_config import (Config, SelfTest, add_common_args, config_from_args, make_config, new_run_dir, set_global_seed,
                         synthetic_overrides)
from hems_env import HEMSEnv, VecEnv, run_episodes


def selection_specs(cfg: Config, data) -> list:
    """Fixed 3-day episodes tiled over the selection period, each with its own fixed grid-scenario seed."""
    n = cfg.episode_steps
    starts = list(range(0, max(1, data.n - n - 1), n))[: cfg.SELECT_EPISODES_MAX]
    return [{"seed": cfg.SELECT_SCENARIO_SEED + j, "options": {"start": int(s), "steps": n, "soc": 60.0, "tank": 60.0}}
            for j, s in enumerate(starts)]


def evaluate_selection(agent, cfg: Config, data, specs, forecast_mode) -> dict:
    res = run_episodes(lambda o, m: agent.act_batch(o, m, deterministic=True), cfg, data, specs, forecast_mode=forecast_mode)
    df = pd.DataFrame([r["metrics"] for r in res])
    out = {"select_reward": float(df["reward"].mean())}
    for k in ("trips", "water_outage_min", "fridge_cut_min", "discomfort_ch", "bill_usd_agent", "grid_kwh", "pv_used_pct",
              "rejected_per_day", "toggles_per_day", "export_kwh"):
        out[f"select_{k}"] = float(df[k].mean())
    return out


def run_name(algo, seed, forecast):
    return f"{algo}_s{seed}_{forecast}"


def find_unfinished(cfg: Config, algo, seed, forecast):
    pat = os.path.join(cfg.OUTPUT_DIR, "train", run_name(algo, seed, forecast) + "_*")
    runs = sorted(d for d in glob.glob(pat) if os.path.isdir(d))
    for d in reversed(runs):
        if os.path.exists(os.path.join(d, "last.pt")) and not os.path.exists(os.path.join(d, "done.json")):
            return d
    return None


def find_finished(cfg: Config, algo, seed, forecast):
    pat = os.path.join(cfg.OUTPUT_DIR, "train", run_name(algo, seed, forecast) + "_*")
    return [d for d in sorted(glob.glob(pat)) if os.path.exists(os.path.join(d, "done.json"))]


def train(cfg: Config, algo: str, seed: int, forecast: str = "ensemble", fresh: bool = False, debug: bool = False,
          data_tr=None, data_sel=None, verbose: bool = True):
    set_global_seed(seed)
    data_tr = data_tr or ff.load_or_build(cfg, "train")
    data_sel = data_sel or ff.load_or_build(cfg, "select")
    run = None if fresh else find_unfinished(cfg, algo, seed, forecast)
    if run:
        agent = load_agent(os.path.join(run, "last.pt"), cfg, full=True)
        best = json.load(open(os.path.join(run, "best.json"))) if os.path.exists(os.path.join(run, "best.json")) else {"select_reward": -np.inf}
        if verbose:
            print(f"resuming {run} at {agent.total_steps} steps")
    else:
        run = new_run_dir(cfg, "train", run_name(algo, seed, forecast))
        t0 = time.time()
        norm = fit_normalizer(cfg, data_tr, forecast, cfg.OBS_NORM_STEPS, seed)
        with open(os.path.join(run, "obs_norm.json"), "w") as f:
            json.dump({**norm.state_dict(), "fitted_on": "train split", "steps": cfg.OBS_NORM_STEPS, "seconds": time.time() - t0}, f)
        agent = make_agent(algo, cfg, norm=norm, seed=seed)
        best = {"select_reward": -np.inf}
    agent.debug = debug
    hp = {"ppo": cfg.PPO, "a2c": cfg.A2C, "ddqn": cfg.DDQN, "sac": cfg.SAC}[algo]
    envs = [HEMSEnv(cfg, data_tr, mode="train", forecast_mode=forecast, seed=seed * 1000 + i + agent.total_steps) for i in range(hp["n_envs"])]
    venv = VecEnv(envs, seed=seed + agent.total_steps)
    specs = selection_specs(cfg, data_sel)
    counters = {"env_steps": 0, "agent_steps": 0, "emergency_steps": 0}
    if debug:
        orig = venv.step

        def counted(actions, _o=orig):
            out = _o(actions)
            for info in out[4]:
                counters["env_steps"] += 1
                counters["agent_steps"] += int(info["agent_in_control"])
                counters["emergency_steps"] += int(not info["agent_in_control"])
            return out
        venv.step = counted
    log_tr = os.path.join(run, "log_train.csv")
    log_sel = os.path.join(run, "log_select.csv")
    while agent.total_steps < cfg.TRAIN_STEPS:
        chunk = int(min(cfg.EVAL_EVERY, cfg.TRAIN_STEPS - agent.total_steps))
        t0 = time.time()
        agent.train_steps(venv, chunk)
        dt = time.time() - t0
        eps = pd.DataFrame(venv.completed)
        venv.completed = []
        row = {"steps": agent.total_steps, "seconds": dt, "steps_per_s": chunk / max(dt, 1e-9), "episodes": len(eps), "updates": agent.updates,
               **{f"train_{k}": float(eps[k].mean()) for k in ("reward", "trips", "rejected_per_day", "water_outage_min", "fridge_cut_min",
                                                             "discomfort_ch", "bill_usd_agent", "emergency_steps") if len(eps)},
               **{f"stat_{k}": v for k, v in agent.stats.items()}}
        pd.DataFrame([row]).to_csv(log_tr, mode="a", header=not os.path.exists(log_tr), index=False)
        sel = evaluate_selection(agent, cfg, data_sel, specs, forecast)
        sel["steps"] = agent.total_steps
        improved = sel["select_reward"] > best["select_reward"]
        sel["best"] = improved
        pd.DataFrame([sel]).to_csv(log_sel, mode="a", header=not os.path.exists(log_sel), index=False)
        if improved:
            best = dict(sel)
            agent.save(os.path.join(run, "best.pt"), extra={"selection": sel, "forecast_mode": forecast, "algo": algo, "seed": seed}, full=False)
            json.dump(best, open(os.path.join(run, "best.json"), "w"), indent=1)
        agent.save(os.path.join(run, "last.pt"), extra={"forecast_mode": forecast, "algo": algo, "seed": seed}, full=True)
        if verbose:
            print(f"[{algo} s{seed} {forecast}] {agent.total_steps:>9d} steps | {row['steps_per_s']:6.0f} steps/s | "
                  f"train R {row.get('train_reward', float('nan')):8.1f} rej/d {row.get('train_rejected_per_day', float('nan')):5.1f} | "
                  f"select R {sel['select_reward']:8.1f}{' *' if improved else ''}", flush=True)
    done = {"algo": algo, "seed": seed, "forecast_mode": forecast, "total_steps": agent.total_steps, "best": best,
            "run_dir": run, "debug_counters": counters}
    if debug:
        done["stored_all_in_control"] = bool(all(agent.debug_stored_in_control))
        done["stored"] = len(agent.debug_stored_in_control)
    json.dump(done, open(os.path.join(run, "done.json"), "w"), indent=1, default=str)
    return run, agent, done


def plan(cfg: Config):
    print("Training plan (each command can be run in its own Colab session; it resumes if interrupted):")
    for algo in cfg.ALGOS:
        for seed in cfg.TRAIN_SEEDS:
            print(f"  !python hems_train.py --algo {algo} --seed {seed}")
    print("Comparison runs, with the algorithm chosen on the selection period (e.g. ppo):")
    print("  !python hems_train.py --algo ppo --seed 42 --forecast none")
    print("  !python hems_train.py --algo ppo --seed 42 --forecast perfect")


def selftest(out_root: str | None = None) -> bool:
    T = SelfTest("hems_train")
    tmp = tempfile.mkdtemp(prefix="hems_tr_")
    try:
        over = synthetic_overrides(tmp)
        over.update(TRAIN_STEPS=3072, EVAL_EVERY=1536, OBS_NORM_STEPS=800, EMERGENCY_EPISODE_FRAC=0.5, HARD_EPISODE_FRAC=0.3,
                    SELECT_EPISODES_MAX=2, DEVICE="cpu")
        over["DDQN"] = {**over["DDQN"], "learning_starts": 300}
        over["SAC"] = {**over["SAC"], "learning_starts": 300}
        over["PPO"] = {**make_config().PPO, "n_steps": 64}
        cfg = make_config(**over)
        ff.make_synthetic_inputs(cfg, seed=4)
        d_tr = ff.load_or_build(cfg, "train", verbose=False)
        d_sel = ff.load_or_build(cfg, "select", verbose=False)
        runs = {}
        for algo in cfg.ALGOS:
            run, agent, done = train(cfg, algo, 42, debug=True, data_tr=d_tr, data_sel=d_sel, verbose=False)
            runs[algo] = run
            files = [f for f in ("settings.json", "obs_norm.json", "log_train.csv", "log_select.csv", "best.pt", "last.pt", "done.json")
                     if os.path.exists(os.path.join(run, f))]
            T.check(len(files) == 7, f"{algo}: run folder with settings, logs, best/last checkpoints", str(files))
            sel = pd.read_csv(os.path.join(run, "log_select.csv"))
            T.check(len(sel) == 2 and agent.total_steps >= cfg.TRAIN_STEPS, f"{algo}: selection evaluated every EVAL_EVERY steps", f"{len(sel)} evals")
            c = done["debug_counters"]
            T.check(c["emergency_steps"] > 0 and done["stored_all_in_control"] and done["stored"] == c["agent_steps"],
                    f"{algo}: emergencies happened and none reached a learning update", f"{c}, stored {done['stored']}")
        # resume: extend the SAC run as if Colab had disconnected before it finished
        cfg2 = cfg.replace(TRAIN_STEPS=4608)
        run = runs["sac"]
        os.remove(os.path.join(run, "done.json"))
        run2, agent2, done2 = train(cfg2, "sac", 42, data_tr=d_tr, data_sel=d_sel, verbose=False)
        sel = pd.read_csv(os.path.join(run2, "log_select.csv"))
        T.check(run2 == run and agent2.total_steps >= 4608 and len(sel) == 3, "an interrupted run resumes from last.pt", f"{agent2.total_steps}")
        ag = load_agent(os.path.join(run2, "best.pt"), cfg2)
        env = HEMSEnv(cfg2, d_sel, mode="eval", seed=1)
        o, _ = env.reset(seed=1)
        a = ag.act(o)
        T.check(0 <= a < 16, "best.pt loads and act(state) returns an action")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = new_run_dir(make_config(), "selftest", "hems_train", base=out_root) if out_root else None
    return T.summary(out)


if __name__ == "__main__":
    from hems_config import _default_selftest_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--algo", choices=["ppo", "sac", "ddqn", "a2c"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--forecast", default="ensemble", choices=["ensemble", "none", "perfect"])
    ap.add_argument("--steps", type=int, default=None, help="override TRAIN_STEPS")
    ap.add_argument("--all", action="store_true", help="train every algorithm with every seed (skips finished runs)")
    ap.add_argument("--fresh", action="store_true", help="start a new run even if an unfinished one exists")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--out", default=None)
    add_common_args(ap)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
    cfg = config_from_args(a)
    if a.steps:
        cfg = cfg.replace(TRAIN_STEPS=a.steps)
    ff.ensure_inputs(cfg, bool(a.synthetic))
    if a.plan:
        plan(cfg)
    elif a.all:
        for algo in cfg.ALGOS:
            for seed in cfg.TRAIN_SEEDS:
                if find_finished(cfg, algo, seed, a.forecast) and not a.fresh:
                    print(f"skip {algo} s{seed} {a.forecast}: finished")
                    continue
                train(cfg, algo, seed, a.forecast, fresh=a.fresh)
    elif a.algo:
        train(cfg, a.algo, a.seed if a.seed is not None else cfg.SEED, a.forecast, fresh=a.fresh)
    else:
        ap.print_help()
