"""
hems_report.py - tables, plots and a run summary from the training logs and the evaluation folder.

Produces OUTPUT_DIR/report/report_<time>/ with
    report.md                  the results report (tables + acceptance checks + chosen agent)
    table_test_by_group.csv    4 algorithms (mean +/- std over seeds 42/43/44), rule-based, no-forecast, perfect-forecast
    table_test_by_agent.csv    every agent and seed (mean over the 20 grid scenarios)
    table_monthly.csv          monthly metrics per agent
    table_scenarios.csv        11-scenario pass matrix
    learning_curves.png, rejected_per_day.png, test_metrics.png, monthly.png, trajectory_<agent>.png
    summary.json               acceptance checks and the chosen agent

Run in Colab (after hems_evaluate.py):
    !python hems_report.py                 # uses the latest evaluation folder
    !python hems_report.py --eval <dir>
    !python hems_report.py --selftest
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hems_config import Config, SelfTest, add_common_args, config_from_args, make_config, new_run_dir, synthetic_overrides  # noqa: E402

# Fixed colour per entity (never by rank). Categorical slots of a CVD-validated palette; baseline in neutral grey.
COLORS = {"ppo": "#2a78d6", "sac": "#eb6834", "ddqn": "#1baf7a", "a2c": "#eda100", "none": "#e87ba4", "perfect": "#6250d6",
          "rule": "#8a8985"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TABLE_METRICS = [("trips", "Inverter trips", "{:.0f}"), ("export_kwh", "Exported kWh", "{:.2f}"), ("water_outage_min", "Water outage min", "{:.0f}"),
                 ("fridge_cut_min", "Fridge/freezer cut min", "{:.0f}"), ("discomfort_ch", "Discomfort C-h (home)", "{:.1f}"),
                 ("grid_kwh", "Grid kWh", "{:.0f}"), ("bill_usd", "Bill $", "{:.1f}"), ("bill_usd_agent", "Agent bill $", "{:.1f}"),
                 ("pv_used_pct", "PV used %", "{:.1f}"), ("rejected_per_day", "Rejected/day", "{:.2f}"), ("toggles_per_day", "Toggles/day", "{:.1f}"),
                 ("gen_right_pct", "Generator right %", "{:.0f}")]


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def group_key(algo, fm):
    if algo == "rule":
        return "rule"
    return algo if fm == "ensemble" else fm


def group_label(key, algo_for_fm=None):
    return {"rule": "rule-based", "ppo": "PPO", "sac": "SAC-Discrete", "ddqn": "DDQN", "a2c": "A2C",
            "none": "no-forecast", "perfect": "perfect-forecast"}.get(key, key)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def latest_eval(cfg: Config):
    ev = sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, "eval", "eval_*")))
    ev = [e for e in ev if os.path.exists(os.path.join(e, "summary.json"))]
    return ev[-1] if ev else None


def selftest_status(cfg: Config) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, "selftest", "*", "selftest_result.json"))):
        d = json.load(open(p))
        out[d["name"]] = {"ok": d["ok"], "passed": len(d["passed"]), "failed": d["failed"], "path": p}
    return out


def build_report(cfg: Config, eval_dir: str, out_dir: str | None = None) -> str:
    out = out_dir or new_run_dir(cfg, "report", "report")
    ov = pd.read_csv(os.path.join(eval_dir, "test_overall.csv"))
    mo = pd.read_csv(os.path.join(eval_dir, "test_monthly.csv"))
    sc = pd.read_csv(os.path.join(eval_dir, "scenarios.csv"))
    sel = pd.read_csv(os.path.join(eval_dir, "selection.csv")) if os.path.getsize(os.path.join(eval_dir, "selection.csv")) > 2 else pd.DataFrame()
    summ = json.load(open(os.path.join(eval_dir, "summary.json")))
    ov["algo"] = ov["algo"].fillna("rule")
    ov["group"] = [group_key(a, f) for a, f in zip(ov["algo"], ov["forecast_mode"])]
    mcols = [m for m, _, _ in TABLE_METRICS]
    # per agent (mean over grid seeds), then per group (mean +/- std over training seeds)
    per_agent = ov.groupby(["agent", "group", "algo", "seed", "forecast_mode"], dropna=False)[mcols].mean().reset_index()
    per_agent.to_csv(os.path.join(out, "table_test_by_agent.csv"), index=False)
    rows = []
    order = [g for g in ("rule", "ppo", "sac", "ddqn", "a2c", "none", "perfect") if g in set(per_agent["group"])]
    for g in order:
        sub = per_agent[per_agent["group"] == g]
        src = ov[ov["group"] == g] if g == "rule" else sub  # rule: spread over grid seeds; agents: over training seeds
        r = {"group": group_label(g), "n": len(sub) if g != "rule" else len(src)}
        for m, lab, fmt in TABLE_METRICS:
            mu, sd = src[m].mean(), src[m].std(ddof=0) if len(src) > 1 else 0.0
            r[lab] = (fmt.format(mu) + " +/- " + fmt.format(sd)) if np.isfinite(mu) else "n/a"
        rows.append(r)
    by_group = pd.DataFrame(rows)
    by_group.to_csv(os.path.join(out, "table_test_by_group.csv"), index=False)
    mo["algo"] = mo["algo"].fillna("rule")
    monthly = mo.groupby(["agent", "month"])[mcols].mean().reset_index()
    monthly.to_csv(os.path.join(out, "table_monthly.csv"), index=False)
    piv = sc.pivot_table(index="agent", columns="scenario", values="passed", aggfunc="first")
    piv.to_csv(os.path.join(out, "table_scenarios.csv"))
    chosen = summ.get("chosen_agent")

    # ---------------------------------------------------------------- plots
    runs = sel["run_dir"].tolist() if len(sel) else []
    logs = []
    for rd in runs:
        p = os.path.join(rd, "log_select.csv")
        t = os.path.join(rd, "log_train.csv")
        if os.path.exists(p):
            d = json.load(open(os.path.join(rd, "done.json")))
            ls = pd.read_csv(p)
            lt = pd.read_csv(t) if os.path.exists(t) else pd.DataFrame()
            logs.append((d["algo"], d["seed"], d["forecast_mode"], ls, lt))
    algos = [a for a in ("ppo", "sac", "ddqn", "a2c") if any(l[0] == a and l[2] == "ensemble" for l in logs)]
    if algos:
        fig, axes = plt.subplots(1, len(algos), figsize=(3.6 * len(algos), 3.0), sharey=True, squeeze=False)
        for ax, a in zip(axes[0], algos):
            _style(ax)
            for (al, sd, fm, ls, _), dash in zip([l for l in logs if l[0] == a and l[2] == "ensemble"], ("-", "--", ":")):
                ax.plot(ls["steps"] / 1e6, ls["select_reward"], dash, color=COLORS[a], lw=1.8, label=f"seed {sd}")
            ax.set_title(group_label(a), fontsize=10, color=INK)
            ax.set_xlabel("environment steps (millions)", fontsize=8, color=INK2)
            ax.legend(fontsize=7, frameon=False)
        axes[0][0].set_ylabel("mean selection-episode reward", fontsize=8, color=INK2)
        fig.suptitle("Checkpoint selection on spring 2025 (fold 5)", fontsize=11, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "learning_curves.png"), dpi=150)
        plt.close(fig)
        curves = {}
        for a in algos:
            frames = [l[4] for l in logs if l[0] == a and l[2] == "ensemble" and "train_rejected_per_day" in l[4].columns]
            if frames:
                curves[a] = pd.concat(frames, ignore_index=True).groupby("steps")["train_rejected_per_day"].mean()
        if curves:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            _style(ax)
            for a, g in curves.items():
                ax.plot(g.index / 1e6, g.values, color=COLORS[a], lw=2, label=group_label(a))
            ax.set_xlabel("environment steps (millions)", fontsize=8, color=INK2)
            ax.set_ylabel("rejected actions per day (training)", fontsize=8, color=INK2)
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("Rejected actions during training (mean over seeds)", fontsize=10, color=INK)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "rejected_per_day.png"), dpi=150)
            plt.close(fig)
    # test metrics: one measure per panel, bars per group with std error bars
    show = [("bill_usd_agent", "Agent bill ($)"), ("discomfort_ch", "Discomfort (C-h, home)"), ("water_outage_min", "Water outage (min)"),
            ("fridge_cut_min", "Fridge/freezer cut (min)"), ("grid_kwh", "Grid energy (kWh)"), ("pv_used_pct", "PV used (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (m, lab) in zip(axes.flat, show):
        _style(ax)
        mus, sds = [], []
        for g in order:
            src = ov[ov["group"] == g] if g == "rule" else per_agent[per_agent["group"] == g]
            mus.append(src[m].mean())
            sds.append(src[m].std(ddof=0) if len(src) > 1 else 0.0)
        x = np.arange(len(order))
        ax.bar(x, mus, yerr=sds, color=[COLORS[g] for g in order], width=0.7, edgecolor="#fcfcfb", linewidth=2,
               error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 3})
        ax.set_xticks(x)
        ax.set_xticklabels([group_label(g) for g in order], rotation=30, ha="right", fontsize=7, color=INK2)
        ax.set_title(lab, fontsize=9, color=INK)
    fig.suptitle(f"Test {cfg.SPLIT_TEST[0]} to {cfg.SPLIT_TEST[1]} (bars: mean; whiskers: std over training seeds, or grid seeds for rule-based)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "test_metrics.png"), dpi=150)
    plt.close(fig)
    # monthly: chosen vs rule-based
    focus = [a for a in (chosen, "rule_based") if a and a in set(monthly["agent"])]
    if focus:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, (m, lab) in zip(axes, (("bill_usd_agent", "Agent bill per month ($)"), ("discomfort_ch", "Discomfort per month (C-h)"))):
            _style(ax)
            for a in focus:
                d = monthly[monthly["agent"] == a]
                key = "rule" if a == "rule_based" else a.split("_")[0]
                ax.plot(d["month"], d[m], "-o", color=COLORS.get(key, INK), lw=2, ms=5, label=a)
            ax.set_title(lab, fontsize=9, color=INK)
            ax.tick_params(axis="x", rotation=30)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "monthly.png"), dpi=150)
        plt.close(fig)
    # trajectories
    for a in focus:
        p = os.path.join(eval_dir, "trajectories", f"{a}.csv")
        if not os.path.exists(p):
            continue
        L = pd.read_csv(p)
        t = np.arange(len(L)) * cfg.dt_h
        fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
        for ax in axes:
            _style(ax)
        axes[0].plot(t, L["soc"], color=COLORS["ppo"], lw=2, label="battery SOC %")
        axes[0].plot(t, L["tank"], color=COLORS["ddqn"], lw=2, label="tank %")
        axes[0].axhline(cfg.SOC_CUT, color=INK2, lw=0.8, ls=":")
        axes[0].set_ylabel("%", fontsize=8, color=INK2)
        axes[0].legend(fontsize=7, frameon=False, ncol=2)
        axes[1].plot(t, L["pv_w"], color=COLORS["a2c"], lw=1.8, label="PV available W")
        axes[1].plot(t, L["load_w"], color=COLORS["sac"], lw=1.8, label="load W")
        axes[1].plot(t, L["grid_w"], color=COLORS["perfect"], lw=1.8, label="grid import W")
        axes[1].set_ylabel("W", fontsize=8, color=INK2)
        axes[1].legend(fontsize=7, frameon=False, ncol=3)
        for i, (col, lab, key) in enumerate((("grid", "grid present", "rule"), ("contactor", "contactor closed", "perfect"),
                                             ("pump", "pump on", "ddqn"), ("ac", "AC on", "sac"))):
            axes[2].fill_between(t, i * 1.2, i * 1.2 + L[col].values, step="post", color=COLORS[key], alpha=0.85, lw=0)
        axes[2].set_yticks([0.5, 1.7, 2.9, 4.1])
        axes[2].set_yticklabels(["grid present", "contactor", "pump", "AC"], fontsize=7)
        axes[2].grid(False)
        axes[3].plot(t, L["humidex"], color=COLORS["sac"], lw=1.8, label="indoor humidex")
        axes[3].step(t, L["thr"], color=INK2, lw=1, where="post", label="threshold")
        axes[3].set_ylabel("humidex", fontsize=8, color=INK2)
        axes[3].legend(fontsize=7, frameon=False, ncol=2)
        axes[3].set_xlabel("hours", fontsize=8, color=INK2)
        fig.suptitle(f"{a}: 3 hot days of the test period", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"trajectory_{a}.png"), dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------------- acceptance and report
    st = selftest_status(cfg)
    rl = per_agent[per_agent["group"].isin(["ppo", "sac", "ddqn", "a2c"])]
    have_grid = {(a, int(s)) for a, s in zip(rl["algo"], rl["seed"])}
    need = {(a, s) for a in cfg.ALGOS for s in cfg.TRAIN_SEEDS}
    chosen_sc = sc[sc["agent"] == chosen] if chosen else pd.DataFrame()
    by_id = {}
    for _, r in chosen_sc.iterrows():
        k = str(r["scenario"]).rstrip("b")
        by_id[k] = by_id.get(k, True) and bool(r["passed"])
    acc = {
        "all_selftests_pass": bool(st) and all(v["ok"] for v in st.values()),
        "zero_trips_every_test_episode": bool((ov["trips"] == 0).all()),
        "zero_export_every_test_episode": bool((ov["export_kwh"] == 0).all()),
        "all_11_scenarios_pass_for_chosen_agent": bool(chosen) and len(by_id) == 11 and all(by_id.values()),
        "results_4_algorithms_x_3_seeds": need <= have_grid,
        "rule_based_present": "rule" in set(per_agent["group"]),
        "no_forecast_present": "none" in set(per_agent["group"]),
        "perfect_forecast_present": "perfect" in set(per_agent["group"]),
    }
    acc["accepted"] = all(acc.values())
    missing = sorted(need - have_grid)
    lines = [f"# Akkar House HEMS - DRL results report", "",
             f"Evaluation folder: `{eval_dir}`  ", f"Test period: {cfg.SPLIT_TEST[0]} to {cfg.SPLIT_TEST[1]}, "
             f"{len(cfg.TEST_SCENARIO_SEEDS)} fixed grid-scenario seeds per agent (continuous runs).  ",
             f"Chosen agent (best selection score on {cfg.SPLIT_SELECT[0]} to {cfg.SPLIT_SELECT[1]}, never the test): **{chosen}**", "",
             "## Acceptance checks", ""]
    for k, v in acc.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: {k.replace('_', ' ')}")
    if missing:
        lines.append(f"- missing trainings: {', '.join(f'{a} seed {s}' for a, s in missing)}")
    if st:
        lines.append("")
        lines.append("Self-tests found: " + ", ".join(f"{k} ({'ok' if v['ok'] else 'FAILED'}, {v['passed']} checks)" for k, v in st.items()))
    lines += ["", "## Test results by group", "", "Algorithms: mean +/- std over training seeds (each seed averaged over the grid scenarios). "
              "Rule-based: mean +/- std over grid scenarios.", "", md_table(by_group), "",
              "![test metrics](test_metrics.png)", "", "## Every agent and seed", "",
              md_table(per_agent[["agent"] + mcols].round(2)), "", "## Scenario suite (11 scenarios; 3b and 4b are the no-grid variants)", "",
              md_table(piv.reset_index()), ""]
    if len(sel):
        lines += ["## Checkpoint selection", "", md_table(sel[[c for c in ("agent", "steps", "select_reward", "select_bill_usd_agent", "select_discomfort_ch",
                                                                         "select_fridge_cut_min", "select_water_outage_min") if c in sel.columns]].round(2)), "",
                  "![learning curves](learning_curves.png)", ""]
        if os.path.exists(os.path.join(out, "rejected_per_day.png")):
            lines += ["![rejected actions](rejected_per_day.png)", ""]
    if focus:
        lines += ["## Monthly results (chosen agent vs rule-based)", "", "![monthly](monthly.png)", ""]
        lines += [f"![trajectory {a}](trajectory_{a}.png)" for a in focus]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    json.dump({"eval_dir": eval_dir, "chosen_agent": chosen, "acceptance": acc, "missing_trainings": missing, "selftests": st},
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    return out


def selftest(out_root: str | None = None) -> bool:
    T = SelfTest("hems_report")
    tmp = tempfile.mkdtemp(prefix="hems_rp_")
    try:
        import hems_forecast_feed as ff
        from hems_evaluate import evaluate_all
        from hems_train import train
        cfg = make_config(**{**synthetic_overrides(tmp), "DEVICE": "cpu", "TRAIN_STEPS": 1024, "EVAL_EVERY": 512, "OBS_NORM_STEPS": 500,
                             "SELECT_EPISODES_MAX": 1, "TEST_SCENARIO_SEEDS": [1000, 1001]})
        cfg = cfg.replace(PPO={**cfg.PPO, "n_steps": 64}, A2C={**cfg.A2C, "n_steps": 16})
        ff.make_synthetic_inputs(cfg, seed=7)
        for algo in ("ppo", "a2c"):
            train(cfg, algo, 42, verbose=False)
        train(cfg, "ppo", 42, forecast="none", verbose=False)
        ev = evaluate_all(cfg, verbose=False)
        out = build_report(cfg, ev)
        files = ["report.md", "table_test_by_group.csv", "table_test_by_agent.csv", "table_monthly.csv", "table_scenarios.csv",
                 "learning_curves.png", "test_metrics.png", "monthly.png", "summary.json"]
        have = [f for f in files if os.path.exists(os.path.join(out, f))]
        T.check(len(have) == len(files), "report folder with tables, plots and summary", str(sorted(set(files) - set(have))))
        s = json.load(open(os.path.join(out, "summary.json")))
        T.check(not s["acceptance"]["results_4_algorithms_x_3_seeds"] and ("sac", 42) in [tuple(x) for x in s["missing_trainings"]],
                "acceptance flags the missing trainings")
        T.check(s["acceptance"]["zero_trips_every_test_episode"] and s["acceptance"]["zero_export_every_test_episode"], "zero trips and export detected")
        g = pd.read_csv(os.path.join(out, "table_test_by_group.csv"))
        T.check(list(g["group"]) == ["rule-based", "PPO", "A2C", "no-forecast"], "groups: rule-based, algorithms, no-forecast", str(list(g["group"])))
        txt = open(os.path.join(out, "report.md")).read()
        T.check("Acceptance checks" in txt and "Scenario suite" in txt and s["chosen_agent"] in txt, "report.md has acceptance, scenarios and the chosen agent")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    outd = new_run_dir(make_config(), "selftest", "hems_report", base=out_root) if out_root else None
    return T.summary(outd)


if __name__ == "__main__":
    from hems_config import _default_selftest_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval", default=None, help="evaluation folder (default: latest)")
    ap.add_argument("--out", default=None)
    add_common_args(ap)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
    cfg = config_from_args(a)
    ev = a.eval or latest_eval(cfg)
    if not ev:
        sys.exit("no evaluation folder found: run hems_evaluate.py first")
    o = build_report(cfg, ev)
    print(json.dumps(json.load(open(os.path.join(o, "summary.json")))["acceptance"], indent=1))
    print("report saved in", o)
