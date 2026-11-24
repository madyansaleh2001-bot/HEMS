"""
hems_safety.py - rule-based safety controller (R1-R13), the only path from any agent to the actuators.

    agent.act(state) -> proposed action (16 = 2^4: contactor, pump, AC, generator request)
        -> SafetyController.decide(...)   checks R1-R13 with the predicted inverter load
        -> safe action executed on the ActuatorBus (rejections logged, penalized, flagged)

Structure that makes a bypass impossible:
  * ActuatorBus writes require a private token that only this module holds.
  * The bus has no inverter channel; any inverter command raises InverterWriteForbidden (R1).
  * A manual user command (app, Telegram, switch) is applied by the controller, always wins over the
    agent, is never blocked and never penalized.

Modes (emergency handling):
  "agent"     normal: the agent proposes, the controller filters.
  "emergency" communication loss, Pi restart: rule-based control on the Pi, agent in shadow mode
              (its decisions are logged, never executed). Back to "agent" after all readings are
              valid for RECOVERY_MIN.
  "safe"      inverter fault code (R11): actions frozen, pump and AC off, user alerted; back only after
              the fault is gone AND the user confirmed.
  One load unavailable (stale sensor R12, MCB trip R13): only that load goes to its fail-safe state,
  the agent keeps the others and sees the unavailable flag.

The rule-based controller used during emergencies and as the baseline is rule_based_action().

Run in Colab:
    !python hems_safety.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from hems_config import Config, SelfTest, make_config, new_run_dir

# -------------------------------------------------------------------------------------
# Actions: one discrete action out of 16 combinations of four switches
# -------------------------------------------------------------------------------------
N_ACTIONS = 16
SWITCHES = ("contactor", "pump", "ac", "gen")


def decode_action(a: int):
    a = int(a)
    if not 0 <= a < N_ACTIONS:
        raise ValueError(f"action {a} outside 0..15")
    return a & 1, (a >> 1) & 1, (a >> 2) & 1, (a >> 3) & 1


def encode_action(c: int, p: int, ac: int, g: int = 0) -> int:
    return int(c) | (int(p) << 1) | (int(ac) << 2) | (int(g) << 3)


ACTION_TABLE = [decode_action(a) for a in range(N_ACTIONS)]


# -------------------------------------------------------------------------------------
# Actuator bus (R1: no inverter channel; writes only with the controller's private token)
# -------------------------------------------------------------------------------------
class InverterWriteForbidden(PermissionError):
    """R1: no write or control command to the inverter, ever."""


_SAFETY_TOKEN = object()  # private: only SafetyController passes it


class ActuatorBus:
    CHANNELS = ("contactor", "pump", "ac", "fridge", "freezer", "heater")

    def __init__(self):
        self._state = {"contactor": 0, "pump": 0, "ac": 0, "fridge": 1, "freezer": 1, "heater": 0}
        self.write_count = 0

    def read(self) -> dict:
        return dict(self._state)

    def _write(self, command: dict, token):
        if token is not _SAFETY_TOKEN:
            raise PermissionError("actuators can only be driven through the SafetyController")
        for k in command:
            if "inverter" in str(k).lower():
                raise InverterWriteForbidden("R1: the inverter is read-only")
            if k not in self.CHANNELS:
                raise KeyError(f"unknown actuator channel {k!r}")
        for k, v in command.items():
            self._state[k] = int(bool(v))
        self.write_count += 1

    def send_inverter_command(self, *args, **kwargs):
        raise InverterWriteForbidden("R1: the inverter is read-only (RS232 values only)")


# -------------------------------------------------------------------------------------
# What the controller knows at a decision instant
# -------------------------------------------------------------------------------------
@dataclass
class SafetyContext:
    grid_present: bool = False
    soc: float = 50.0  # %
    tank: float = 50.0  # %
    contactor_on: int = 0  # executed states before this decision
    pump_on: int = 0
    ac_on: int = 0
    pump_run_s: float = 0.0  # continuous pump run time so far
    pump_rest_left_s: float = 0.0  # rest still due after a PUMP_MAX_H run
    ac_off_s: float = 1e9  # time since the AC compressor stopped
    s_since_pump_start: float = 1e9
    s_since_ac_start: float = 1e9
    heater_request: bool = False  # heater program window active
    r4_cut_active: bool = False  # R4 hysteresis state
    base_va: float = 333.0  # measured base load (VA), worst case if unknown
    freezer_user_on: bool = True
    pump_stale: bool = False  # R12 (pump ESP32 or tank sensor)
    ac_stale: bool = False  # R12 (AC ESP32 or indoor sensor)
    fridge_stale: bool = False
    pump_unavailable: bool = False  # R13
    ac_unavailable: bool = False
    blackout: bool = False  # inverter output off (trip, fault, low-voltage shutdown)
    user_fixed: dict = field(default_factory=dict)  # channel -> value forced by the user / heater program
    dt_s: float = 900.0


@dataclass
class Decision:
    executed: tuple  # (contactor, pump, ac, gen)
    proposed: tuple
    rejected: bool
    violations: list  # rules violated by the proposal
    mode: str
    relays: dict  # full actuator command written to the bus
    user_override: bool
    alerts: list


def fridge_va(cfg: Config) -> float:
    return cfg.FRIDGE_RUN_A[1] * cfg.GRID_V


def freezer_va(cfg: Config) -> float:
    return cfg.FREEZER_RUN_A * cfg.GRID_V


def pump_va(cfg: Config) -> float:
    return cfg.PUMP_RUN_A * cfg.GRID_V


def ac_va_pred(cfg: Config) -> float:
    return cfg.AC_RATED_A * cfg.GRID_V  # conservative (rated, not running current)


def grid_charge_ac_va(cfg: Config) -> float:
    return cfg.MAX_GRID_CHARGE_A * cfg.BATT_CHARGE_V / cfg.GRID_CHARGER_EFF


def r4_active(ctx: SafetyContext, off_grid: bool, cfg: Config) -> bool:
    if not off_grid:
        return False
    return ctx.soc < cfg.SOC_CUT or (ctx.r4_cut_active and ctx.soc < cfg.SOC_CUT + cfg.SOC_CUT_HYST)


def rule_violations(c: int, p: int, a: int, ctx: SafetyContext, cfg: Config) -> list:
    """Rules R3-R13 (+ blackout) violated if contactor=c, pump=p, AC=a were executed now."""
    v = []
    line = bool(ctx.grid_present and c)
    off = not line
    adds_p = bool(p and not ctx.pump_on)
    adds_a = bool(a and not ctx.ac_on)
    if ctx.blackout and (p or a):
        v.append("BLACKOUT")
    if off and ctx.soc < cfg.SOC_RESTRICT and (p or a):
        v.append("R3")
    cut = r4_active(ctx, off, cfg)
    if cut and a:
        v.append("R4")
    if off and p and a:
        v.append("R5")
    if adds_p or adds_a:
        fr = 0.0 if cut else fridge_va(cfg)
        fz = 0.0 if (cut or not ctx.freezer_user_on) else freezer_va(cfg)
        load = ctx.base_va + fr + fz + cfg.INVERTER_IDLE_W + p * pump_va(cfg) + a * ac_va_pred(cfg)
        if off:
            if load > cfg.INV_LIMIT_W:
                v.append("R6")
        else:
            heater = cfg.HEATER_W if ctx.heater_request else 0.0
            if load + heater + grid_charge_ac_va(cfg) > cfg.BYPASS_LIMIT_FRAC * cfg.BYPASS_MAX_A * cfg.GRID_V:
                v.append("R6")
    if adds_p and adds_a:
        v.append("R7")
    elif (adds_p and ctx.s_since_ac_start < cfg.START_STAGGER_S) or (adds_a and ctx.s_since_pump_start < cfg.START_STAGGER_S):
        v.append("R7")
    if adds_a and ctx.ac_off_s < cfg.AC_MIN_OFF_MIN * 60.0:
        v.append("R8")
    if ctx.heater_request and ctx.grid_present and c and (p or a):
        v.append("R9")
    if p and (ctx.tank >= cfg.TANK_FULL or ctx.pump_run_s + ctx.dt_s > cfg.PUMP_MAX_H * 3600.0 + 1e-6 or ctx.pump_rest_left_s > 0):
        v.append("R10")
    if (p and ctx.pump_stale) or (a and ctx.ac_stale):
        v.append("R12")
    if (p and ctx.pump_unavailable) or (a and ctx.ac_unavailable):
        v.append("R13")
    return v


def nearest_safe(prop: tuple, ctx: SafetyContext, cfg: Config):
    """Nearest safe (contactor, pump, ac): fewest changed switches, then fewest switched on,
    then keep contactor > pump (water) > AC (comfort). User-fixed channels are never changed."""
    pc, pp, pa = prop
    fixed = ctx.user_fixed
    best, best_key = None, None
    for c, p, a in itertools.product((0, 1), repeat=3):
        if "contactor" in fixed and c != fixed["contactor"]:
            continue
        if "pump" in fixed and p != fixed["pump"]:
            continue
        if "ac" in fixed and a != fixed["ac"]:
            continue
        nv = len(rule_violations(c, p, a, ctx, cfg))
        diff = (c != pc, p != pp, a != pa)
        n_on = (c and not pc) + (p and not pp) + (a and not pa)
        key = (nv, sum(diff), n_on, diff[0], diff[1], diff[2])
        if best_key is None or key < best_key:
            best, best_key = (c, p, a), key
    return best


def valid_action_mask(ctx: SafetyContext, cfg: Config) -> np.ndarray:
    """Actions the controller would execute unchanged (for the USE_ACTION_MASK ablation)."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    fixed = ctx.user_fixed
    for a_idx, (c, p, a, g) in enumerate(ACTION_TABLE):
        cc = fixed.get("contactor", c)
        pp = fixed.get("pump", p)
        aa = fixed.get("ac", a)
        v = [r for r in rule_violations(cc, pp, aa, ctx, cfg) if r != "BLACKOUT"]
        mask[a_idx] = not v
    if not mask.any():
        mask[0] = True
    return mask


# -------------------------------------------------------------------------------------
# Measurements used to decide the control mode (emergencies)
# -------------------------------------------------------------------------------------
@dataclass
class Measurements:
    time_s: float = 0.0
    comm_ok: bool = True  # MQTT to the ESP32s alive
    sensor_age_s: dict = field(default_factory=lambda: {"tank": 0.0, "indoor": 0.0, "pump_meter": 0.0, "ac_meter": 0.0, "fridge_meter": 0.0})
    inverter_fault_code: int = 0  # 0 = no fault, 1..61 = fault
    pi_restarted: bool = False
    user_confirmed: bool = False  # user confirmed after an inverter fault
    pump_meter_w: float = 0.0  # measured during the last step
    ac_meter_w: float = 0.0
    pump_cmd_last: int = 0  # commanded during the last step
    ac_cmd_last: int = 0
    power_available: bool = True  # inverter output was on during the last step
    user_reset: dict = field(default_factory=dict)  # load -> True when the user reset an MCB


class SafetyController:
    def __init__(self, cfg: Config, bus: ActuatorBus | None = None, start_mode: str = "emergency", keep_log: bool = False):
        self.cfg = cfg
        self.bus = bus if bus is not None else ActuatorBus()
        self.mode = start_mode  # after any restart the Pi starts in rule-based mode
        self.keep_log = keep_log
        self.log = []
        self.alerts = []
        self.r4_cut_active = False
        self.fault_latched = False
        self.valid_s = 0.0
        self._prev_valid = False
        self._last_t = None
        self.unavailable = {"pump": False, "ac": False}
        self.stale = {"pump": False, "ac": False, "fridge": False}
        self.counts = {"steps": 0, "rejected": 0, "user_override": 0, "agent_steps": 0}
        self.rule_counts = {}

    # -- alerts -----------------------------------------------------------------------
    def alert(self, t, code, msg):
        a = {"time_s": t, "code": code, "message": msg}
        self.alerts.append(a)
        return a

    # -- mode logic (emergency, safe mode, R11-R13) ---------------------------------
    def update_mode(self, m: Measurements) -> str:
        cfg = self.cfg
        t = m.time_s
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        stale_lim = cfg.STALE_S
        ages = m.sensor_age_s
        if not m.comm_ok:
            ages = {k: max(v, stale_lim + 1) for k, v in ages.items()}
        # R12: per-load fail-safe on stale data
        new_stale = {
            "pump": ages.get("tank", 0) > stale_lim or ages.get("pump_meter", 0) > stale_lim,
            "ac": ages.get("indoor", 0) > stale_lim or ages.get("ac_meter", 0) > stale_lim,
            "fridge": ages.get("fridge_meter", 0) > stale_lim,
        }
        for k, v in new_stale.items():
            if v and not self.stale[k]:
                self.alert(t, "R12", f"{k}: sensor/MQTT data older than {stale_lim:.0f} s -> fail-safe")
        self.stale = new_stale
        # R13: commanded on but the meter reads 0 -> MCB trip suspected
        if m.power_available and m.comm_ok:
            if m.pump_cmd_last and m.pump_meter_w <= 1.0 and not self.unavailable["pump"] and not new_stale["pump"]:
                self.unavailable["pump"] = True
                self.alert(t, "R13", "pump commanded on but its meter reads 0: MCB trip suspected, pump unavailable")
            if m.ac_cmd_last and m.ac_meter_w <= 1.0 and not self.unavailable["ac"] and not new_stale["ac"]:
                self.unavailable["ac"] = True
                self.alert(t, "R13", "AC commanded on but its meter reads 0: MCB trip suspected, AC unavailable")
        for k, v in m.user_reset.items():
            if v and self.unavailable.get(k):
                self.unavailable[k] = False
                self.alert(t, "R13", f"{k}: user reset, load available again")
        # Full emergencies
        valid_now = m.comm_ok and m.inverter_fault_code == 0 and not m.pi_restarted
        prev_valid, self._prev_valid = self._prev_valid, valid_now
        if not prev_valid:
            self.valid_s = 0.0
        if m.inverter_fault_code != 0:
            if self.mode != "safe":
                self.alert(t, "R11", f"inverter fault code {m.inverter_fault_code}: safe mode, pump and AC off")
            self.mode, self.fault_latched, self.valid_s = "safe", True, 0.0
            return self.mode
        if self.fault_latched:
            if not m.user_confirmed:
                return self.mode  # stays in safe mode until the user confirms
            self.fault_latched = False
            self.mode, self.valid_s = "emergency", 0.0
            self.alert(t, "R11", "user confirmed after the inverter fault: rule-based until readings are valid")
        if not m.comm_ok or m.pi_restarted:
            if self.mode == "agent":
                self.alert(t, "EMERGENCY", "communication loss" if not m.comm_ok else "Pi restart: rule-based mode")
            self.mode, self.valid_s = "emergency", 0.0
            return self.mode
        if self.mode in ("emergency", "safe"):
            # readings must have been valid for RECOVERY_MIN (counted from the first valid reading;
            # single stale loads stay load-level and do not block the return)
            if prev_valid and valid_now:
                self.valid_s += dt
            if self.valid_s >= self.cfg.RECOVERY_MIN * 60.0 - 1e-6:
                self.mode = "agent"
                self.alert(t, "RECOVERED", f"readings valid for {self.cfg.RECOVERY_MIN:.0f} min: control back to the agent")
        return self.mode

    # -- decision for one control instant --------------------------------------------
    def decide(self, proposal: tuple, ctx: SafetyContext, t: float = 0.0) -> Decision:
        cfg = self.cfg
        pc, pp, pa, pg = (int(x) for x in proposal)
        mode = self.mode
        ctx.r4_cut_active = self.r4_cut_active
        ctx.pump_stale = ctx.pump_stale or self.stale["pump"]
        ctx.ac_stale = ctx.ac_stale or self.stale["ac"]
        ctx.pump_unavailable = ctx.pump_unavailable or self.unavailable["pump"]
        ctx.ac_unavailable = ctx.ac_unavailable or self.unavailable["ac"]
        alerts = []
        if mode == "safe":
            # R11: frozen; pump and AC off; contactor kept as it is (user may still force it)
            c = ctx.user_fixed.get("contactor", ctx.contactor_on)
            ex = (int(c), 0, 0)
            viol = []
        else:
            viol = rule_violations(pc, pp, pa, ctx, cfg)
            fixed_ok = all(dict(contactor=pc, pump=pp, ac=pa)[k] == v for k, v in ctx.user_fixed.items() if k in ("contactor", "pump", "ac"))
            if not viol and fixed_ok:
                ex = (pc, pp, pa)
            else:
                ex = nearest_safe((pc, pp, pa), ctx, cfg)
                rest = rule_violations(*ex, ctx, cfg)
                if rest:
                    alerts.append(self.alert(t, "USER_OVERRIDE", f"user command kept although it violates {rest}"))
        gen = pg if mode == "agent" else 0
        executed = (int(ex[0]), int(ex[1]), int(ex[2]), int(gen))
        # rejection: the agent proposed something the rules do not allow (user overrides and blackout are not counted)
        counted = [r for r in viol if r != "BLACKOUT"]
        user_diff = any(k in ctx.user_fixed and dict(contactor=pc, pump=pp, ac=pa)[k] != ctx.user_fixed[k] for k in ("contactor", "pump", "ac"))
        own_diff = any(dict(contactor=pc, pump=pp, ac=pa)[k] != dict(contactor=ex[0], pump=ex[1], ac=ex[2])[k]
                       for k in ("contactor", "pump", "ac") if k not in ctx.user_fixed)
        rejected = bool(mode == "agent" and counted and own_diff)
        # R4 hysteresis and the fridge / freezer / heater relays
        off_exec = not (ctx.grid_present and executed[0])
        if off_exec and ctx.soc < cfg.SOC_CUT:
            if not self.r4_cut_active:
                alerts.append(self.alert(t, "R4", f"SOC {ctx.soc:.1f} % on battery: fridge, freezer and AC disconnected"))
            self.r4_cut_active = True
        elif self.r4_cut_active and (not off_exec or ctx.soc >= cfg.SOC_CUT + cfg.SOC_CUT_HYST):
            self.r4_cut_active = False
        fridge = 1 if (ctx.fridge_stale or self.stale["fridge"]) else int(not self.r4_cut_active)
        freezer = int(ctx.freezer_user_on and not self.r4_cut_active)
        heater = int(ctx.heater_request and ctx.grid_present and executed[0] and not executed[1] and not executed[2] and not ctx.blackout)
        if ctx.blackout:
            relays = {"contactor": executed[0], "pump": 0, "ac": 0, "fridge": fridge, "freezer": freezer, "heater": 0}
        else:
            relays = {"contactor": executed[0], "pump": executed[1], "ac": executed[2], "fridge": fridge, "freezer": freezer, "heater": heater}
        self.bus._write(relays, _SAFETY_TOKEN)
        # bookkeeping
        self.counts["steps"] += 1
        if mode == "agent":
            self.counts["agent_steps"] += 1
        if rejected:
            self.counts["rejected"] += 1
            for r in counted:
                self.rule_counts[r] = self.rule_counts.get(r, 0) + 1
        if user_diff:
            self.counts["user_override"] += 1
        d = Decision(executed=executed, proposed=(pc, pp, pa, pg), rejected=rejected, violations=viol, mode=mode,
                     relays=relays, user_override=bool(ctx.user_fixed), alerts=alerts)
        if self.keep_log:
            self.log.append({"time_s": t, "mode": mode, "proposed": d.proposed, "executed": executed, "rejected": rejected,
                             "violations": ",".join(viol), "user_fixed": dict(ctx.user_fixed), "soc": ctx.soc, "tank": ctx.tank,
                             "grid": int(ctx.grid_present)})
        return d

    def save_log(self, path: str):
        import pandas as pd
        pd.DataFrame(self.log).to_csv(path, index=False)
        pd.DataFrame(self.alerts).to_csv(path.replace(".csv", "_alerts.csv"), index=False)


# -------------------------------------------------------------------------------------
# Rule-based controller (baseline and emergency controller on the Pi)
# -------------------------------------------------------------------------------------
def context_from_obs(o: dict, cfg: Config) -> SafetyContext:
    """Approximate SafetyContext from the raw observation (used by the rule-based policy to stay safe)."""
    return SafetyContext(
        grid_present=o["grid_present"] > 0.5, soc=float(o["soc_pct"]), tank=float(o["tank_pct"]),
        contactor_on=int(o["contactor_closed"] > 0.5), pump_on=int(o["pump_on"] > 0.5), ac_on=int(o["ac_on"] > 0.5),
        pump_run_s=float(o["pump_run_h"]) * 3600.0, pump_rest_left_s=0.0, ac_off_s=1e9,
        heater_request=o["heater_on"] > 0.5, r4_cut_active=o["fridge_cut_active"] > 0.5,
        base_va=float(o["base_load_w"]) / cfg.BASE_PF, freezer_user_on=o["freezer_on"] > 0.5 or o["fridge_cut_active"] > 0.5,
        pump_unavailable=o["pump_unavailable"] > 0.5, ac_unavailable=o["ac_unavailable"] > 0.5,
        user_fixed={"contactor": 1} if o["user_override_contactor"] > 0.5 else {}, dt_s=cfg.STEP_MIN * 60.0)


def rule_based_proposal(o: dict, cfg: Config) -> tuple:
    """Simple rules: contactor closed whenever the grid is present and SOC < 50 % (hysteresis RB_CONTACTOR_HYST, and
    while the tank is below RB_TANK_GRID); pump on surplus PV or on the grid; AC when the humidex is above the threshold,
    someone is home and SOC allows (hysteresis RB_AC_HYST)."""
    grid = o["grid_present"] > 0.5
    soc = float(o["soc_pct"])
    tank = float(o["tank_pct"])
    soc_lim = cfg.RB_CONTACTOR_SOC + (cfg.RB_CONTACTOR_HYST if o["contactor_closed"] > 0.5 else 0.0)
    c = int(grid and (soc < soc_lim or tank < cfg.RB_TANK_GRID or o["user_override_contactor"] > 0.5))
    on_grid = grid and c
    pump_w = cfg.PUMP_RUN_A * cfg.GRID_V * cfg.PUMP_PF
    other = float(o["total_load_w"]) - o["pump_on"] * pump_w - o["ac_on"] * cfg.AC_INPUT_W
    hot = o["humidex_excess"] > 0 or (o["ac_on"] > 0.5 and o["humidex_in"] > o["humidex_thr"] - cfg.RB_AC_HYST)
    want_ac = hot and o["occ_effective_home"] > 0.5 and (on_grid or soc >= cfg.RB_AC_MIN_SOC)
    surplus = float(o["pv_now_w"]) * cfg.INV_EFF - other - (cfg.AC_INPUT_W if want_ac else 0.0)
    tank_ok = tank < cfg.RB_PUMP_START_TANK or (o["pump_on"] > 0.5 and tank < cfg.TANK_FULL)
    pv_ok = surplus >= cfg.RB_PV_SURPLUS_FRAC * pump_w or (soc >= cfg.RB_FULL_SOC and surplus >= cfg.RB_PV_SURPLUS_FRAC_FULL * pump_w)
    want_pump = tank_ok and (on_grid or pv_ok)
    return c, int(want_pump), int(want_ac)


def rule_based_action(o: dict, cfg: Config) -> int:
    c, p, a = rule_based_proposal(o, cfg)
    ctx = context_from_obs(o, cfg)
    if rule_violations(c, p, a, ctx, cfg):
        c, p, a = nearest_safe((c, p, a), ctx, cfg)
    return encode_action(c, p, a, 0)


# -------------------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------------------
def selftest(out_root: str | None = None) -> bool:
    cfg = make_config()
    T = SelfTest("hems_safety")
    rng = np.random.default_rng(cfg.SEED)

    # R1 and the bus token
    bus = ActuatorBus()
    try:
        bus.send_inverter_command("set_output_priority", "SBU")
        T.check(False, "R1: inverter command refused")
    except InverterWriteForbidden:
        T.check(True, "R1: inverter command refused")
    try:
        bus._write({"inverter_priority": 1}, _SAFETY_TOKEN)
        T.check(False, "R1: inverter channel refused even with the token")
    except InverterWriteForbidden:
        T.check(True, "R1: inverter channel refused even with the token")
    T.check("inverter" not in " ".join(ActuatorBus.CHANNELS), "bus has no inverter channel")
    for tok in (None, object(), "token"):
        try:
            bus._write({"pump": 1}, tok)
            T.check(False, f"direct bus write without the controller refused ({type(tok).__name__})")
        except PermissionError:
            T.check(True, f"direct bus write without the controller refused ({type(tok).__name__})")
    T.check(bus.read()["pump"] == 0 and bus.write_count == 0, "bus unchanged after refused writes")

    # action encoding
    T.check(all(encode_action(*decode_action(a)) == a for a in range(16)) and len(set(ACTION_TABLE)) == 16, "16 actions encode/decode")

    def ctx(**kw):
        base = dict(grid_present=False, soc=60.0, tank=50.0, base_va=150.0)
        base.update(kw)
        return SafetyContext(**base)

    def dec(prop, c_kw, mode="agent", sc=None):
        sc = sc or SafetyController(cfg, start_mode=mode)
        sc.mode = mode
        return sc.decide(prop, ctx(**c_kw)), sc

    # R3
    d, _ = dec((0, 1, 0, 0), dict(soc=34.0))
    T.check(d.executed[1] == 0 and d.rejected and "R3" in d.violations, "R3: no pump on battery below 35 %")
    d, _ = dec((0, 0, 1, 0), dict(soc=34.0))
    T.check(d.executed[2] == 0 and d.rejected, "R3: no AC on battery below 35 %")
    d, _ = dec((1, 1, 0, 0), dict(soc=34.0, grid_present=True))
    T.check(d.executed == (1, 1, 0, 0) and not d.rejected, "R3 does not apply on the grid")
    # R4 with hysteresis
    sc = SafetyController(cfg, start_mode="agent")
    d = sc.decide((0, 0, 0, 0), ctx(soc=29.0))
    T.check(d.relays["fridge"] == 0 and d.relays["freezer"] == 0 and sc.r4_cut_active, "R4: fridge and freezer cut below 30 % on battery")
    d = sc.decide((0, 0, 0, 0), ctx(soc=31.0))
    T.check(d.relays["fridge"] == 0, "R4 hysteresis: still cut at 31 %")
    d = sc.decide((0, 0, 1, 0), ctx(soc=32.5))
    T.check(d.executed[2] == 0 and "R4" in d.violations, "R4: AC disconnected while cut")
    d = sc.decide((0, 0, 0, 0), ctx(soc=33.0))
    T.check(d.relays["fridge"] == 1 and not sc.r4_cut_active, "R4: back on at SOC_CUT + 3")
    sc.decide((0, 0, 0, 0), ctx(soc=29.0))
    d = sc.decide((1, 0, 0, 0), ctx(soc=29.0, grid_present=True))
    T.check(d.relays["fridge"] == 1, "R4: back on when the grid is connected")
    # R5
    d, _ = dec((0, 1, 1, 0), dict(soc=80.0, pump_on=1, ac_on=1))
    T.check(d.executed[:3] == (0, 1, 0) and d.rejected and "R5" in d.violations, "R5: never pump and AC together on battery (keeps the pump)")
    d, _ = dec((1, 1, 1, 0), dict(soc=80.0, grid_present=True, pump_on=1, ac_on=1))
    T.check(d.executed[:3] == (1, 1, 1) and not d.rejected, "R5 does not apply on the grid")
    # R6
    d, _ = dec((0, 1, 0, 0), dict(soc=80.0, base_va=1500.0))
    T.check(d.executed[1] == 0 and "R6" in d.violations, "R6: reject the action that adds load above 3600 W")
    d, _ = dec((0, 1, 0, 0), dict(soc=80.0, base_va=1500.0, pump_on=1))
    T.check(d.executed[1] == 1 and "R6" not in d.violations, "R6 only rejects added load")
    # R7
    d, _ = dec((1, 1, 1, 0), dict(soc=80.0, grid_present=True))
    T.check(d.executed[1] + d.executed[2] == 1 and "R7" in d.violations, "R7: pump and AC never start in the same step")
    d, _ = dec((1, 1, 0, 0), dict(soc=80.0, grid_present=True, ac_on=1, s_since_ac_start=10.0))
    T.check(d.executed[1] == 0 and "R7" in d.violations, "R7: starts staggered by 30 s")
    # R8
    d, _ = dec((0, 0, 1, 0), dict(soc=80.0, ac_off_s=60.0))
    T.check(d.executed[2] == 0 and "R8" in d.violations, "R8: AC off at least 3 min before a restart")
    d, _ = dec((0, 0, 1, 0), dict(soc=80.0, ac_off_s=200.0))
    T.check(d.executed[2] == 1, "R8: restart allowed after 3 min")
    # R9
    d, _ = dec((1, 1, 0, 0), dict(soc=80.0, grid_present=True, heater_request=True))
    T.check(d.executed[1] == 0 and d.relays["heater"] == 1 and "R9" in d.violations, "R9: no pump while the water heater runs")
    d, _ = dec((1, 0, 0, 0), dict(soc=80.0, grid_present=False, heater_request=True))
    T.check(d.relays["heater"] == 0, "R9: heater off at once when the grid drops")
    # R10
    d, _ = dec((1, 1, 0, 0), dict(grid_present=True, tank=95.0))
    T.check(d.executed[1] == 0 and "R10" in d.violations, "R10: pump stops when the tank is full")
    d, _ = dec((1, 1, 0, 0), dict(grid_present=True, pump_on=1, pump_run_s=10 * 3600.0 - 600))
    T.check(d.executed[1] == 0 and "R10" in d.violations, "R10: pump stops after 10 h continuous")
    # R11 safe mode
    sc = SafetyController(cfg, start_mode="agent")
    sc.update_mode(Measurements(time_s=0, inverter_fault_code=7))
    d = sc.decide((1, 1, 1, 1), ctx(grid_present=True, pump_on=1, ac_on=1, contactor_on=1))
    T.check(sc.mode == "safe" and d.executed[1:] == (0, 0, 0) and not d.rejected and any(a["code"] == "R11" for a in sc.alerts),
            "R11: fault -> safe mode, pump and AC off, user alerted, no penalty")
    sc.update_mode(Measurements(time_s=900, inverter_fault_code=0, user_confirmed=False))
    T.check(sc.mode == "safe", "R11: stays safe until the user confirms")
    sc.update_mode(Measurements(time_s=1800, inverter_fault_code=0, user_confirmed=True))
    sc.update_mode(Measurements(time_s=2700, inverter_fault_code=0, user_confirmed=True))
    T.check(sc.mode == "agent", "R11: back to the agent after confirmation and RECOVERY_MIN of valid readings")
    # R12 stale data
    sc = SafetyController(cfg, start_mode="agent")
    ages = {"tank": 130.0, "indoor": 0.0, "pump_meter": 0.0, "ac_meter": 0.0, "fridge_meter": 0.0}
    sc.update_mode(Measurements(time_s=0, sensor_age_s=ages))
    d = sc.decide((1, 1, 1, 0), ctx(grid_present=True))
    T.check(d.executed[1] == 0 and sc.mode == "agent" and any(a["code"] == "R12" for a in sc.alerts), "R12: stale tank sensor -> pump fail-safe off, agent keeps the rest")
    ages = {"tank": 0.0, "indoor": 0.0, "pump_meter": 0.0, "ac_meter": 0.0, "fridge_meter": 500.0}
    sc.update_mode(Measurements(time_s=900, sensor_age_s=ages))
    d = sc.decide((0, 0, 0, 0), ctx(soc=25.0))
    T.check(d.relays["fridge"] == 1, "R12: stale fridge data -> fridge fail-safe on")
    sc = SafetyController(cfg, start_mode="agent")
    sc.update_mode(Measurements(time_s=0, comm_ok=False))
    d = sc.decide((1, 1, 1, 0), ctx(grid_present=True))
    T.check(sc.mode == "emergency" and d.executed[1:3] == (0, 0) and not d.rejected, "MQTT loss -> emergency, pump/AC fail-safe off, no penalty")
    sc.update_mode(Measurements(time_s=900))
    T.check(sc.mode == "emergency", "recovery needs readings valid for RECOVERY_MIN")
    sc.update_mode(Measurements(time_s=1800))
    T.check(sc.mode == "agent", "back to the agent after RECOVERY_MIN")
    # R13 MCB trip
    sc = SafetyController(cfg, start_mode="agent")
    sc.update_mode(Measurements(time_s=0, pump_cmd_last=1, pump_meter_w=0.0))
    d = sc.decide((1, 1, 0, 0), ctx(grid_present=True))
    T.check(sc.unavailable["pump"] and d.executed[1] == 0 and any(a["code"] == "R13" for a in sc.alerts), "R13: meter 0 while on -> MCB trip, pump unavailable, alert")
    sc.update_mode(Measurements(time_s=900, user_reset={"pump": True}))
    T.check(not sc.unavailable["pump"], "R13: available again after the user reset")
    # Pi restart starts rule-based
    T.check(SafetyController(cfg).mode == "emergency", "after any restart the Pi starts in rule-based mode")
    # Manual user command always wins, never penalized
    d, _ = dec((0, 0, 0, 0), dict(grid_present=True, user_fixed={"contactor": 1}))
    T.check(d.executed[0] == 1 and not d.rejected, "user closes the grid contactor: agent cannot open it, no penalty")
    d, _ = dec((0, 1, 0, 0), dict(soc=80.0, user_fixed={"ac": 1}, ac_on=1, pump_on=1))
    T.check(d.executed[2] == 1 and d.executed[1] == 0, "user AC command wins; agent's pump gives way (R5)")
    # Exhaustive property: every executed action passes the rules
    bad = 0
    n = 0
    for _ in range(4000):
        c_kw = dict(grid_present=bool(rng.random() < 0.5), soc=float(rng.uniform(0, 100)), tank=float(rng.uniform(0, 100)),
                    contactor_on=int(rng.random() < 0.5), pump_on=int(rng.random() < 0.5), ac_on=int(rng.random() < 0.5),
                    pump_run_s=float(rng.uniform(0, 11 * 3600)), ac_off_s=float(rng.uniform(0, 1000)),
                    heater_request=bool(rng.random() < 0.2), base_va=float(rng.uniform(30, 1200)),
                    pump_stale=bool(rng.random() < 0.05), ac_unavailable=bool(rng.random() < 0.05),
                    s_since_pump_start=float(rng.uniform(0, 100)), s_since_ac_start=float(rng.uniform(0, 100)))
        sc = SafetyController(cfg, start_mode="agent")
        sc.r4_cut_active = bool(rng.random() < 0.3)
        a = int(rng.integers(16))
        d = sc.decide(decode_action(a), ctx(**c_kw))
        cc = ctx(**c_kw)
        cc.r4_cut_active = d.relays["fridge"] == 0 and sc.r4_cut_active
        v = rule_violations(*d.executed[:3], ctx(**{**c_kw}), cfg)
        n += 1
        if v:
            bad += 1
        if d.relays["pump"] and d.relays["ac"] and not (c_kw["grid_present"] and d.executed[0]):
            bad += 1
    T.check(bad == 0, f"{n} random proposals: every executed action satisfies R3-R13", f"bad={bad}")
    # Action mask consistency: masked-valid actions are executed unchanged
    ok = True
    for _ in range(500):
        c_kw = dict(grid_present=bool(rng.random() < 0.5), soc=float(rng.uniform(20, 100)), tank=float(rng.uniform(0, 100)),
                    pump_on=int(rng.random() < 0.5), ac_on=int(rng.random() < 0.5), base_va=float(rng.uniform(30, 600)))
        mask = valid_action_mask(ctx(**c_kw), cfg)
        for a in np.flatnonzero(mask):
            sc = SafetyController(cfg, start_mode="agent")
            d = sc.decide(decode_action(a), ctx(**c_kw))
            ok &= (not d.rejected) and d.executed[:3] == decode_action(a)[:3]
    T.check(ok, "action mask: valid actions execute unchanged")
    out = new_run_dir(cfg, "selftest", "hems_safety", base=out_root) if out_root else None
    return T.summary(out)


if __name__ == "__main__":
    from hems_config import _default_selftest_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
