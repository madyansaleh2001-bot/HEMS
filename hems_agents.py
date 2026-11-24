"""
hems_agents.py - PPO, SAC-Discrete, DDQN and A2C in one PyTorch framework, plus the rule-based baseline.

  * Same network for all: MLP 2 x 256, ReLU (actor/critic/Q networks each use it).
  * Same state normalization: fitted once on training-split rollouts (training statistics only), stored
    with the agent and frozen.
  * One clean interface for later deployment on the Pi:
        agent = load_agent("best.pt")
        action = agent.act(state)          # raw state vector (hems_env.FEATURE_NAMES order) -> 0..15
    The action must still go through hems_safety.SafetyController (the only path to the actuators).
  * Emergency steps never enter learning: a transition is stored only when info["agent_in_control"] is True
    (not in the replay buffer for DDQN/SAC-Discrete, not in the rollout for PPO/A2C). The step where an
    emergency starts is truncated, so it bootstraps from the next state's value.
  * Generator request (advisory, changes no power flow): with GEN_BANDIT_HEAD each algorithm learns the 8
    power-flow actions (contactor, pump, AC) and a generator head (same 2 x 256 MLP) learns the hindsight reward
    of a request by regression; the request is made when that expected reward is positive. The executed action is
    still one of the 16. With GEN_BANDIT_HEAD = False the algorithm learns all 16 (factorized output layer) and the
    hindsight reward is credited to the request step.

Run in Colab:
    !python hems_agents.py --selftest     # toy-task learning checks for the 4 algorithms, save/load, emergency filter
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hems_config import Config, SelfTest, make_config, new_run_dir, set_global_seed
from hems_env import FEATURE_NAMES, IDX, N_FEATURES, HEMSEnv, obs_to_dict
from hems_safety import N_ACTIONS, rule_based_action


def get_device(cfg: Config):
    if cfg.TORCH_THREADS:
        torch.set_num_threads(int(cfg.TORCH_THREADS))
    if cfg.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.DEVICE)


def mlp(n_in, n_out, hidden):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class Factored16(nn.Module):
    """Same 2 x 256 ReLU MLP; its last layer gives 8 values for (contactor, pump, AC) and 2 for the generator request,
    combined into the 16 actions as out[a] = base[a & 7] + gen[a >> 3]. Exact for Q-values and a product policy for
    logits, because the advisory generator request changes no power flow (only its own hindsight reward)."""

    def __init__(self, n_in, hidden):
        super().__init__()
        self.net = mlp(n_in, 10, hidden)

    def forward(self, x):
        y = self.net(x)
        return y[..., :8].repeat(*([1] * (y.dim() - 1)), 2) + y[..., 8:].repeat_interleave(8, dim=-1)


def action_net(n_in, cfg, n_out):
    if n_out == N_ACTIONS and cfg.FACTORED_GEN_HEAD:
        return Factored16(n_in, cfg.HIDDEN)
    return mlp(n_in, n_out, cfg.HIDDEN)


# =====================================================================================
# State normalization (training statistics only)
# =====================================================================================
class ObsNormalizer:
    def __init__(self, mean, std, clip=10.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.clip = float(clip)

    def __call__(self, x):
        return np.clip((np.asarray(x, dtype=np.float32) - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}

    @staticmethod
    def from_state(d):
        return ObsNormalizer(d["mean"], d["std"], d.get("clip", 10.0))

    @staticmethod
    def fit(samples, clip=10.0):
        x = np.asarray(samples, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return ObsNormalizer(mean, std, clip)

    @staticmethod
    def identity(n):
        return ObsNormalizer(np.zeros(n), np.ones(n))


def fit_normalizer(cfg: Config, train_data, forecast_mode: str, steps: int, seed: int) -> ObsNormalizer:
    """Roll out a mix of the rule-based controller and random actions on the TRAINING split only."""
    env = HEMSEnv(cfg, train_data, mode="train", forecast_mode=forecast_mode, seed=seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
    samples, use_rule = [], True
    for _ in range(int(steps)):
        samples.append(obs)
        a = rule_based_action(obs_to_dict(obs), cfg) if (use_rule and rng.random() < 0.8) else int(rng.integers(N_ACTIONS))
        obs, _, _, _, info = env.step(a)
        if info["episode_end"]:
            obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
            use_rule = rng.random() < 0.5
    return ObsNormalizer.fit(samples, cfg.OBS_CLIP)


# =====================================================================================
# Base class
# =====================================================================================
class BaseAgent:
    algo = "base"

    def __init__(self, cfg: Config, obs_dim: int = N_FEATURES, norm: ObsNormalizer | None = None, seed: int = 42, device=None):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_actions = N_ACTIONS
        self.norm = norm or ObsNormalizer.identity(obs_dim)
        self.seed = seed
        self.device = device or get_device(cfg)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.gamma = cfg.GAMMA
        self.rscale = cfg.REWARD_SCALE
        self.use_mask = bool(cfg.USE_ACTION_MASK)
        self.total_steps = 0  # environment steps seen in training
        self.updates = 0
        self.stats = {}
        self.debug = False
        self.debug_stored_in_control = []  # filled when debug=True (self-test)
        self._obs = None
        self._masks = None
        self.patch = bool(cfg.GEN_REWARD_AT_REQUEST)
        self.bandit = bool(cfg.GEN_BANDIT_HEAD)
        self.n_pol = 8 if self.bandit else N_ACTIONS  # actions learned by the RL algorithm
        if self.bandit:
            self.gen_net = mlp(obs_dim, 1, cfg.HIDDEN).to(self.device)
            self.gen_opt = torch.optim.Adam(self.gen_net.parameters(), lr=cfg.GEN_HEAD["lr"])
            cap = int(cfg.GEN_HEAD["buffer"])
            self.gen_o = np.zeros((cap, obs_dim), dtype=np.float32)
            self.gen_r = np.zeros(cap, dtype=np.float32)
            self.gen_n = 0
            self.gen_ptr = 0
            self._gen_pending = {}
            self._gen_since = 0.0

    # -- generator request (advisory) -------------------------------------------------
    def _gen_nets(self):
        return {"gen": self.gen_net} if self.bandit else {}

    def _gen_opts(self):
        return {"gen": self.gen_opt} if self.bandit else {}

    @torch.no_grad()
    def gen_value(self, obs_n):
        """Expected hindsight reward of a generator request (per 15-min step)."""
        return self.gen_net(self._t(obs_n)).squeeze(-1).cpu().numpy()

    def _gen_bits(self, obs_n, explore: bool):
        if not self.bandit:
            return None
        g = (self.gen_value(obs_n) > 0.0).astype(np.int64)
        if explore:
            g = np.where(self.rng.random(len(g)) < self.cfg.GEN_EXPLORE, 1, g)
        return g

    def _compose(self, pol, g, masks16=None):
        if not self.bandit:
            return pol
        if masks16 is not None:  # request only where the composite action is allowed
            g = g * np.asarray(masks16, dtype=bool)[np.arange(len(pol)), pol + 8]
        return pol + 8 * g

    def _pol_masks(self, masks):
        if masks is None:
            return None
        masks = np.asarray(masks, dtype=bool)
        return masks[..., :8] if self.bandit else masks  # validity never depends on the advisory bit

    def _gen_observe(self, i, info, obs_n_i):
        """Collect (state, hindsight reward) pairs of the requests for the generator head."""
        if info.get("gen_request", False):
            self._gen_pending[(i, info.get("ep_id", 0), info.get("k", 0))] = obs_n_i
        for ep, kq, rg in info.get("gen_judged", ()):
            o = self._gen_pending.pop((i, ep, kq), None)
            if o is not None:
                j = self.gen_ptr
                self.gen_o[j], self.gen_r[j] = o, rg
                self.gen_ptr = (j + 1) % len(self.gen_r)
                self.gen_n = min(self.gen_n + 1, len(self.gen_r))

    def _gen_train(self, env_steps):
        if not self.bandit:
            return
        self._gen_since += env_steps * self.cfg.GEN_HEAD["updates_per_1k_steps"] / 1000.0
        n_upd = int(self._gen_since)
        self._gen_since -= n_upd
        if self.gen_n < 32:
            return
        for _ in range(n_upd):
            idx = self.rng.integers(0, self.gen_n, size=min(self.cfg.GEN_HEAD["batch"], self.gen_n))
            loss = F.mse_loss(self.gen_net(self._t(self.gen_o[idx])).squeeze(-1), self._t(self.gen_r[idx]))
            self.gen_opt.zero_grad()
            loss.backward()
            self.gen_opt.step()
        if n_upd:
            self.stats["gen_loss"] = loss.item()

    # -- action selection shared by all algorithms ----------------------------------------
    @torch.no_grad()
    def act_batch(self, obs_raw, masks=None, deterministic=False):
        obs_n = self.norm(obs_raw)
        pol = self._select(obs_n, self._pol_masks(masks), deterministic)
        return self._compose(pol, self._gen_bits(obs_n, explore=False), masks)

    # -- the clean deployment interface -------------------------------------------------
    @torch.no_grad()
    def act(self, state, mask=None, deterministic: bool = True) -> int:
        return int(self.act_batch(np.asarray(state, dtype=np.float32)[None], None if mask is None else np.asarray(mask)[None], deterministic)[0])

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @staticmethod
    def _mask_logits(logits, masks):
        if masks is None:
            return logits
        m = torch.as_tensor(masks, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~m, -1e9)

    def _record(self, info):
        if self.debug:
            self.debug_stored_in_control.append(bool(info["agent_in_control"]))

    # -- persistence --------------------------------------------------------------------
    def networks(self) -> dict:
        raise NotImplementedError

    def optimizers(self) -> dict:
        return {}

    def save(self, path: str, extra: dict | None = None, full: bool = True):
        d = {"algo": self.algo, "obs_dim": self.obs_dim, "seed": self.seed, "norm": self.norm.state_dict(),
             "feature_names": FEATURE_NAMES, "cfg": self.cfg.to_dict(), "total_steps": self.total_steps, "updates": self.updates,
             "nets": {k: v.state_dict() for k, v in self.networks().items()}, "extra": extra or {}}
        if full:
            d["opts"] = {k: v.state_dict() for k, v in self.optimizers().items()}
            d["rng"] = self.rng.bit_generator.state
            d["algo_state"] = self.algo_state()
        torch.save(d, path)

    def algo_state(self) -> dict:
        return {}

    def load_algo_state(self, d: dict):
        pass

    def load_state(self, d: dict):
        for k, v in self.networks().items():
            v.load_state_dict(d["nets"][k])
        for k, v in self.optimizers().items():
            if "opts" in d and k in d["opts"]:
                v.load_state_dict(d["opts"][k])
        self.total_steps = d.get("total_steps", 0)
        self.updates = d.get("updates", 0)
        if "rng" in d:
            self.rng.bit_generator.state = d["rng"]
        if "algo_state" in d:
            self.load_algo_state(d["algo_state"])


# =====================================================================================
# PPO and A2C (on-policy, shared rollout code)
# =====================================================================================
class _OnPolicy(BaseAgent):
    def __init__(self, cfg, hp: dict, **kw):
        super().__init__(cfg, **kw)
        self.hp = dict(hp)
        self.actor = action_net(self.obs_dim, cfg, self.n_pol).to(self.device)
        self.critic = mlp(self.obs_dim, 1, cfg.HIDDEN).to(self.device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.hp["lr"], eps=1e-5)

    def networks(self):
        return {"actor": self.actor, "critic": self.critic, **self._gen_nets()}

    def optimizers(self):
        return {"opt": self.opt, **self._gen_opts()}

    @torch.no_grad()
    def _select(self, obs_n, masks, deterministic):
        logits = self._mask_logits(self.actor(self._t(obs_n)), masks)
        if deterministic:
            return logits.argmax(-1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

    @torch.no_grad()
    def _value(self, obs_n):
        return self.critic(self._t(obs_n)).squeeze(-1).cpu().numpy()

    def train_steps(self, venv, n_steps: int):
        """Collect rollouts and update until n_steps more environment steps were taken."""
        if self._obs is None:
            self._obs, _ = venv.reset()
        done_steps = 0
        while done_steps < n_steps:
            done_steps += self._rollout_and_update(venv)
        return done_steps

    def _rollout_and_update(self, venv):
        """Collect n_steps per env, then update on every transition whose reward is final. A transition that made a
        generator request waits (with everything after it, to keep sequences contiguous) until the environment has
        judged the request in hindsight and its reward has been credited to it."""
        hp, ne = self.hp, venv.n
        if getattr(self, "_seq", None) is None or len(self._seq) != ne:
            self._seq = [[] for _ in range(ne)]
        patch = bool(self.cfg.GEN_REWARD_AT_REQUEST)
        ent_acc, nst = 0.0, 0
        for _ in range(hp["n_steps"]):
            obs_n = self.norm(self._obs)
            masks16 = venv.action_masks() if self.use_mask else None
            masks = self._pol_masks(masks16)
            with torch.no_grad():
                logits = self._mask_logits(self.actor(self._t(obs_n)), masks)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a).cpu().numpy()
                ent_acc += float(dist.entropy().mean())
            a = a.cpu().numpy()
            v = self._value(obs_n)
            nxt, r, term, trunc, infos = venv.step(self._compose(a, self._gen_bits(obs_n, explore=True), masks16))
            nst += ne
            boot_obs, boot_tr = [], []
            for i, info in enumerate(infos):
                if info["agent_in_control"]:  # emergency steps are never part of the rollout
                    tr = {"o": obs_n[i], "a": a[i], "lp": lp[i], "v": float(v[i]), "r": float(r[i]) * self.rscale,
                          "d": bool(term[i] or trunc[i]), "m": masks[i] if masks is not None else None,
                          "key": (info.get("ep_id", 0), info.get("k", 0)),
                          "ready": self.bandit or not (patch and info.get("gen_request", False))}
                    self._seq[i].append(tr)
                    if trunc[i] and not term[i]:
                        boot_obs.append(info["final_obs"] if info["episode_end"] else nxt[i])
                        boot_tr.append(tr)
                    self._record(info)
                if self.bandit:
                    self._gen_observe(i, info, obs_n[i])
                    continue
                for ep, kq, rg in info.get("gen_judged", ()):  # hindsight judgement -> reward of the request step
                    for tr in reversed(self._seq[i]):
                        if tr["key"] == (ep, kq):
                            tr["r"] += float(rg) * self.rscale
                            tr["ready"] = True
                            break
            if boot_obs:
                bv = self._value(self.norm(np.stack(boot_obs)))
                for tr, val in zip(boot_tr, bv):
                    tr["r"] += self.gamma * float(val)  # truncation: bootstrap instead of "end of the world"
            self._obs = nxt
        self.total_steps += nst
        self._gen_train(nst)
        last_v = self._value(self.norm(self._obs))
        O, A, LP, ADV, RET, M = [], [], [], [], [], []
        lam = hp["gae_lambda"]
        for i in range(ne):
            seq = self._seq[i]
            f = next((j for j, tr in enumerate(seq) if not tr["ready"]), len(seq))
            use, self._seq[i] = seq[:f], seq[f:]
            if not use:
                continue
            if use[-1]["d"]:
                nv = 0.0
            elif self._seq[i]:
                nv = self._seq[i][0]["v"]
            else:
                nv = float(last_v[i])
            adv = np.zeros(len(use), dtype=np.float32)
            gae = 0.0
            for t in range(len(use) - 1, -1, -1):
                tr = use[t]
                nd = 0.0 if tr["d"] else 1.0
                delta = tr["r"] + self.gamma * nv * nd - tr["v"]
                gae = delta + self.gamma * lam * nd * gae
                adv[t] = gae
                nv = tr["v"]
            vals = np.array([tr["v"] for tr in use], dtype=np.float32)
            O += [tr["o"] for tr in use]; A += [tr["a"] for tr in use]; LP += [tr["lp"] for tr in use]; M += [tr["m"] for tr in use]
            ADV.append(adv); RET.append(adv + vals)
        if not O:
            return nst
        O = self._t(np.stack(O)); A = torch.as_tensor(np.array(A), device=self.device, dtype=torch.long)
        LP = self._t(np.array(LP)); ADV = self._t(np.concatenate(ADV)); RET = self._t(np.concatenate(RET))
        Mt = None if (not self.use_mask or M[0] is None) else np.stack(M)
        self._update(O, A, LP, ADV, RET, Mt)
        self.stats["entropy"] = ent_acc / hp["n_steps"]
        return nst


class PPOAgent(_OnPolicy):
    algo = "ppo"

    def __init__(self, cfg, **kw):
        super().__init__(cfg, cfg.PPO, **kw)

    def _update(self, O, A, LP, ADV, RET, M):
        hp = self.hp
        n = len(A)
        for _ in range(hp["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, hp["minibatch"]):
                idx = perm[s: s + hp["minibatch"]]
                logits = self._mask_logits(self.actor(O[idx]), None if M is None else M[idx.cpu().numpy()])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[idx])
                adv = ADV[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(idx) > 1 else adv
                ratio = torch.exp(lp - LP[idx])
                pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - hp["clip"], 1 + hp["clip"]) * adv).mean()
                vl = F.mse_loss(self.critic(O[idx]).squeeze(-1), RET[idx])
                loss = pl + hp["vf_coef"] * vl - hp["ent_coef"] * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), hp["max_grad_norm"])
                self.opt.step()
                self.updates += 1
        self.stats.update(policy_loss=pl.item(), value_loss=vl.item())


class A2CAgent(_OnPolicy):
    algo = "a2c"

    def __init__(self, cfg, **kw):
        super().__init__(cfg, cfg.A2C, **kw)

    def _update(self, O, A, LP, ADV, RET, M):
        hp = self.hp
        logits = self._mask_logits(self.actor(O), M)
        dist = torch.distributions.Categorical(logits=logits)
        pl = -(dist.log_prob(A) * ADV).mean()
        vl = F.mse_loss(self.critic(O).squeeze(-1), RET)
        loss = pl + hp["vf_coef"] * vl - hp["ent_coef"] * dist.entropy().mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), hp["max_grad_norm"])
        self.opt.step()
        self.updates += 1
        self.stats.update(policy_loss=pl.item(), value_loss=vl.item())


# =====================================================================================
# Replay buffer and off-policy agents (DDQN, SAC-Discrete)
# =====================================================================================
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, n_actions, with_mask=False):
        self.cap = int(capacity)
        self.o = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.o2 = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.a = np.zeros(self.cap, dtype=np.int64)
        self.r = np.zeros(self.cap, dtype=np.float32)
        self.d = np.zeros(self.cap, dtype=np.float32)
        self.m2 = np.ones((self.cap, n_actions), dtype=bool) if with_mask else None
        self.n = 0
        self.ptr = 0
        self.added = 0

    def add(self, o, a, r, o2, d, m2=None):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        if self.m2 is not None and m2 is not None:
            self.m2[i] = m2
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        self.added += 1

    def sample(self, rng, batch):
        idx = rng.integers(0, self.n, size=batch)
        return idx, self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx], (None if self.m2 is None else self.m2[idx])

    def state(self):
        return {k: getattr(self, k) for k in ("o", "o2", "a", "r", "d", "m2", "n", "ptr", "added")}

    def load(self, s):
        for k, v in s.items():
            setattr(self, k, v)


class _OffPolicy(BaseAgent):
    def __init__(self, cfg, hp, **kw):
        super().__init__(cfg, **kw)
        self.hp = dict(hp)
        self.buffer = ReplayBuffer(self.hp["buffer"], self.obs_dim, self.n_pol, self.use_mask)
        self._pending = {}  # transitions with a generator request waiting for the hindsight judgement

    def algo_state(self):
        return {"buffer": self.buffer.state()} if self.cfg.SAVE_REPLAY_BUFFER else {}

    def load_algo_state(self, d):
        if "buffer" in d:
            self.buffer.load(d["buffer"])

    def train_steps(self, venv, n_steps: int):
        if self._obs is None:
            self._obs, _ = venv.reset()
            self._masks16 = venv.action_masks() if self.use_mask else None
            self._masks = self._pol_masks(self._masks16)
        done = 0
        vstep = 0
        while done < n_steps:
            obs_n = self.norm(self._obs)
            a = self._explore(obs_n, self._masks)
            nxt, r, term, trunc, infos = venv.step(self._compose(a, self._gen_bits(obs_n, explore=True), getattr(self, "_masks16", None)))
            masks16_2 = venv.action_masks() if self.use_mask else None
            masks2 = self._pol_masks(masks16_2)
            for i, info in enumerate(infos):
                if info["agent_in_control"]:  # emergency steps are never stored in the replay buffer
                    o2 = info["final_obs"] if info["episode_end"] else nxt[i]
                    if masks2 is None:
                        m2 = None
                    else:
                        m2 = self._pol_masks(info["action_mask"]) if (info["episode_end"] and "action_mask" in info) else masks2[i]
                    tr = [obs_n[i], a[i], float(r[i]) * self.rscale, self.norm(o2), float(term[i]), m2]
                    if self.patch and not self.bandit and info.get("gen_request", False):
                        self._pending[(i, info.get("ep_id", 0), info.get("k", 0))] = tr  # waits for its hindsight judgement
                    else:
                        self.buffer.add(*tr)
                    self._record(info)
                if self.bandit:
                    self._gen_observe(i, info, obs_n[i])
                    continue
                for ep, kq, rg in info.get("gen_judged", ()):
                    tr = self._pending.pop((i, ep, kq), None)
                    if tr is not None:
                        tr[2] += float(rg) * self.rscale
                        self.buffer.add(*tr)
            self._obs, self._masks, self._masks16 = nxt, masks2, masks16_2
            done += venv.n
            self.total_steps += venv.n
            self._gen_train(venv.n)
            vstep += 1
            if self.buffer.n >= self.hp["learning_starts"] and vstep % self.hp["train_every"] == 0:
                for _ in range(self.hp["grad_steps"]):
                    self._update()
                    self.updates += 1
        return done


class DDQNAgent(_OffPolicy):
    algo = "ddqn"

    def __init__(self, cfg, **kw):
        super().__init__(cfg, cfg.DDQN, **kw)
        self.q = action_net(self.obs_dim, cfg, self.n_pol).to(self.device)
        self.q_t = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.hp["lr"])
        self.eps_total = cfg.TRAIN_STEPS

    def networks(self):
        return {"q": self.q, "q_t": self.q_t, **self._gen_nets()}

    def optimizers(self):
        return {"opt": self.opt, **self._gen_opts()}

    def epsilon(self):
        frac = min(1.0, self.total_steps / max(1.0, self.hp["eps_frac"] * self.eps_total))
        return self.hp["eps_start"] + frac * (self.hp["eps_end"] - self.hp["eps_start"])

    @torch.no_grad()
    def _select(self, obs_n, masks, deterministic):
        q = self._mask_logits(self.q(self._t(obs_n)), masks)
        return q.argmax(-1).cpu().numpy()

    @torch.no_grad()
    def _explore(self, obs_n, masks):
        q = self._mask_logits(self.q(self._t(obs_n)), masks)
        a = q.argmax(-1).cpu().numpy()
        eps = self.epsilon()
        for i in range(len(a)):
            if self.rng.random() < eps:
                valid = np.flatnonzero(masks[i]) if masks is not None else np.arange(self.n_pol)
                a[i] = int(self.rng.choice(valid))
        return a

    def _update(self):
        _, o, a, r, o2, d, m2 = self.buffer.sample(self.rng, self.hp["batch"])
        o, o2, r, d = self._t(o), self._t(o2), self._t(r), self._t(d)
        a = torch.as_tensor(a, device=self.device)
        with torch.no_grad():
            a2 = self._mask_logits(self.q(o2), m2).argmax(-1, keepdim=True)  # double DQN: online selects
            y = r + self.gamma * (1 - d) * self.q_t(o2).gather(1, a2).squeeze(1)  # target evaluates
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.hp["max_grad_norm"])
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_t.parameters()):
                pt.mul_(1 - self.hp["target_tau"]).add_(self.hp["target_tau"] * p)
        self.stats.update(q_loss=loss.item(), epsilon=self.epsilon())


class SACDAgent(_OffPolicy):
    algo = "sac"

    def __init__(self, cfg, **kw):
        super().__init__(cfg, cfg.SAC, **kw)
        H = cfg.HIDDEN
        self.actor = action_net(self.obs_dim, cfg, self.n_pol).to(self.device)
        self.q1 = action_net(self.obs_dim, cfg, self.n_pol).to(self.device)
        self.q2 = action_net(self.obs_dim, cfg, self.n_pol).to(self.device)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = self.hp["target_entropy_ratio"] * np.log(self.n_pol)
        lr = self.hp["lr"]
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_a = torch.optim.Adam([self.log_alpha], lr=lr)

    def networks(self):
        return {"actor": self.actor, "q1": self.q1, "q2": self.q2, "q1_t": self.q1_t, "q2_t": self.q2_t, **self._gen_nets()}

    def optimizers(self):
        return {"pi": self.opt_pi, "q": self.opt_q, "alpha": self.opt_a, **self._gen_opts()}

    def algo_state(self):
        d = super().algo_state()
        d["log_alpha"] = float(self.log_alpha.item())
        return d

    def load_algo_state(self, d):
        super().load_algo_state(d)
        if "log_alpha" in d:
            with torch.no_grad():
                self.log_alpha.fill_(d["log_alpha"])

    @torch.no_grad()
    def _select(self, obs_n, masks, deterministic):
        logits = self._mask_logits(self.actor(self._t(obs_n)), masks)
        if deterministic:
            return logits.argmax(-1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

    @torch.no_grad()
    def _explore(self, obs_n, masks):
        logits = self._mask_logits(self.actor(self._t(obs_n)), masks)
        return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

    def _update(self):
        _, o, a, r, o2, d, m2 = self.buffer.sample(self.rng, self.hp["batch"])
        o, o2, r, d = self._t(o), self._t(o2), self._t(r), self._t(d)
        a = torch.as_tensor(a, device=self.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            lg2 = self._mask_logits(self.actor(o2), m2)
            p2 = F.softmax(lg2, -1)
            lp2 = F.log_softmax(lg2, -1)
            qt = torch.min(self.q1_t(o2), self.q2_t(o2))
            v2 = (p2 * (qt - alpha * lp2)).sum(-1)
            y = r + self.gamma * (1 - d) * v2
        q1 = self.q1(o).gather(1, a[:, None]).squeeze(1)
        q2 = self.q2(o).gather(1, a[:, None]).squeeze(1)
        lq = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad()
        lq.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.hp["max_grad_norm"])
        self.opt_q.step()
        lg = self.actor(o)
        p = F.softmax(lg, -1)
        lp = F.log_softmax(lg, -1)
        with torch.no_grad():
            qmin = torch.min(self.q1(o), self.q2(o))
        lpi = (p * (alpha * lp - qmin)).sum(-1).mean()
        self.opt_pi.zero_grad()
        lpi.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp["max_grad_norm"])
        self.opt_pi.step()
        ent = -(p.detach() * lp.detach()).sum(-1).mean()
        la = self.log_alpha * (ent - self.target_entropy)
        self.opt_a.zero_grad()
        la.backward()
        self.opt_a.step()
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p_, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.hp["tau"]).add_(self.hp["tau"] * p_)
        self.stats.update(q_loss=lq.item(), pi_loss=lpi.item(), alpha=alpha.item(), entropy=ent.item())


# =====================================================================================
# Rule-based baseline (same act(state) interface)
# =====================================================================================
class RuleBasedAgent:
    algo = "rule"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def act(self, state, mask=None, deterministic=True) -> int:
        return rule_based_action(obs_to_dict(state), self.cfg)

    def act_batch(self, obs_raw, masks=None, deterministic=True):
        return np.array([self.act(o) for o in obs_raw])


AGENTS = {"ppo": PPOAgent, "a2c": A2CAgent, "ddqn": DDQNAgent, "sac": SACDAgent}


def make_agent(algo: str, cfg: Config, norm=None, seed=42, obs_dim=N_FEATURES, device=None):
    if algo == "rule":
        return RuleBasedAgent(cfg)
    return AGENTS[algo](cfg, obs_dim=obs_dim, norm=norm, seed=seed, device=device)


def load_agent(path: str, cfg: Config | None = None, device=None, full: bool = False):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if cfg is None:
        cfg = make_config().replace(**{k: v for k, v in d["cfg"].items() if k in make_config().to_dict()})
    if d.get("feature_names") and d["feature_names"] != FEATURE_NAMES:
        raise ValueError("checkpoint was trained with a different state layout")
    ag = make_agent(d["algo"], cfg, norm=ObsNormalizer.from_state(d["norm"]), seed=d.get("seed", 42), obs_dim=d["obs_dim"], device=device)
    if full:
        ag.load_state(d)
    else:
        for k, v in ag.networks().items():
            v.load_state_dict(d["nets"][k])
        ag.total_steps = d.get("total_steps", 0)
    ag.checkpoint_extra = d.get("extra", {})
    return ag


# =====================================================================================
# Self-test on a toy task (known optimal action), plus the emergency filter
# =====================================================================================
class ToyEnv:
    """Contextual task: obs shows a target in 0..3; reward 1 for (action & 7) == target*2+1. Actions >= 8 carry a
    'generator request' judged two steps later: right (+1) when the target was 0, wrong (-1) otherwise. Optimal
    action: target*2+1, plus 8 when the target is 0.
    10% of steps are 'emergency' steps (agent_in_control False, reward 0) that must never reach a learning update."""

    def __init__(self, seed, obs_dim=12):
        self.rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.k = 0

    def _o(self):
        o = self.rng.normal(0, 0.1, self.obs_dim).astype(np.float32)
        self.tgt = int(self.rng.integers(4))
        o[self.tgt] += 1.0
        return o

    def reset(self, seed=None):
        self.k = 0
        self.emerg = False
        return self._o(), {}

    def step(self, a):
        in_ctrl = not self.emerg
        r = (1.0 if (a & 7) == self.tgt * 2 + 1 else 0.0) if in_ctrl else 0.0
        k = self.k
        self.k += 1
        end = self.k >= 10
        nxt_emerg = (not end) and self.rng.random() < 0.1
        trunc = end or (in_ctrl and nxt_emerg)
        self.emerg = nxt_emerg
        info = {"agent_in_control": in_ctrl, "episode_end": end, "ep_id": getattr(self, "ep", 0), "k": k,
                "gen_request": bool(in_ctrl and a >= 8), "gen_judged": []}
        self.pend = getattr(self, "pend", [])
        if info["gen_request"]:
            self.pend.append((info["ep_id"], k, self.tgt))
        due = [p for p in self.pend if p[1] <= k - 2 or end]
        self.pend = [p for p in self.pend if p not in due]
        info["gen_judged"] = [(e, kk, 1.0 if t == 0 else -1.0) for e, kk, t in due]
        if end:
            self.ep = getattr(self, "ep", 0) + 1
        return self._o(), r, False, trunc, info

    def action_mask(self):
        return np.ones(N_ACTIONS, dtype=bool)


def selftest(out_root: str | None = None) -> bool:
    from hems_env import VecEnv
    T = SelfTest("hems_agents")
    cfg = make_config(REWARD_SCALE=1.0, GAMMA=0.5, TRAIN_STEPS=20000,
                      PPO={**make_config().PPO, "n_envs": 4, "n_steps": 64, "minibatch": 64, "lr": 1e-3},
                      A2C={**make_config().A2C, "n_envs": 4, "n_steps": 8, "lr": 1e-3},
                      DDQN={**make_config().DDQN, "n_envs": 4, "learning_starts": 500, "buffer": 20000, "eps_frac": 0.3, "lr": 5e-4},
                      SAC={**make_config().SAC, "n_envs": 4, "learning_starts": 500, "buffer": 20000, "lr": 1e-3},
                      DEVICE="cpu")
    set_global_seed(cfg.SEED)
    d = 12
    runs = [(a, s_, True) for a, s_ in (("ppo", 16000), ("a2c", 20000), ("ddqn", 8000), ("sac", 8000))] + [("ddqn", 8000, False), ("ppo", 16000, False)]
    for algo, steps, bandit in runs:
        c_run = cfg.replace(GEN_BANDIT_HEAD=bandit)
        ag = make_agent(algo, c_run, seed=cfg.SEED, obs_dim=d)
        algo = algo if bandit else algo + " (16 actions, no generator head)"
        nets = ag.networks()
        first = next(iter(nets.values()))
        seq = first.net if isinstance(first, Factored16) else first
        lin = [m for m in seq if isinstance(m, nn.Linear)]
        T.check([l.out_features for l in lin[:-1]] == [256, 256] and all(isinstance(m, (nn.Linear, nn.ReLU)) for m in seq),
                f"{algo}: MLP 2 x 256 ReLU")
        ag.debug = True
        venv = VecEnv([ToyEnv(100 + i, d) for i in range(4)], seed=1)
        env_counter = {"in_ctrl": 0}
        orig_step = venv.step

        def counting_step(actions, _orig=orig_step):
            out = _orig(actions)
            env_counter["in_ctrl"] += sum(int(i["agent_in_control"]) for i in out[4])
            return out
        venv.step = counting_step
        ag.train_steps(venv, steps)
        probe = ToyEnv(999, d)
        hits = 0
        for _ in range(400):
            o, _ = probe.reset()
            hits += int(ag.act(o) == probe.tgt * 2 + 1 + (8 if probe.tgt == 0 else 0))
        T.check(hits / 400 > 0.9, f"{algo}: learns the toy task, incl. the delayed request reward", f"accuracy {hits / 400:.2f}")
        stored = ag.debug_stored_in_control
        T.check(len(stored) > 0 and all(stored) and len(stored) == env_counter["in_ctrl"],
                f"{algo}: no emergency step reached a learning update", f"stored {len(stored)}, in-control {env_counter['in_ctrl']}")
        # save / load round trip gives the same actions
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_agent.pt")
        ag.save(tmp)
        ag2 = load_agent(tmp, c_run)
        os.remove(tmp)
        obs = np.random.default_rng(0).normal(size=(50, d)).astype(np.float32)
        T.check(np.array_equal(ag.act_batch(obs, deterministic=True), ag2.act_batch(obs, deterministic=True)), f"{algo}: save/load keeps the policy")
    # factorized head: out[a] = base[a & 7] + gen[a >> 3]
    fh = Factored16(d, [256, 256])
    x = torch.randn(5, d)
    y, raw = fh(x), fh.net(x)
    T.check(all(torch.allclose(y[:, a], raw[:, a & 7] + raw[:, 8 + (a >> 3)]) for a in range(16)), "factorized head combines 8 + 2 outputs into the 16 actions")
    # masks
    ag = make_agent("ppo", cfg.replace(USE_ACTION_MASK=True), seed=1, obs_dim=d)
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[7] = True
    T.check(ag.act(np.zeros(d, dtype=np.float32), mask=m) == 7, "action masking ablation restricts the choice")
    # normalizer
    x = np.random.default_rng(1).normal(5, 3, size=(1000, 4))
    x[:, 3] = 2.0
    nz = ObsNormalizer.fit(x)
    y = nz(x)
    T.check(abs(y[:, 0].mean()) < 1e-3 and abs(y[:, 0].std() - 1) < 1e-2 and np.all(y[:, 3] == 0), "normalizer: zero mean, unit std, constant features safe")
    # rule-based baseline uses the same act(state) interface on real HEMS states
    import tempfile, shutil
    import hems_forecast_feed as ff
    from hems_config import synthetic_overrides
    tmpd = tempfile.mkdtemp(prefix="hems_ag_")
    try:
        c2 = make_config(**synthetic_overrides(tmpd))
        ff.make_synthetic_inputs(c2, seed=3)
        data = ff.build_split(c2, "train")
        env = HEMSEnv(c2, data, mode="eval", seed=2)
        rb = RuleBasedAgent(c2)
        o, _ = env.reset(seed=5)
        rej = 0
        for _ in range(288):
            o, r, te, tr, info = env.step(rb.act(o))
            rej += info["rejected"]
        T.check(rej <= 3, "rule-based baseline proposes safe actions (almost never rejected)", f"rejected {rej}")
        nz = fit_normalizer(c2, data, "ensemble", 1000, seed=1)
        T.check(nz.mean.shape == (N_FEATURES,) and np.all(nz.std > 0), "normalizer fitted on training-split rollouts")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    out = new_run_dir(make_config(), "selftest", "hems_agents", base=out_root) if out_root else None
    return T.summary(out)


if __name__ == "__main__":
    from hems_config import _default_selftest_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.exit(0 if selftest(a.out or _default_selftest_root()) else 1)
