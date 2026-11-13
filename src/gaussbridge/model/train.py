"""C4: the training recipe for the motion model (CLAUDE.md §8.2), as a library used by scripts/train_motion.py.

Recipe
------
    data         all goal modes (none 0.1 / objects 0.3 / full 0.6), endpoint noise on the given copy
    optimiser    AdamW (lr 2e-4, weight decay 0.01), gradient clipping 1.0
    lr schedule  linear warm-up (2k steps), then cosine decay to min_lr_ratio * lr at the last step
    precision    bf16 autocast on GPU (weights, optimiser and EMA stay in fp32)
    EMA          exponential moving average of the weights (decay 0.999); validation and sampling use it
    checkpoints  every ckpt_every steps (atomic write, last keep_ckpts kept); a run resumes from the newest one

Exact resume
------------
Everything random in training step n (which episodes, goal modes, endpoint noise, tau, eps) comes from one generator
seeded with (seed, n). So a run stopped after step n and resumed from its checkpoint draws exactly the batches an
uninterrupted run would draw, and (on CPU) ends with bit-identical weights. On GPU, kernels are not bitwise
deterministic, so resumed runs match closely but not exactly.

Time budget (CHTC)
------------------
With max_minutes set, the trainer checkpoints and stops when the budget is used; `fit` then returns "paused" and the
job script exits with code 85, which HTCondor's self-checkpointing treats as "requeue me and resume" (chtc/train.sub).

Configuration
-------------
Plain YAML (decision 2026-11-13, instead of Hydra): `load_config(path, overrides)`; overrides are "a.b=value" strings
parsed as YAML values. The resolved config is saved in every checkpoint and as config.yaml in the run directory.
"""
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

from gaussbridge.model.dataset import MotionDataset
from gaussbridge.model.dit import MotionDiT
from gaussbridge.model.flow import flow_loss, sample
from gaussbridge.trajectory import Episode, decode_motion, known_mask

EXIT_PAUSED = 85  # HTCondor checkpoint_exit_code: "not finished, resume me"


# ---------------------------------------------------------------------------------------------------- config
def load_config(path, overrides=()):
    """YAML file -> dict, with "a.b=value" overrides applied (values parsed as YAML: numbers, bools, lists...).
    Only keys that exist in the file can be overridden, so a typo fails loudly instead of being ignored."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            if not isinstance(node.get(k), dict):
                raise KeyError(f"unknown config key {key!r}")
            node = node[k]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[leaf] = parse_value(value)
    return cfg


def parse_value(text):
    """A YAML value, except that number-like strings become floats: PyYAML (YAML 1.1) reads "1e-3" as a string,
    and only "1.0e-3" as a number, a classic source of silent config bugs."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


# ---------------------------------------------------------------------------------------------------- pieces
def lr_factor(step, warmup, total, min_ratio=0.0):
    """Learning-rate multiplier at optimiser step `step` (0-based): linear warm-up to 1 over `warmup` steps, then a
    cosine decay to `min_ratio` at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class EMA:
    """Exponential moving average of a model's weights: ema <- decay * ema + (1 - decay) * weights, after every
    optimiser step. Kept in fp32 on the model's device; `module` is a full copy usable for evaluation."""

    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.decay)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


def step_generator(seed, step, device):
    """The generator for everything random in training step `step` (see "Exact resume")."""
    return torch.Generator(device).manual_seed(seed * 1_000_003 + step)


def load_episode_dir(path):
    files = sorted(Path(path).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz episodes in {path}")
    return [Episode.load(f) for f in files]


def save_atomic(obj, path):
    """torch.save to a temporary file, then rename: a job killed mid-write never leaves a broken checkpoint."""
    tmp = Path(f"{path}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoints(ckpt_dir):
    """Checkpoint files in a directory, oldest first."""
    return sorted(Path(ckpt_dir).glob("ckpt_*.pt"))


# ---------------------------------------------------------------------------------------------------- trainer
class Trainer:
    """Holds data, model, optimiser, schedule and EMA for one run directory; `fit()` trains (resuming if a
    checkpoint exists) and returns "done" or "paused"."""

    def __init__(self, cfg, run_dir, device=None):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = self.run_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        d, o = cfg["data"], cfg["optim"]

        train_eps = load_episode_dir(d["train"])
        self.train = MotionDataset(train_eps, p_none=d["p_none"], p_objects=d["p_objects"],
                                   endpoint_noise=d["endpoint_noise"]).to(self.device)
        self.val = None
        if d.get("val"):
            self.val = MotionDataset(load_episode_dir(d["val"]), dur_stats=self.train.dur_stats, p_none=d["p_none"],
                                     p_objects=d["p_objects"], endpoint_noise=False).to(self.device)
            if self.val.obj_names != self.train.obj_names:
                raise ValueError(f"val objects {self.val.obj_names} != train {self.train.obj_names}")
        self.robot_uid = train_eps[0].meta.get("robot_uid", "panda_wristcam")

        torch.manual_seed(cfg["seed"])  # the initial weights depend only on the config's seed (reproducible runs)
        self.model = MotionDiT(**cfg["model"]).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, o["warmup"], o["steps"], o.get("min_lr_ratio", 0.0)))
        self.ema = EMA(self.model, o["ema"])
        self.step = 0
        self.bf16 = o["bf16"] and self.device == "cuda"
        self._fk = None

    # ------------------------------------------------------------------ checkpoints
    def state(self):
        return {"step": self.step, "model": self.model.state_dict(), "ema": self.ema.state_dict(),
                "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "config": self.cfg,
                "model_config": self.model.config, "dur_stats": self.train.dur_stats,
                "obj_names": self.train.obj_names, "robot_uid": self.robot_uid}

    def save(self):
        save_atomic(self.state(), self.ckpt_dir / f"ckpt_{self.step:07d}.pt")
        for old in checkpoints(self.ckpt_dir)[:-self.cfg["log"]["keep_ckpts"]]:
            old.unlink()

    def resume(self):
        """Load the newest checkpoint of this run, if any. Returns the step resumed from (0 = fresh start)."""
        found = checkpoints(self.ckpt_dir)
        if not found:
            return 0
        s = torch.load(found[-1], map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.ema.load_state_dict(s["ema"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.step = s["step"]
        return self.step

    # ------------------------------------------------------------------ training
    def train_step(self):
        """One optimiser step (step index self.step, then incremented). Returns the loss statistics."""
        o = self.cfg["optim"]
        g = step_generator(self.cfg["seed"], self.step, self.device)
        idx = torch.randint(len(self.train), (o["batch"],), generator=g, device=self.device)
        batch = self.train.make_batch(idx, g)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.bf16):
            loss, stats = flow_loss(self.model, batch, g)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        stats["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.model.parameters(), o["grad_clip"]).item()
        self.opt.step()
        self.sched.step()
        self.ema.update(self.model)
        self.step += 1
        stats["lr"] = self.sched.get_last_lr()[0]
        if not math.isfinite(stats["loss"]):
            raise FloatingPointError(f"loss is {stats['loss']} at step {self.step}")
        return stats

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def val_loss(self, chunk=256):
        """Mean flow loss of the EMA model over the whole val set, with fixed randomness (comparable over time)."""
        total, n = 0.0, 0
        for start in range(0, len(self.val), chunk):
            idx = torch.arange(start, min(start + chunk, len(self.val)), device=self.device)
            g = step_generator(self.cfg["seed"] + 7919, start, self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.bf16):
                loss, _ = flow_loss(self.ema.module, self.val.make_batch(idx, g), g)
            total += loss.item() * len(idx)
            n += len(idx)
        return total / n

    @torch.no_grad()
    def val_samples(self, n_samples, sample_steps, goal="objects", max_cases=None):
        """Sample n_samples motions per val start (EMA model, goal from the case's first reference) and score them
        against all references of that start with the C5 metrics. Returns mean metrics over cases."""
        from gaussbridge.eval import metrics as M  # needs pytorch_kinematics + the URDF (FK)
        from gaussbridge.sim.kinematics import PandaFK

        if self._fk is None:
            self._fk = PandaFK(self.robot_uid, device=self.device)
        cases = {}
        for i, m in enumerate(self.val.meta):
            cases.setdefault(m["seed"], []).append(i)
        rows = []
        for seed, idx in list(cases.items())[:max_cases]:
            x0 = self.val.x0[idx]
            ent = self.val.ent_mask.expand(n_samples, -1)
            known = known_mask(ent, goal)
            x_known = torch.where(known[..., None], x0[:1].expand(n_samples, -1, -1, -1), torch.zeros_like(x0[:1]))
            shapes = self.val.shapes.expand(n_samples, *self.val.shapes.shape)
            g = torch.Generator(self.device).manual_seed(int(seed))
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.bf16):
                x, log_dur = sample(self.ema.module, x_known, known, ent, shapes, steps=sample_steps, generator=g)
            x, log_dur = x.float(), log_dur.float()
            dur = decode_motion(x, log_dur, self.train.dur_stats)["duration"]
            tracks = M.motion_tracks(x, self.val.n_objects, self._fk)
            refs = M.motion_tracks(x0, self.val.n_objects, self._fk)
            row = M.case_metrics(tracks, refs, dur)
            row["coverage_1cm"] = M.coverage(tracks, refs, 0.01).item()
            row["precision_1cm"] = M.precision(tracks, refs, 0.01).item()
            end = M.endpoint_errors(x, x0[:1].expand_as(x), self.val.n_objects, self._fk)
            row["end_obj_pos_cm"] = end["obj_pos_cm"].mean().item()
            row["end_obj_rot_deg"] = end["obj_rot_deg"].mean().item()
            rows.append(row)
        return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}

    # ------------------------------------------------------------------ loop
    def fit(self, max_minutes=None, stop_after=None, log=print):
        """Train to cfg.optim.steps, resuming from the newest checkpoint. Stops early (with a checkpoint) when
        max_minutes are used or after `stop_after` steps in this call (tests). Returns "done" or "paused"."""
        L, total = self.cfg["log"], self.cfg["optim"]["steps"]
        resumed = self.resume()
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        log(json.dumps({"event": "start", "resumed_from": resumed, "device": self.device, "bf16": self.bf16,
                        "train_episodes": len(self.train), "val_episodes": len(self.val) if self.val else 0,
                        "params_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 2)}))
        t0, done_here, window = time.time(), 0, []
        with open(self.run_dir / "log.jsonl", "a") as f:
            def emit(row):
                f.write(json.dumps(row) + "\n")
                f.flush()
                log(json.dumps(row))

            while self.step < total:
                window.append(self.train_step())
                done_here += 1
                if self.step % L["every"] == 0 or self.step == total:
                    avg = {k: sum(w[k] for w in window) / len(window) for k in window[0]}
                    emit({"step": self.step, "minutes": round((time.time() - t0) / 60, 2), **avg})
                    window = []
                if self.val is not None and (self.step % L["val_loss_every"] == 0 or self.step == total):
                    emit({"step": self.step, "val_loss": self.val_loss()})
                if self.val is not None and L.get("val_sample_every") and (
                        self.step % L["val_sample_every"] == 0 or self.step == total):
                    emit({"step": self.step, **{f"val_{k}": v for k, v in self.val_samples(
                        L["val_samples"], L["sample_steps"], max_cases=L.get("val_max_cases")).items()}})
                if self.step % L["ckpt_every"] == 0 or self.step == total:
                    self.save()
                out_of_time = max_minutes is not None and (time.time() - t0) / 60 >= max_minutes
                if self.step < total and (out_of_time or (stop_after is not None and done_here >= stop_after)):
                    if window:  # keep the losses of the unfinished logging window (marked partial)
                        avg = {k: sum(w[k] for w in window) / len(window) for k in window[0]}
                        emit({"step": self.step, "partial": True, **avg})
                    self.save()
                    emit({"event": "paused", "step": self.step})
                    return "paused"
        torch.save({"model": self.ema.state_dict(), "model_config": self.model.config, "dur_stats":
                    self.train.dur_stats, "obj_names": self.train.obj_names, "robot_uid": self.robot_uid,
                    "config": self.cfg, "step": self.step}, self.run_dir / "final_ema.pt")
        log(json.dumps({"event": "done", "step": self.step}))
        return "done"
