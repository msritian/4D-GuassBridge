"""C4: checks of model/train.py (CPU, tiny model, a few seconds).

The key test is exact resume: CHTC can stop a job at any time, so a run stopped and resumed from its checkpoint must
end exactly where an uninterrupted run ends (same weights, same EMA, same logged losses)."""
import json
import math
from pathlib import Path

import pytest
import torch

from gaussbridge.model.train import EMA, Trainer, checkpoints, load_config, lr_factor
from motion_fakes import fake_episode

CONFIG = Path(__file__).parents[1] / "configs/train/stackcube_v1.yaml"
TINY = ["model.d_model=32", "model.depth=1", "model.heads=2", "optim.batch=4", "optim.steps=12", "optim.warmup=3",
        "log.every=2", "log.val_loss_every=4", "log.val_sample_every=6", "log.val_samples=2", "log.sample_steps=2",
        "log.val_max_cases=2", "log.ckpt_every=4", "log.keep_ckpts=2"]


@pytest.fixture
def data_dirs(tmp_path):
    for split, seeds in (("train", range(6)), ("val", range(100, 104))):
        d = tmp_path / split
        d.mkdir()
        for s in seeds:
            ep = fake_episode(140 + 7 * (s % 5), seed=s)
            ep.meta = {"seed": s // 2, "robot_uid": "panda_wristcam"}           # two references per start
            ep.save(d / f"{s:05d}.npz")
    return tmp_path


def tiny_config(data_dirs, extra=()):
    return load_config(CONFIG, TINY + [f"data.train={data_dirs / 'train'}", f"data.val={data_dirs / 'val'}",
                                       *extra])


def test_lr_schedule():
    assert lr_factor(0, 10, 100) == pytest.approx(0.1) and lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100) == pytest.approx(0.5)                     # half way through the cosine
    assert lr_factor(100, 10, 100) == pytest.approx(0.0)
    assert lr_factor(100, 10, 100, min_ratio=0.1) == pytest.approx(0.1)


def test_ema_update():
    m = torch.nn.Linear(2, 2)
    ema = EMA(m, decay=0.9)
    old = ema.module.weight.clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    torch.testing.assert_close(ema.module.weight, 0.9 * old + 0.1 * m.weight)
    assert not any(p.requires_grad for p in ema.module.parameters())


def test_config_overrides(tmp_path):
    cfg = load_config(CONFIG, ["optim.lr=1e-3", "optim.bf16=false", "data.val=null"])
    assert cfg["optim"]["lr"] == 1e-3 and cfg["optim"]["bf16"] is False and cfg["data"]["val"] is None
    for bad in ("optim.lr_rate=1", "optimm.lr=1", "nonsense"):
        with pytest.raises((KeyError, ValueError)):
            load_config(CONFIG, [bad])


def _log_losses(run):
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in rows if "loss" in r}


def test_resume_is_exact(data_dirs, tmp_path):
    """Uninterrupted 12 steps == 5 steps, stop, new Trainer object, resume to 12 (bit-identical on CPU)."""
    cfg = tiny_config(data_dirs)
    straight = Trainer(cfg, tmp_path / "a", device="cpu")
    assert straight.fit(log=lambda s: None) == "done"

    first = Trainer(cfg, tmp_path / "b", device="cpu")
    assert first.fit(stop_after=5, log=lambda s: None) == "paused" and first.step == 5
    second = Trainer(cfg, tmp_path / "b", device="cpu")                    # a fresh process, as on CHTC
    assert second.fit(log=lambda s: None) == "done" and second.step == 12

    for k, v in straight.model.state_dict().items():
        assert torch.equal(v, second.model.state_dict()[k]), k
    for k, v in straight.ema.state_dict().items():
        assert torch.equal(v, second.ema.state_dict()[k]), k
    assert straight.opt.state_dict()["state"][0]["step"] == second.opt.state_dict()["state"][0]["step"]
    # same batches, same losses; except the logging window cut by the pause (steps 5-6: logged as 5 and 6 separately)
    a_log, b_log = _log_losses(tmp_path / "a"), _log_losses(tmp_path / "b")
    assert {k: v for k, v in a_log.items() if k != 6} == {k: v for k, v in b_log.items() if k not in (5, 6)}
    assert 5 in b_log                                                      # the partial window was not dropped


def test_logging_checkpoints_and_outputs(data_dirs, tmp_path):
    run = tmp_path / "run"
    t = Trainer(tiny_config(data_dirs), run, device="cpu")
    assert t.fit(log=lambda s: None) == "done"
    names = [p.name for p in checkpoints(run / "ckpt")]
    assert names == ["ckpt_0000008.pt", "ckpt_0000012.pt"]                # every 4 steps, keep the last 2
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    val_loss = [r for r in rows if "val_loss" in r]
    assert [r["step"] for r in val_loss] == [4, 8, 12] and all(math.isfinite(r["val_loss"]) for r in val_loss)
    sampled = [r for r in rows if "val_coverage" in r]
    assert [r["step"] for r in sampled] == [6, 12]
    assert {"val_min_ade_m", "val_coverage_1cm", "val_end_obj_pos_cm", "val_apd_m"} <= set(sampled[0])
    lrs = [r["lr"] for r in rows if "lr" in r]
    assert lrs[-1] < max(lrs)                                              # cosine decay happened
    assert (run / "config.yaml").exists() and (run / "final_ema.pt").exists()
    final = torch.load(run / "final_ema.pt", weights_only=False)
    assert final["dur_stats"] == t.train.dur_stats and final["model_config"]["d_model"] == 32


def test_time_budget_pauses_with_a_checkpoint(data_dirs, tmp_path):
    t = Trainer(tiny_config(data_dirs), tmp_path / "run", device="cpu")
    assert t.fit(max_minutes=0, log=lambda s: None) == "paused"           # budget used up after the first step
    assert t.step == 1 and [p.name for p in checkpoints(tmp_path / "run" / "ckpt")] == ["ckpt_0000001.pt"]


def test_same_seed_same_initial_weights_and_number_like_values(data_dirs, tmp_path):
    a = Trainer(tiny_config(data_dirs), tmp_path / "a", device="cpu")
    b = Trainer(tiny_config(data_dirs), tmp_path / "b", device="cpu")
    c = Trainer(tiny_config(data_dirs, ["seed=1"]), tmp_path / "c", device="cpu")
    assert all(torch.equal(x, y) for x, y in zip(a.model.parameters(), b.model.parameters()))
    assert not all(torch.equal(x, y) for x, y in zip(a.model.parameters(), c.model.parameters()))
    cfg = load_config(CONFIG)
    assert all(isinstance(cfg["optim"][k], float) for k in ("lr", "weight_decay", "grad_clip", "ema"))
