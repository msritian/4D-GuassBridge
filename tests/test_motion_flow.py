"""C3: checks of model/flow.py (CPU).

The two "oracle" tests are the important ones. An oracle knows the true motion x0 and returns the exact velocity
(x_t - x0) / tau. If the noising, the training target and the sampler's step direction are mutually consistent,
the oracle's loss is 0 and the sampler, following the oracle, lands exactly on x0. A sign or scaling mistake
anywhere breaks one of these. The other tests check the masks, exact endpoints and that a small model learns."""
import pytest
import torch

from gaussbridge.model.dataset import GOAL_FULL, GOAL_NAMES, MotionDataset
from gaussbridge.model.dit import MotionDiT
from gaussbridge.model.flow import ROBOT_PAD_FEATURE, flow_loss, model_inputs, noisy_sample, sample
from gaussbridge.trajectory import known_mask
from motion_fakes import fake_episode

COND = ("x_known", "known", "ent_mask", "shapes")


def make_batch(goal="full", n=4, obj_names=("cubeA", "cubeB")):
    """A batch from synthetic episodes with a fixed goal mode and exact (un-noised) endpoints."""
    ds = MotionDataset([fake_episode(140 + 20 * i, obj_names, seed=i) for i in range(n)], endpoint_noise=False)
    batch = torch.utils.data.default_collate([ds[i] for i in range(n)])
    batch["known"] = known_mask(batch["ent_mask"], goal)
    batch["x_known"] = torch.where(batch["known"][..., None], batch["x0"], torch.zeros_like(batch["x0"]))
    return batch


class Oracle(torch.nn.Module):
    """Knows the answer: velocity (x_t - x0) / tau, which is exactly eps - x0 on the line x_t = (1-tau) x0 + tau eps."""

    def __init__(self, x0, dur0):
        super().__init__()
        self.x0, self.dur0 = x0, dur0

    def forward(self, x_t, dur_t, tau, **cond):
        return (x_t - self.x0) / tau[:, None, None, None], (dur_t - self.dur0) / tau


def test_noisy_sample_ends_and_given_cells():
    batch = make_batch()
    eps, eps_dur = torch.randn_like(batch["x0"]), torch.randn(4)
    real = batch["ent_mask"][:, None, :, None].expand_as(eps).clone()
    real[:, :, 0, ROBOT_PAD_FEATURE] = False                      # the robot spacer is not a real number either
    free = (~batch["known"][..., None]).expand_as(eps) & real
    x_0, d_0 = noisy_sample(batch, torch.zeros(4), eps, eps_dur)   # tau = 0: the data
    x_1, d_1 = noisy_sample(batch, torch.ones(4), eps, eps_dur)    # tau = 1: pure noise
    torch.testing.assert_close(x_0, batch["x0"])
    torch.testing.assert_close(x_1[free], eps[free])
    torch.testing.assert_close(d_0, batch["log_dur"]) and torch.testing.assert_close(d_1, eps_dur)
    for x in (x_0, x_1):  # at every noise level: given cells hold their given value, padding and spacer are 0
        assert torch.equal(x[batch["known"]], batch["x_known"][batch["known"]])
        assert (x[~real] == 0).all()


def test_oracle_has_zero_loss():
    batch = make_batch()
    loss, stats = flow_loss(Oracle(batch["x0"], batch["log_dur"]), batch, torch.Generator().manual_seed(0))
    assert loss.item() < 1e-8, stats


@pytest.mark.parametrize("goal", ["full", "objects", "none"])
def test_sampler_following_the_oracle_lands_on_the_data(goal):
    """Euler steps along the exact velocity reproduce x0 (and the duration): the sampler walks the right way."""
    batch = make_batch(goal)
    x, dur = sample(Oracle(batch["x0"], batch["log_dur"]), *(batch[k] for k in COND), steps=10,
                    generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(x, batch["x0"], atol=1e-5, rtol=0)
    torch.testing.assert_close(dur, batch["log_dur"], atol=1e-5, rtol=0)


def test_loss_ignores_given_padded_and_spacer_cells():
    batch = make_batch("objects")

    def loss_of(b):
        model = MotionDiT(d_model=32, depth=1, heads=2)
        torch.manual_seed(0)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        return flow_loss(model, b, torch.Generator().manual_seed(1))[0].item()

    base = loss_of(batch)
    ignored = {k: v.clone() for k, v in batch.items()}
    ignored["x0"][:, 0] += 3.0                        # start row: given
    ignored["x0"][:, -1, 1:3] += 3.0                  # objects' end: given in "objects" mode
    ignored["x0"][:, :, 3] += 3.0                     # padded slot
    ignored["x0"][:, :, 0, ROBOT_PAD_FEATURE] += 3.0  # robot spacer feature
    assert loss_of(ignored) == pytest.approx(base, abs=1e-6)
    counted = {k: v.clone() for k, v in batch.items()}
    counted["x0"][:, -1, 0, :3] += 3.0                # robot end: NOT given in "objects" mode -> must count
    assert abs(loss_of(counted) - base) > 1e-3


@pytest.mark.parametrize("goal", ["full", "objects", "none"])
def test_samples_keep_given_cells_exactly(goal):
    """C3 acceptance: sampled endpoints equal the conditioning exactly (max abs diff 0), even for a random network."""
    batch = make_batch(goal, obj_names=("cube",))
    model = MotionDiT(d_model=32, depth=1, heads=2)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    x, dur = sample(model, *(batch[k] for k in COND), steps=5, generator=torch.Generator().manual_seed(0))
    known = batch["known"]
    assert torch.equal(x[known], batch["x_known"][known])
    assert (x[:, :, 2:] == 0).all()                                   # PickCube: slots 2, 3 are padding
    assert torch.isfinite(x).all() and torch.isfinite(dur).all()


def test_sampling_is_reproducible_and_diverse():
    batch = make_batch()
    model = MotionDiT(d_model=32, depth=1, heads=2)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    cond = [batch[k] for k in COND]
    a, _ = sample(model, *cond, steps=5, generator=torch.Generator().manual_seed(0))
    b, _ = sample(model, *cond, steps=5, generator=torch.Generator().manual_seed(0))
    c, _ = sample(model, *cond, steps=5, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)                                          # same noise -> same motion
    assert (a[:, 1:-1] - c[:, 1:-1]).abs().max() > 1e-3               # different noise -> different middle


def test_small_model_learns_two_motions():
    """End to end: train a tiny network on 2 synthetic motions; its samples move most of the way to them.
    (Measured while writing: mean |error| 0.65 untrained -> 0.06 after 100 steps -> 0.02 after 600.)"""
    torch.manual_seed(0)
    ds = MotionDataset([fake_episode(150, seed=0), fake_episode(200, seed=1)], endpoint_noise=False,
                       p_none=0.0, p_objects=0.0)
    batch = torch.utils.data.default_collate([ds[i % 2] for i in range(8)])
    assert (batch["goal"] == GOAL_FULL).all(), GOAL_NAMES
    model = MotionDiT(d_model=64, depth=2, heads=4)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    cond = {k: v[:2] for k, v in model_inputs(batch).items()}

    def sample_error():
        x, _ = sample(model, **cond, steps=20, generator=torch.Generator().manual_seed(0))
        return (x - batch["x0"][:2]).abs().mean().item()

    before = sample_error()
    losses = []
    for _ in range(150):
        loss, _ = flow_loss(model, batch)  # one random noise level per sample: single values are noisy
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = sample_error()
    early, late = sum(losses[:10]) / 10, sum(losses[-20:]) / 20
    assert late < early / 5, (early, late)
    assert after < before / 4, (before, after)


def test_uniform_tau_mix_reaches_low_noise():
    g = torch.Generator().manual_seed(0)
    from gaussbridge.model.flow import sample_tau
    assert (sample_tau(20000, g) < 0.02).float().mean() < 0.001          # logit-normal: almost never
    low = (sample_tau(20000, g, p_uniform=0.3) < 0.02).float().mean()
    assert 0.004 < low < 0.008                                             # 0.3 * 0.02 = 0.006


@pytest.mark.parametrize("p_uniform,w", [(0.3, 0.0), (0.0, 1.0)])
def test_oracle_has_zero_loss_with_the_jitter_fixes(p_uniform, w):
    batch = make_batch()
    loss, stats = flow_loss(Oracle(batch["x0"], batch["log_dur"]), batch, torch.Generator().manual_seed(0),
                            p_uniform=p_uniform, w_vel=w, w_acc=w)
    assert loss.item() < 1e-8, stats


def test_smoothness_loss_sees_jitter_the_velocity_loss_averages_out():
    """Same per-cell error size, one smooth (a constant offset) and one alternating: the smoothness loss is 0 for the
    offset and large for the alternation."""
    from gaussbridge.model.flow import smoothness_loss
    batch = make_batch()
    x0, ent = batch["x0"], batch["ent_mask"]
    offset = x0 + 0.01
    sign = torch.tensor([1.0, -1.0]).repeat(64)[None, :, None, None]
    jitter = x0 + 0.01 * sign
    for order in (1, 2):
        assert smoothness_loss(offset, x0, ent, order) < 1e-10
        assert smoothness_loss(jitter, x0, ent, order) > (0.02 * order) ** 2 * 0.5


@pytest.mark.parametrize("goal", ["full", "objects"])
def test_last_tau_jump_still_lands_on_the_data(goal):
    batch = make_batch(goal)
    x, dur = sample(Oracle(batch["x0"], batch["log_dur"]), *(batch[k] for k in COND), steps=10, last_tau=0.1,
                    generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(x, batch["x0"], atol=1e-5, rtol=0)
    torch.testing.assert_close(dur, batch["log_dur"], atol=1e-5, rtol=0)
