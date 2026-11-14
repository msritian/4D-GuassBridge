"""C6: checks of guidance/guide.py and the guidance hook in model/flow.sample.

The step-size rule is checked on a toy energy with a known answer (a linear residual, where the Polyak step is exact),
then on real physics: a planner motion with a cube pushed into the table, which one guided step must lift out."""
from pathlib import Path

import pytest
import torch

from gaussbridge.guidance.guide import MAX_CELL_STEP, guidable_mask, guidance_step
from gaussbridge.model.flow import loss_mask, sample
from gaussbridge.trajectory import POS_SCALE, known_mask
from test_motion_flow import COND, make_batch

REPO = Path(__file__).parents[1]


def toy_energy(cells, target):
    """E = sum over the chosen cells of (x - target)^2: residuals linear in x, so the Polyak step is exact."""
    return lambda x: (x[cells] - target).reshape(x.shape[0], -1).square().sum(1)


def _free(batch):
    return guidable_mask(loss_mask(batch["known"], batch["ent_mask"], batch["x0"].shape[-1]))


def _cells(B):
    """Object slot 1, position feature 0, steps 30..39 of every sample (generated cells)."""
    c = torch.zeros(B, 128, 4, 10, dtype=torch.bool)
    c[:, 30:40, 1, 0] = True
    return c


@pytest.mark.parametrize("lam,left", [(2.0, 0.0), (1.0, 0.5), (0.5, 0.75), (0.0, 1.0)])
def test_lambda_is_the_fraction_of_the_violation_removed(lam, left):
    """lambda = 2 removes a linear violation, 1 halves it, 0 does nothing; only the allowed cells change."""
    batch = make_batch()
    x = batch["x0"].clone()
    cells = _cells(len(x))
    target = x[cells] + 0.01                                                  # 1 cm-ish below where it should be
    delta, e = guidance_step(x, toy_energy(cells, target), lam, _free(batch))
    x2 = x - delta
    torch.testing.assert_close(x2[cells] - target, left * (x[cells] - target), atol=1e-6, rtol=0)
    assert (delta[~cells] == 0).all() and e.shape == (len(x),)


def test_step_is_capped_and_zero_energy_means_no_step():
    batch = make_batch()
    x = batch["x0"].clone()
    cells = _cells(len(x))
    delta, _ = guidance_step(x, toy_energy(cells, x[cells] + 1.0), 2.0, _free(batch))     # a huge violation
    assert delta.abs().max() == pytest.approx(MAX_CELL_STEP, rel=1e-5)
    delta, e = guidance_step(x, toy_energy(cells, x[cells]), 2.0, _free(batch))           # nothing to fix
    assert (e == 0).all() and (delta == 0).all()


def test_given_cells_and_attached_flags_are_never_moved():
    batch = make_batch("objects")
    x = batch["x0"].clone()
    cells = torch.ones_like(x, dtype=torch.bool)                              # an energy that touches every cell
    delta, _ = guidance_step(x, toy_energy(cells, (x + 0.01)[cells]), 1.0, _free(batch))
    assert (delta[batch["known"]] == 0).all()                                 # start + objects' end
    assert (delta[:, :, 1:, 9] == 0).all()                                    # attached flags
    assert (delta[:, :, 3] == 0).all()                                        # padded slot
    assert (delta[:, 1:-1, 0, :3] != 0).all()                                 # but the generated robot cells do move


class Still(torch.nn.Module):
    """Velocity 0: without guidance the sampler would keep its initial noise; x0_hat = x at every step."""

    def forward(self, x_t, dur_t, tau, **cond):
        return torch.zeros_like(x_t), torch.zeros_like(dur_t)


def test_sampler_hook():
    """guide_scale 0 / no guidance = the unguided sampler bit for bit; with guidance the target is reached, only
    below tau = 0.6, and the given cells stay exact."""
    batch = make_batch()
    cond = [batch[k] for k in COND]
    gen = lambda: torch.Generator().manual_seed(0)
    base, _ = sample(Still(), *cond, steps=10, generator=gen())
    cells = _cells(len(base))
    energy = toy_energy(cells, base[cells] + 0.02)              # small, so the per-step cap does not bind
    off, _ = sample(Still(), *cond, steps=10, generator=gen(), guidance=energy, guide_scale=0.0)
    assert torch.equal(off, base)
    trace = []
    x, _ = sample(Still(), *cond, steps=10, generator=gen(), guidance=energy, guide_scale=2.0, trace=trace)
    torch.testing.assert_close(x[cells], base[cells] + 0.02, atol=1e-6, rtol=0)  # pulled exactly onto the target
    assert all(t["tau"] < 0.6 for t in trace) and len(trace) == 5             # taus 0.5, 0.4, .., 0.1 (of 1.0, 0.9, ..)
    assert torch.equal(x[batch["known"]], batch["x_known"][batch["known"]])
    torch.testing.assert_close(x[~cells], base[~cells])                       # nothing else changed


# ---------------------------------------------------------------------------------------------------- real physics
pk = pytest.importorskip("pytorch_kinematics")
pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def stack():
    from gaussbridge.guidance.guide import PhysicsEnergy
    from gaussbridge.model.dataset import MotionDataset
    from gaussbridge.sim.extract import load_shard
    from gaussbridge.sim.kinematics import PandaFK

    ds = MotionDataset(load_shard(REPO / "artifacts/raw/StackCube-v1/dev"), endpoint_noise=False)
    return ds, PhysicsEnergy(PandaFK("panda_wristcam"), ds.n_objects)


def test_guidance_lifts_a_cube_out_of_the_table(stack):
    """cubeB (never held, resting) is pushed 1 cm into the table for 20 steps. One guided step with lambda = 2
    removes the penetration (to < 1 mm), lambda = 1 halves it. Elsewhere the motion barely moves: cubeB < 0.01 mm,
    the rest < 1.5 mm (the attachment term's own small correction of the planner's grasp, ~0.5 mm)."""
    ds, energy = stack
    x = ds.x0[:1].clone()
    x[0, 40:60, 2, 2] -= 0.01 / POS_SCALE
    known = known_mask(ds.ent_mask[None], "full")
    free = guidable_mask(loss_mask(known, ds.ent_mask[None], x.shape[-1]))
    pen0 = energy.scene.max_penetration(x).item()
    assert pen0 == pytest.approx(0.01, abs=5e-4)
    depth = {}
    for lam in (1.0, 2.0):
        delta, _ = guidance_step(x, energy, lam, free)
        depth[lam] = energy.scene.max_penetration(x - delta).item()
        outside = torch.cat([delta[0, :40], delta[0, 60:]])
        assert outside[:, 2].abs().max() < 1e-5 and outside.abs().max() < 0.005            # local fix
    assert depth[2.0] < 0.001 and depth[1.0] == pytest.approx(0.005, abs=0.0015), depth


def test_guidance_steadies_a_jittering_grasp(stack):
    """cubeA jitters 3 mm in the hand, every other step of the carry: three guided steps (lambda = 1) make it steady
    again (attachment energy down > 10x; measured 40x) without creating penetration. (A slow 2 cm drift instead only
    halves in 5 steps: low-frequency, left to project_attachment, see the module docstring.)"""
    from gaussbridge.guidance.attach import attach_energy, longest_run
    from gaussbridge.trajectory import decode_motion

    ds, energy = stack
    x = ds.x0[:1].clone()
    n0, n1 = longest_run(decode_motion(x, 0.0)["attached"][0, :, 0] > 0)
    jitter = torch.zeros(n1 - n0 - 10)
    jitter[::2] = 0.003
    x[0, n0 + 5:n1 - 5, 1, 1] += jitter / POS_SCALE
    known = known_mask(ds.ent_mask[None], "full")
    free = guidable_mask(loss_mask(known, ds.ent_mask[None], x.shape[-1]))
    e0 = attach_energy(x, 2, energy.fk).item()
    for _ in range(3):
        x = x - guidance_step(x, energy, 1.0, free)[0]
    assert attach_energy(x, 2, energy.fk).item() < e0 / 10
    assert energy.scene.max_penetration(x).item() < 0.005
