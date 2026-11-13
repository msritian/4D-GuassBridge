"""C3: checks of model/encoders.py and model/dit.py (CPU, small configurations, a few seconds).

These test the WIRING of the network, the kind of bug that trains without crashing but silently learns less:
padding that leaks into real tokens, inputs that are never used, samples in a batch that mix, a wrong token order.
With the DiT's zero initialisation the output is always 0, which would make "does X change the output?" checks
pass trivially, so most tests first fill all weights with random values (`randomise`)."""
import pytest
import torch

from gaussbridge.model.dataset import box_shape_points
from gaussbridge.model.dit import MotionDiT
from gaussbridge.model.encoders import PointSetEncoder

SMALL = dict(d_model=64, depth=2, heads=4)
B, N, E, F = 3, 128, 4, 10


def randomise(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return model


def make_inputs(seed=0, n_objects=2):
    """A random batch shaped like the dataset's output: start + end known, n_objects present."""
    g = torch.Generator().manual_seed(seed)
    ent_mask = torch.tensor([True] * (1 + n_objects) + [False] * (E - 1 - n_objects)).expand(B, E).clone()
    known = torch.zeros(B, N, E, dtype=torch.bool)
    known[:, 0] = known[:, -1] = ent_mask
    x_known = torch.randn(B, N, E, F, generator=g) * known[..., None]
    x_t = torch.where(known[..., None], x_known, torch.randn(B, N, E, F, generator=g)) * ent_mask[:, None, :, None]
    shapes = torch.stack([box_shape_points(rgb=(1.0, 0, 0), seed=k) for k in range(E - 1)]).expand(B, -1, -1, -1)
    return dict(x_t=x_t, dur_t=torch.randn(B, generator=g), tau=torch.rand(B, generator=g), x_known=x_known,
                known=known, ent_mask=ent_mask, shapes=shapes.clone())


@torch.no_grad()
def run(model, **inp):
    return model(**inp)


def test_output_shapes_and_zero_at_initialisation():
    model = MotionDiT(**SMALL)
    v, v_dur = run(model, **make_inputs())
    assert v.shape == (B, N, E, F) and v_dur.shape == (B,)
    assert (v == 0).all() and (v_dur == 0).all()  # adaLN-Zero: the untrained network predicts exactly 0


def test_default_size_is_about_30M_parameters():
    n = sum(p.numel() for p in MotionDiT().parameters())
    assert 28e6 < n < 32e6, n  # CLAUDE.md §8.1: about 30 M


def test_padded_slots_do_not_affect_real_tokens():
    """Anything written into a padded entity slot (slot 3 for StackCube) must not change real tokens' outputs."""
    model = randomise(MotionDiT(**SMALL))
    inp = make_inputs(n_objects=2)
    v, v_dur = run(model, **inp)
    inp["x_t"][:, :, 3] = torch.randn(B, N, F) * 5
    inp["x_known"][:, :, 3] = torch.randn(B, N, F) * 5
    inp["shapes"][:, 2] = torch.randn(512, 6)  # the absent object's shape
    v2, v_dur2 = run(model, **inp)
    torch.testing.assert_close(v2[:, :, :3], v[:, :, :3], atol=1e-5, rtol=0)
    torch.testing.assert_close(v_dur2, v_dur, atol=1e-5, rtol=0)


def test_every_input_is_used():
    """Changing any one input changes the prediction for the middle of the motion (nothing is silently ignored).
    Middle = steps 1..126, which are never given, so information must travel there through attention."""
    model = randomise(MotionDiT(**SMALL))
    inp = make_inputs()
    v, _ = run(model, **inp)
    changes = {
        "x_t at another step": lambda i: i["x_t"][:, 5].add_(1.0),
        "given end value": lambda i: i["x_known"][:, -1, 1].add_(1.0),
        "known flags": lambda i: i["known"][:, -1, 0].logical_not_(),
        "duration": lambda i: i["dur_t"].add_(1.0),
        "noise level": lambda i: i["tau"].mul_(0.5),
        "object shape": lambda i: i["shapes"][:, 0, :, :3].mul_(2.0),
    }
    for name, change in changes.items():
        inp2 = {k: t.clone() for k, t in inp.items()}
        change(inp2)
        v2, _ = run(model, **inp2)
        assert (v2[:, 60, :3] - v[:, 60, :3]).abs().max() > 1e-4, f"{name} does not reach step 60"


def test_known_flag_is_seen_on_its_own():
    """A cell given with value 0 must look different from a cell that is not given. Only the known flag tells
    them apart (x_known is 0 in both), so this fails if the flag is not fed to the network."""
    model = randomise(MotionDiT(**SMALL))
    inp = make_inputs()
    inp["x_known"][:, -1, 0] = 0.0  # the robot's end is "given" with all values 0 ...
    v, _ = run(model, **inp)
    inp["known"][:, -1, 0] = False  # ... versus not given: identical numbers, only the flag differs
    v2, _ = run(model, **inp)
    assert (v2[:, 60, :3] - v[:, 60, :3]).abs().max() > 1e-4


def test_samples_in_a_batch_do_not_mix():
    """Each sample's output depends only on that sample (catches reshape / broadcasting mistakes)."""
    model = randomise(MotionDiT(**SMALL))
    inp = make_inputs()
    v, v_dur = run(model, **inp)
    inp["x_t"][1:] += 1.0
    inp["tau"][1:] *= 0.3
    v2, v_dur2 = run(model, **inp)
    torch.testing.assert_close(v2[0], v[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(v_dur2[0], v_dur[0], atol=1e-5, rtol=0)
    assert (v2[1] - v[1]).abs().max() > 1e-4


def test_slot_and_step_identity_matter():
    """Swapping two steps' inputs must not just swap the outputs: the network knows which step and which entity
    each token is (step and entity embeddings are applied). Without them attention would be order-blind."""
    model = randomise(MotionDiT(**SMALL))
    inp = make_inputs()
    v, _ = run(model, **inp)
    inp2 = {k: t.clone() for k, t in inp.items()}
    inp2["x_t"][:, [40, 80]] = inp["x_t"][:, [80, 40]]
    v2, _ = run(model, **inp2)
    assert (v2[:, 40] - v[:, 80]).abs().max() > 1e-4


def test_all_parameters_receive_gradient():
    """One backward pass reaches every parameter: no module is built but left unused."""
    model = randomise(MotionDiT(**SMALL))
    v, v_dur = model(**make_inputs())
    (v.square().mean() + v_dur.square().mean()).backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, dead


def test_point_encoder_ignores_point_order():
    enc = randomise(PointSetEncoder(32))
    pts = box_shape_points(n=512)
    with torch.no_grad():
        a, b = enc(pts), enc(pts[torch.randperm(512)])
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_gpu():
    dev = "cuda" if torch.cuda.is_available() else "mps"
    model = randomise(MotionDiT(**SMALL)).to(dev)
    inp = {k: t.to(dev) for k, t in make_inputs().items()}
    v, _ = run(model, **inp)
    v_cpu, _ = run(model.cpu(), **make_inputs())
    torch.testing.assert_close(v.cpu(), v_cpu, atol=1e-3, rtol=1e-3)
