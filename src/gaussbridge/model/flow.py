"""C3: flow matching for the motion model: the training loss and the sampler (CLAUDE.md §8.2, §8.3).

The idea in one paragraph
-------------------------
Draw a straight line from a real motion x0 (at tau = 0) to pure random noise eps (at tau = 1):

        x_tau = (1 - tau) * x0 + tau * eps          for tau in [0, 1]

Moving along that line towards the noise has constant velocity  v = d x_tau / d tau = eps - x0.
TRAINING teaches the network to predict v from the point x_tau (it never sees x0 or eps directly).
SAMPLING starts at pure noise (tau = 1) and walks backwards along the predicted velocity in small steps,
        x  <-  x - dtau * v_predicted,
until tau = 0, where it arrives at a motion that looks like the training data. Different starting noise gives a
different motion: that is where the diversity comes from.

Goal conditioning by inpainting
-------------------------------
Cells that are given (start row, and some or all of the end row) are never noised: in x_tau they always hold their
given value x_known, during training AND at every sampling step. The loss ignores them (there is nothing to
predict). So samples hit the given start and end exactly, and the network learns to fill in the rest consistently.

Cells that never count
----------------------
Padded entity slots (e.g. slot 3 in StackCube) and the robot token's last feature (always 0, a spacer, see
trajectory.episode_to_motion) are held at 0 in every x_tau and ignored by the loss.

The duration
------------
The duration (normalised log seconds) is one more number generated the same way, with its own velocity. It is
never given. Its loss is weighted by 0.1 (§8.2) so it does not dominate the 5000 motion numbers.
"""
import torch

ROBOT_PAD_FEATURE = 9   # robot token feature 9 is always 0 (trajectory.episode_to_motion)
DUR_LOSS_WEIGHT = 0.1   # CLAUDE.md §8.2


# ---------------------------------------------------------------------------------------------------- shared
def sample_tau(batch, generator=None, device="cpu"):
    """Noise levels for training: logit-normal, tau = sigmoid(N(0, 1)) (Esser et al. 2024, SD3).
    It puts most samples at medium noise, where predicting the velocity is hardest and most useful."""
    return torch.sigmoid(torch.randn(batch, generator=generator, device=device))


def impose(x, x_known, known, ent_mask):
    """Put the given values into their cells and zero everything that is not real: padded entity slots and the
    robot spacer feature. Applied to every x_tau in training and after every sampling step, so the network sees
    the same thing in both (a spacer left noisy would be pure noise in training but drift in sampling)."""
    x = torch.where(known[..., None], x_known, x) * ent_mask[:, None, :, None]
    x[:, :, 0, ROBOT_PAD_FEATURE] = 0
    return x


def loss_mask(known, ent_mask, n_features):
    """[B, N, E, F] bool: the cells whose velocity is learned = not given, real entity, real feature."""
    mask = (~known & ent_mask[:, None, :])[..., None].expand(*known.shape, n_features).clone()
    mask[:, :, 0, ROBOT_PAD_FEATURE] = False
    return mask


def model_inputs(batch):
    """The conditioning part of a batch (what the network may see besides the noisy motion)."""
    return {k: batch[k] for k in ("x_known", "known", "ent_mask", "shapes")}


# ---------------------------------------------------------------------------------------------------- training
def noisy_sample(batch, tau, eps, eps_dur):
    """Point x_tau on the line from the data to the noise, with the given cells imposed. Returns (x_t, dur_t)."""
    t = tau[:, None, None, None]
    x_t = impose((1 - t) * batch["x0"] + t * eps, batch["x_known"], batch["known"], batch["ent_mask"])
    dur_t = (1 - tau) * batch["log_dur"] + tau * eps_dur
    return x_t, dur_t


def flow_loss(model, batch, generator=None):
    """Flow-matching loss on one batch (a dict as produced by MotionDataset + DataLoader).

    Returns (loss, stats): loss = mean squared velocity error over the learned cells + 0.1 * duration error.
    stats (floats, for logging) splits the motion error into robot and objects."""
    x0, dur0 = batch["x0"], batch["log_dur"].to(batch["x0"].dtype)
    B = x0.shape[0]
    tau = sample_tau(B, generator, x0.device)
    eps = torch.randn(x0.shape, generator=generator, device=x0.device)
    eps_dur = torch.randn(B, generator=generator, device=x0.device)
    x_t, dur_t = noisy_sample({**batch, "log_dur": dur0}, tau, eps, eps_dur)

    v, v_dur = model(x_t, dur_t, tau, **model_inputs(batch))

    mask = loss_mask(batch["known"], batch["ent_mask"], x0.shape[-1])
    err = (v.float() - (eps - x0)).square() * mask
    motion = err.sum() / mask.sum().clamp_min(1)
    duration = (v_dur.float() - (eps_dur - dur0)).square().mean()
    loss = motion + DUR_LOSS_WEIGHT * duration
    stats = {"loss": loss.item(), "motion": motion.item(), "duration": duration.item(),
             "robot": (err[:, :, 0].sum() / mask[:, :, 0].sum().clamp_min(1)).item(),
             "objects": (err[:, :, 1:].sum() / mask[:, :, 1:].sum().clamp_min(1)).item()}
    return loss, stats


# ---------------------------------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(model, x_known, known, ent_mask, shapes, steps=40, generator=None, guidance=None, guide_scale=1.0,
           guide_tau=None, trace=None):
    """Generate motions for a batch of conditions by Euler integration from tau = 1 (noise) to tau = 0 (data).

    Args: the conditioning tensors as in a batch ([B, ...]); steps = number of Euler steps (§8.3: 40).
    Returns (x [B, N, E, F], log_dur [B]): the given cells equal x_known exactly, padded slots are 0.
    To draw S different motions for one condition, repeat it S times along the batch dimension.

    Guidance (C6, guidance/guide.py): guidance = an energy callable x -> [B] (e.g. guide.PhysicsEnergy) and
    guide_scale = lambda. While tau < guide_tau (default 0.6), each step also computes the predicted clean motion
    x0_hat = x - tau * v and moves x against the energy gradient at x0_hat (Polyak step, no backprop through the
    network). The given cells stay exact (re-imposed after the step). guidance=None or guide_scale=0 gives exactly
    the unguided sampler. trace: optional list; each guided step appends {"step", "tau", "energy" [B]}."""
    B, device = x_known.shape[0], x_known.device
    x = impose(torch.randn(x_known.shape, generator=generator, device=device), x_known, known, ent_mask)
    dur = torch.randn(B, generator=generator, device=device)
    taus = torch.linspace(1.0, 0.0, steps + 1, device=device)
    cond = {"x_known": x_known, "known": known, "ent_mask": ent_mask, "shapes": shapes}
    guided = guidance is not None and guide_scale != 0
    if guided:
        from gaussbridge.guidance import guide  # lazy: guidance pulls in FK, which plain sampling does not need
        guide_tau = guide.GUIDE_TAU if guide_tau is None else guide_tau
        free = guide.guidable_mask(loss_mask(known, ent_mask, x_known.shape[-1]))
    for i in range(steps):
        tau, dtau = taus[i], taus[i + 1] - taus[i]  # dtau < 0: walking from noise back to data
        v, v_dur = model(x, dur, tau.expand(B), **cond)
        x_next = x + dtau * v.to(x.dtype)
        if guided and tau < guide_tau:
            delta, energy = guide.guidance_step(x - tau * v.to(x.dtype), guidance, guide_scale, free)
            x_next = x_next - delta
            if trace is not None:
                trace.append({"step": i, "tau": tau.item(), "energy": energy})
        x = impose(x_next, x_known, known, ent_mask)
        dur = dur + dtau * v_dur.to(dur.dtype)
    return x, dur
