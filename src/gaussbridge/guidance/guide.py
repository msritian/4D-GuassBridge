"""C6: physics guidance during sampling (CLAUDE.md §8.3), the hook that model/flow.sample calls.

What happens at one sampling step (only while tau < GUIDE_TAU = 0.6)
--------------------------------------------------------------------
The network gives the velocity v at the current noisy motion x_tau. Following the line back to tau = 0 gives the
network's current guess of the finished motion,

        x0_hat = x_tau - tau * v                                   (exact on the straight noise-to-data line)

E(x0_hat) = collision energy + attachment energy (guidance/collision.py, guidance/attach.py) is measured on that
guess, and x_tau is moved against its gradient. The gradient is taken with respect to x0_hat only, never through the
network (DPS-style, cheap: one FK + geometry pass per step, no backprop through the 30 M parameters). Below
tau = 0.6 the guess is good enough for its physics to mean something; above it x0_hat is mostly the data average.

Step size: lambda is a fraction of the violation, not a raw learning rate
-------------------------------------------------------------------------
The energies are sums of squared residuals in metres (E = sum r_i^2), with gradients of very different sizes per
cell. A fixed learning rate would need tuning per task and per energy. Instead the step is the Polyak step

        delta = lambda * E / |g|^2 * g ,      g = dE/dx0_hat  (on the cells guidance may change)

To first order this changes every residual r_i by the same fraction -lambda/2 (for violations whose gradients do not
overlap): lambda = 1 halves the violation in one step, lambda = 2 removes it, lambda = 0 is no guidance. Each energy
term (collision, attachment) gets its OWN Polyak step: with one shared scale, a large collision would also multiply
the small attachment gradient of an otherwise fine grasp and move it (seen in the tests). That makes
the §8.3 sweep lambda in {0, 0.25, 0.5, 1, 2} meaningful without knowing the energies' scale. Applied over the ~24
steps below tau = 0.6, even lambda = 0.25 compounds. Each step is capped at MAX_CELL_STEP per grid cell
(0.05 in normalised units = 1.5 cm object position, ~0.07 rad arm joint) so a nearly flat gradient (|g| ~ 0 with
E > 0) can never fling the motion.

What guidance can and cannot fix (measured on planner motions, tests/test_guidance_guide.py)
-------------------------------------------------------------------------------------------
* Penetration (a cube in the table, a link in a cube): local and nearly linear, one lambda = 2 step removes it.
* Grasp slip: the attachment residuals are step-to-step differences r_{n+1} - r_n, so neighbouring residuals share
  cells (a chain). Step-to-step jitter of the object in the hand is removed quickly; a slow drift over the whole
  carry is low-frequency and shrinks only slowly. That is what project_attachment, after sampling, fixes exactly.

What guidance may change
------------------------
Only cells the model generates (not given, real entity, not the robot spacer, flow.loss_mask), and never the
objects' "attached" flags: lowering a flag lowers the attachment energy's weight, so the gradient would "repair"
a slipping grasp by declaring the object not held. The flags stay the model's decision; project_attachment
(after sampling) makes the poses consistent with them.
"""
import torch

from gaussbridge.guidance.attach import attach_energy
from gaussbridge.guidance.collision import SceneSpheres

GUIDE_TAU = 0.6          # guidance acts for tau below this (§8.3)
MAX_CELL_STEP = 0.05     # largest change of one grid cell per guidance step (normalised units)
ATTACHED_FEATURE = 9     # object token feature: attached flag (trajectory.episode_to_motion)


class PhysicsEnergy:
    """E(x) [B] = w_collision * collision energy + w_attach * attachment energy, for motion grids x [B, N, E, F].
    Units: m^2 (both are sums of squared metres). Differentiable in x."""

    def __init__(self, fk, n_objects, w_collision=1.0, w_attach=1.0, device="cpu", scene=None):
        self.fk, self.n_objects = fk, n_objects
        self.w_collision, self.w_attach = w_collision, w_attach
        self.scene = scene or SceneSpheres.for_robot(fk, n_objects, device=device)

    def terms(self, x):
        """{"collision": [B], "attach": [B]} (unweighted)."""
        return {"collision": self.scene.energy(x), "attach": attach_energy(x, self.n_objects, self.fk)}

    def __call__(self, x):
        """{"collision": [B], "attach": [B]}, weighted. guidance_step takes a separate Polyak step per term."""
        t = self.terms(x)
        return {"collision": self.w_collision * t["collision"], "attach": self.w_attach * t["attach"]}

    def total(self, x):
        """[B] weighted sum of the terms."""
        return sum(self(x).values())


def guidable_mask(free):
    """free [B, N, E, F] bool (flow.loss_mask) minus the objects' attached flags -> the cells guidance may move."""
    free = free.clone()
    free[:, :, 1:, ATTACHED_FEATURE] = False
    return free


def guidance_step(x0_hat, energy, lam, free, max_cell_step=MAX_CELL_STEP):
    """The Polyak step (module docstring) for predicted clean motions x0_hat [B, N, E, F].

    energy: callable x -> [B], or -> {name: [B]} for several terms (each gets its own Polyak step, summed);
    lam: step fraction (0 = none, 1 = halve violations, 2 = remove them to first order);
    free: [B, N, E, F] bool, cells that may change (guidable_mask). Returns (delta [B, N, E, F] to SUBTRACT,
    total energy [B]). Samples with zero energy get delta = 0."""
    with torch.enable_grad():
        xg = x0_hat.detach().float().requires_grad_(True)
        terms = energy(xg)
        terms = terms if isinstance(terms, dict) else {"energy": terms}
        delta, total = torch.zeros_like(xg), 0
        for e in terms.values():
            (g,) = torch.autograd.grad(e.sum(), xg, retain_graph=True)
            g = torch.nan_to_num(g) * free
            e = e.detach()
            delta = delta + (lam * e / g.square().flatten(1).sum(1).clamp_min(1e-20))[:, None, None, None] * g
            total = total + e
    cap = (delta.abs().flatten(1).amax(1) / max_cell_step).clamp_min(1.0)               # >1 only if a cell is over
    return (delta / cap[:, None, None, None]).to(x0_hat.dtype), total
