"""B4: articulated multi-configuration reconstruction (CLAUDE.md §7.2).

One canonical Gaussian set per part, optimised across every training configuration: for each sampled view, pose all
parts with that configuration's part_T, render, and compare with the capture (RGB L1 + SSIM, depth, part labels,
foreground mask). Robot links, objects and the near background become Gaussians; the ground beyond `ground_radius`
(it reaches the horizon) and the sky are an `EnvMap` behind them.

Input = the B2 captures dict (`sim/capture.py::load_captures`): cameras[uid] = {rgb [R,H,W,3] u8, depth_mm, seg
[R,H,W] (part index, -1 none), viewmat [R,4,4], K, valid [R]}, part_T [R,P,4,4], part_names/kinds, split [R].
This module imports no ManiSkill code (it runs in a torch + gsplat environment).
"""
import dataclasses
import math
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

from gaussbridge import geometry as G
from gaussbridge.gs.background import EnvMap
from gaussbridge.gs.render import render
from gaussbridge.gs.splats import init_from_points

IGNORE = -1  # = sim.capture.IGNORE: pixel belongs to no part (sky), or to ground modelled by the EnvMap


@dataclasses.dataclass
class ReconConfig:
    iters: int = 30_000
    color_mode: str = "lit"
    sh_degree: int = 1
    # init (CLAUDE.md §7.2)
    max_points: int = 20_000           # per part
    max_pixels_per_view_part: int = 512  # back-projected candidates per (view, part) before the per-part subsample
    init_opacity: float = 0.1
    scale_clamp: tuple = (1e-4, 0.05)
    # ground within this horizontal distance of ground_center (m) is Gaussians; beyond it, the EnvMap
    ground_radius: float = 2.5
    ground_center: tuple = (-0.15, 0.0)
    # view sampling: policy cameras (base/hand) with this probability, capture rings otherwise
    p_policy: float = 0.25
    # loss = (1 - l_ssim) L1 + l_ssim (1 - SSIM) + l_depth |D - D_gt| + l_label CE + l_mask |alpha - fg|
    lambda_ssim: float = 0.2
    lambda_depth: float = 1.0
    lambda_label: float = 0.05
    lambda_mask: float = 0.1
    # learning rates (gsplat simple_trainer defaults at scene_scale 1; means decay to lr_means_final x)
    lr_means: float = 1.6e-4
    lr_means_final: float = 0.01
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    lr_albedo: float = 1e-2            # logit space
    lr_normals: float = 1e-3
    lr_env: float = 1e-2
    lr_lights: float = 0.0             # 0 = keep ManiSkill's exact light values (ambient 0.3, 1/pi)
    # densification (gsplat DefaultStrategy)
    densify: bool = True
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 2e-4
    # optional per-(config, part) se(3) correction; configs in pose_fixed stay as captured
    refine_poses: bool = False
    lr_pose: float = 1e-4
    pose_fixed: tuple = (0,)
    backend: str = "auto"
    seed: int = 0
    log_every: int = 500


# ---------------------------------------------------------------- views
@dataclasses.dataclass
class View:
    config: int
    camera: str
    policy: bool
    rgb: torch.Tensor      # [H,W,3] u8
    depth: torch.Tensor    # [H,W] m, 0 = invalid
    label: torch.Tensor    # [H,W] long part index; IGNORE = sky or far ground (EnvMap)
    viewmat: torch.Tensor  # [4,4]
    K: torch.Tensor        # [3,3]

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]  # (W, H)


def ring_names(cap):
    return {ring["name"] for ring in cap.get("meta", {}).get("rings", [])}


def is_policy_camera(cap, uid):
    """Policy cameras are the capture-env cameras that are not part of a capture ring (`<ring>_e.._a..`)."""
    return uid.split("_")[0] not in ring_names(cap)


def prepare_views(cap, configs, device, ground_radius=2.5, ground_center=(-0.15, 0.0), cameras=None):
    """Valid views of the given configurations on `device`. Ground pixels farther than ground_radius (horizontal,
    from ground_center) are relabelled IGNORE: the EnvMap renders them, so they are background, not Gaussians.
    Defining 'far' in the world (not by camera depth) keeps it consistent across cameras."""
    names = cap["part_names"]
    ground = names.index("ground") if "ground" in names else None
    center = torch.tensor(ground_center, dtype=torch.float32, device=device)
    views = []
    for uid, cam in cap["cameras"].items():
        if cameras is not None and uid not in cameras:
            continue
        K = cam["K"].float().to(device)
        for r in configs:
            if not bool(cam["valid"][r]):
                continue
            depth = cam["depth_mm"][r].to(device).float() / 1000.0
            label = cam["seg"][r].to(device).long()
            viewmat = cam["viewmat"][r].float().to(device)
            if ground is not None:
                rows, cols = (label == ground).nonzero(as_tuple=True)
                if len(rows):
                    xw = G.backproject(depth, viewmat, K, rows, cols)
                    far = ((xw[:, :2] - center).norm(dim=-1) > ground_radius) | (depth[rows, cols] <= 0)
                    label[rows[far], cols[far]] = IGNORE
            views.append(View(r, uid, is_policy_camera(cap, uid), cam["rgb"][r].to(device), depth, label,
                              viewmat, K))
    return views


# ---------------------------------------------------------------- init
def depth_normals(depth, viewmat, K):
    """World-space surface normals [H,W,3] from a z-depth image by central differences, oriented towards the
    camera, and a validity mask [H,W] (all 4 neighbours have depth; the border is invalid)."""
    H, W = depth.shape
    rows, cols = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device),
                                indexing="ij")
    X = G.backproject(depth, viewmat, K, rows.reshape(-1), cols.reshape(-1)).reshape(H, W, 3)
    dx = X[1:-1, 2:] - X[1:-1, :-2]
    dy = X[2:, 1:-1] - X[:-2, 1:-1]
    n = torch.zeros_like(X)
    n[1:-1, 1:-1] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    ok = torch.zeros(H, W, dtype=torch.bool, device=depth.device)
    d = depth > 0
    ok[1:-1, 1:-1] = d[1:-1, 2:] & d[1:-1, :-2] & d[2:, 1:-1] & d[:-2, 1:-1] & d[1:-1, 1:-1]
    to_cam = G.viewmat_to_campos(viewmat) - X
    n = torch.where(((n * to_cam).sum(-1, keepdim=True) < 0), -n, n)
    ok &= n.norm(dim=-1) > 0.5
    return n, ok


def _cap_per_group(group, cap, generator):
    """Indices keeping at most `cap` random members of every group id (1-D long tensor)."""
    if len(group) == 0:
        return group
    key = group.double() * 2 + torch.rand(len(group), generator=generator, dtype=torch.float64).to(group.device)
    order = key.argsort()
    g = group[order]
    _, counts = torch.unique_consecutive(g, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(g), device=g.device) - torch.repeat_interleave(starts, counts)
    return order[rank < cap]


def gather_points(views, part_T, max_pixels_per_view_part=512, max_points=20_000, seed=0):
    """Back-project labelled pixels of every view into their part's canonical frame.
    Returns canonical points [M,3], colours [M,3], canonical normals [M,3], part rotation at capture [M,3,3],
    part ids [M]. Pixels without a valid depth normal get the direction to the camera as normal."""
    gen = torch.Generator().manual_seed(seed)
    pts, cols, nrm, Rs, pids = [], [], [], [], []
    for v in views:
        ok = (v.label != IGNORE) & (v.depth > 0)
        rows, colsi = ok.nonzero(as_tuple=True)
        pid = v.label[rows, colsi]
        keep = _cap_per_group(pid, max_pixels_per_view_part, gen)
        rows, colsi, pid = rows[keep], colsi[keep], pid[keep]
        xw = G.backproject(v.depth, v.viewmat, v.K, rows, colsi)
        n_img, n_ok = depth_normals(v.depth, v.viewmat, v.K)
        nw = n_img[rows, colsi]
        to_cam = F.normalize(G.viewmat_to_campos(v.viewmat) - xw, dim=-1)
        nw = torch.where(n_ok[rows, colsi][:, None], nw, to_cam)
        T = part_T[v.config][pid]  # [m,4,4]
        R, t = T[:, :3, :3], T[:, :3, 3]
        pts.append(torch.einsum("mji,mj->mi", R, xw - t))  # R^T (x - t)
        nrm.append(torch.einsum("mji,mj->mi", R, nw))
        cols.append(v.rgb[rows, colsi].float() / 255.0)
        Rs.append(R)
        pids.append(pid)
    pts, cols, nrm, Rs, pids = (torch.cat(x) for x in (pts, cols, nrm, Rs, pids))
    keep = _cap_per_group(pids, max_points, gen)
    return pts[keep], cols[keep], nrm[keep], Rs[keep], pids[keep]


def init_splats(views, cap, cfg, device):
    part_T = cap["part_T"].float().to(device)
    pts, cols, nrm, Rs, pids = gather_points(views, part_T, cfg.max_pixels_per_view_part, cfg.max_points, cfg.seed)
    meta = {"task": cap.get("meta", {}).get("task"), "source": "B4 recon"}
    return init_from_points(pts, cols, pids, cap["part_names"], cap["part_kinds"], sh_degree=cfg.sh_degree,
                            color_mode=cfg.color_mode, normals=nrm if cfg.color_mode == "lit" else None,
                            point_R=Rs, init_opacity=cfg.init_opacity, scale_clamp=cfg.scale_clamp, meta=meta)


def background_colour(views):
    """Mean colour of IGNORE pixels (far ground + sky): the EnvMap's initial colour."""
    acc, n = torch.zeros(3), 0
    for v in views:
        m = v.label == IGNORE
        if m.any():
            acc += v.rgb[m].float().sum(0).cpu() / 255.0
            n += int(m.sum())
    return (acc / max(n, 1)).tolist()


# ---------------------------------------------------------------- losses and metrics
def _gauss_window(size=11, sigma=1.5, device=None):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None]).expand(3, 1, size, size).contiguous()


def ssim(a, b, size=11):
    """Mean SSIM of images [H,W,3] in [0,1] (Gaussian window 11, sigma 1.5, as 3DGS)."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    w = _gauss_window(size, device=a.device).to(a.dtype)
    pad = size // 2
    mu_x, mu_y = F.conv2d(x, w, padding=pad, groups=3), F.conv2d(y, w, padding=pad, groups=3)
    sxx = F.conv2d(x * x, w, padding=pad, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12)))


def recon_losses(out, view, cfg):
    """Loss terms for one rendered view (render() output with part one-hot features and the EnvMap)."""
    rgb = out["rgb"][0]
    gt = view.rgb.float() / 255.0
    alpha = out["alpha"][0, ..., 0]
    fg = view.label != IGNORE
    terms = {"l1": (rgb - gt).abs().mean(), "ssim": 1 - ssim(rgb, gt)}
    dmask = fg & (view.depth > 0)
    terms["depth"] = (out["depth"][0, ..., 0] - view.depth)[dmask].abs().mean() if dmask.any() else rgb.new_zeros(())
    terms["mask"] = (alpha - fg.float()).abs().mean()
    if fg.any():
        p = out["features"][0].gather(-1, view.label.clamp_min(0)[..., None])[..., 0] / alpha.clamp_min(1e-6)
        terms["label"] = -torch.log(p.clamp(1e-6, 1.0))[fg].mean()
    else:
        terms["label"] = rgb.new_zeros(())
    total = ((1 - cfg.lambda_ssim) * terms["l1"] + cfg.lambda_ssim * terms["ssim"] + cfg.lambda_depth * terms["depth"]
             + cfg.lambda_label * terms["label"] + cfg.lambda_mask * terms["mask"])
    return total, terms


# ---------------------------------------------------------------- pose refinement
def _skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z],
                       -1).unflatten(-1, (3, 3))


class PoseRefiner(nn.Module):
    """Learnable correction per (configuration, part): R' = R · exp([w]x), t' = t + v (CLAUDE.md §7.2, for real
    captures whose poses are estimated). Configurations in `fixed` are never changed (they anchor the frame)."""

    def __init__(self, n_configs, n_parts, fixed=(0,)):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(n_configs, n_parts, 6))
        free = torch.ones(n_configs, 1, 1)
        free[list(fixed)] = 0
        self.register_buffer("free", free)

    def forward(self, r, part_T):
        d = self.delta[r] * self.free[r]
        R = part_T[:, :3, :3] @ torch.linalg.matrix_exp(_skew(d[:, :3]))
        t = part_T[:, :3, 3] + d[:, 3:]
        top = torch.cat([R, t[:, :, None]], -1)
        return torch.cat([top, part_T[:, 3:]], 1)


# ---------------------------------------------------------------- training
def make_optimizers(splats, cfg):
    lrs = {"means": cfg.lr_means, "scales": cfg.lr_scales, "quats": cfg.lr_quats, "opacities": cfg.lr_opacities,
           "sh0": cfg.lr_sh0, "shN": cfg.lr_shN, "albedo": cfg.lr_albedo, "normals": cfg.lr_normals}
    return {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15) for k, p in splats.params.items() if p.requires_grad}


def reconstruct(cap, cfg=None, device="cuda", log=print):
    """Optimise PartSplats + EnvMap on the training configurations of `cap`. Returns (splats, env, info) with
    info = {"history": [...], "pose_refiner": PoseRefiner or None, "seconds": float, "num_views": int}."""
    cfg = cfg or ReconConfig()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    train = [r for r, s in enumerate(cap["split"]) if s == "train"]
    views = prepare_views(cap, train, device, cfg.ground_radius, cfg.ground_center)
    if not views:
        raise ValueError("no valid training views")
    splats = init_splats(views, cap, cfg, device)
    env = EnvMap(init_rgb=background_colour(views)).to(device)
    part_T_all = cap["part_T"].float().to(device)
    log(f"init: {len(splats)} Gaussians from {len(views)} views of configs {train}; {splats.counts()}")

    opts = make_optimizers(splats, cfg)
    extra = [torch.optim.Adam(env.parameters(), lr=cfg.lr_env)]
    if cfg.lr_lights > 0:
        extra.append(torch.optim.Adam(splats.lights.parameters(), lr=cfg.lr_lights))
    else:
        splats.lights.requires_grad_(False)
    refiner = None
    if cfg.refine_poses:
        refiner = PoseRefiner(len(cap["split"]), splats.num_parts, cfg.pose_fixed).to(device)
        extra.append(torch.optim.Adam(refiner.parameters(), lr=cfg.lr_pose))
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=cfg.lr_means_final ** (1.0 / cfg.iters))

    strategy = sstate = None
    if cfg.densify:
        from gsplat.strategy import DefaultStrategy
        strategy = DefaultStrategy(refine_start_iter=cfg.refine_start, refine_stop_iter=cfg.refine_stop,
                                   refine_every=cfg.refine_every, reset_every=cfg.reset_every,
                                   grow_grad2d=cfg.grow_grad2d)
        strategy.check_sanity(splats.params, opts)
        sstate = strategy.initialize_state(scene_scale=1.0)

    policy = [i for i, v in enumerate(views) if v.policy]
    rings = [i for i, v in enumerate(views) if not v.policy] or policy
    history, t0, acc = [], time.time(), {}
    for step in range(cfg.iters):
        pool = policy if (policy and rng.random() < cfg.p_policy) else rings
        v = views[rng.choice(pool)]
        part_T = part_T_all[v.config]
        if refiner is not None:
            part_T = refiner(v.config, part_T)
        W, H = v.size
        out = render(splats, part_T, v.viewmat[None], v.K[None], W, H, features=splats.part_onehot(), env=env,
                     backend=cfg.backend)
        if strategy is not None:
            if out["meta"].get("backend") != "gsplat":
                raise RuntimeError("densification needs the gsplat backend (CUDA); set densify=False otherwise")
            strategy.step_pre_backward(splats.params, opts, sstate, step, out["meta"])
        loss, terms = recon_losses(out, v, cfg)
        loss.backward()
        for o in list(opts.values()) + extra:
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if strategy is not None:
            strategy.step_post_backward(splats.params, opts, sstate, step, out["meta"], packed=False)
        for k, val in {"loss": loss, **terms}.items():
            acc[k] = acc.get(k, 0.0) + float(val)
        acc["psnr"] = acc.get("psnr", 0.0) + psnr(out["rgb"][0].detach().clamp(0, 1), v.rgb.float() / 255.0)
        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.iters:
            n = cfg.log_every if (step + 1) % cfg.log_every == 0 else (step + 1) % cfg.log_every
            row = {"step": step + 1, "num_gaussians": len(splats), "seconds": round(time.time() - t0, 1),
                   **{k: round(a / n, 5) for k, a in acc.items()}}
            history.append(row)
            log(row)
            acc = {}
    info = {"history": history, "pose_refiner": refiner, "seconds": time.time() - t0, "num_views": len(views),
            "train_configs": train}
    return splats, env, info


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def render_view(splats, env, part_T, viewmat, K, width, height, backend="auto"):
    """One camera: rgb [H,W,3] in [0,1], alpha [H,W], depth [H,W], part label [H,W] (IGNORE where alpha < 0.5)."""
    out = render(splats, part_T, viewmat[None], K[None], width, height, features=splats.part_onehot(), env=env,
                 backend=backend)
    alpha = out["alpha"][0, ..., 0]
    label = torch.where(alpha > 0.5, out["features"][0].argmax(-1), torch.full_like(alpha, IGNORE, dtype=torch.long))
    return {"rgb": out["rgb"][0].clamp(0, 1), "alpha": alpha, "depth": out["depth"][0, ..., 0], "label": label}


@torch.no_grad()
def evaluate(splats, env, cap, configs, device, cfg=None, cameras=None, backend="auto"):
    """Per-view metrics on the given configurations: PSNR, SSIM (full image, EnvMap included), mask IoU (rendered
    alpha > 0.5 vs labelled pixels) and label accuracy (labelled pixels whose rendered label is correct; pixels the
    render leaves empty count as wrong)."""
    cfg = cfg or ReconConfig()
    views = prepare_views(cap, configs, device, cfg.ground_radius, cfg.ground_center, cameras)
    part_T_all = cap["part_T"].float().to(device)
    rows = []
    for v in views:
        W, H = v.size
        o = render_view(splats, env, part_T_all[v.config], v.viewmat, v.K, W, H, backend)
        gt = v.rgb.float() / 255.0
        fg = v.label != IGNORE
        fr = o["label"] != IGNORE
        rows.append({"config": v.config, "camera": v.camera, "policy": v.policy, "size": H,
                     "psnr": psnr(o["rgb"], gt), "ssim": float(ssim(o["rgb"], gt)),
                     "mask_iou": float((fg & fr).sum() / (fg | fr).sum().clamp_min(1)),
                     "label_acc": float((o["label"][fg] == v.label[fg]).float().mean()) if fg.any() else math.nan})
    return rows


def summarize(rows):
    """Mean metrics per camera group: each policy camera by name, and 'rings' (capture-resolution ring views)."""
    groups = {}
    for r in rows:
        groups.setdefault(r["camera"] if r["policy"] else "rings", []).append(r)
    out = {}
    for g, rs in groups.items():
        out[g] = {k: sum(r[k] for r in rs) / len(rs) for k in ("psnr", "ssim", "mask_iou", "label_acc")}
        out[g]["min_psnr"] = min(r["psnr"] for r in rs)
        out[g]["views"] = len(rs)
    return out
