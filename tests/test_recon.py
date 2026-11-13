"""B4 tests: EnvMap, view preparation, init helpers, losses, pose refinement and a small end-to-end reconstruction.
All CPU (dense rasteriser, no densification); the real-capture run is scripts/train_recon.py."""
import math

import torch

from gaussbridge import geometry as G
from gaussbridge.gs.background import EnvMap
from gaussbridge.gs.recon import (
    IGNORE,
    PoseRefiner,
    ReconConfig,
    _cap_per_group,
    depth_normals,
    evaluate,
    prepare_views,
    reconstruct,
    ssim,
    summarize,
)
from gaussbridge.gs.render import render, surface_points
from gaussbridge.gs.splats import PartSplats, init_from_points


def _cube_points(n, half=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = (torch.rand(n, 3, generator=g) * 2 - 1) * half
    ax = torch.randint(0, 3, (n,), generator=g)
    p[torch.arange(n), ax] = half * torch.sign(torch.rand(n, generator=g) - 0.5)
    return p


def _ground_points(n, extent=0.6, seed=1):
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g) * 2 - 1) * extent
    return torch.cat([xy, torch.zeros(n, 1)], -1)


def _gt_scene():
    """Ground truth: a two-tone cube ('cube', object) on a grey ground plane ('ground', background)."""
    cube = _cube_points(400)
    ground = _ground_points(900)
    rgb = torch.cat([torch.where(cube[:, 2:3] > 0, torch.tensor([0.9, 0.2, 0.1]), torch.tensor([0.2, 0.3, 0.9])),
                     torch.tensor([0.5, 0.5, 0.55]).expand(len(ground), 3)])
    pts = torch.cat([cube, ground])
    pid = torch.cat([torch.zeros(len(cube)), torch.ones(len(ground))]).long()
    s = init_from_points(pts, rgb, pid, ["cube", "ground"], ["object", "background"], init_opacity=0.95,
                         scale_clamp=(0.01, 0.03))
    return s


def _pose(yaw, t):
    T = torch.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = torch.tensor([[c, -s], [s, c]])
    T[:3, 3] = torch.tensor(t)
    return T


def _synthetic_captures(size=24):
    """B2-format captures of the GT scene: 4 configurations (cube moved/rotated; 3 train + 1 held out), a ring of 6
    cameras + one 'policy' camera, rendered with the dense rasteriser."""
    gt = _gt_scene()
    configs = [_pose(0.0, (0, 0, 0.05)), _pose(0.7, (0.1, 0.05, 0.05)), _pose(-0.5, (-0.1, 0.0, 0.12)),
               _pose(1.2, (0.05, -0.08, 0.05))]
    part_T = torch.stack([torch.stack([T, torch.eye(4)]) for T in configs])  # [R,2,4,4]
    K = G.fov_to_K(math.radians(60), size, size)
    eyes = {f"ring_e40_a{i:02d}": (0.5 * math.cos(a), 0.5 * math.sin(a), 0.35)
            for i, a in enumerate(torch.linspace(0, 2 * math.pi, 7)[:-1].tolist())}
    eyes["base_camera"] = (0.45, 0.1, 0.5)
    cams = {}
    for uid, eye in eyes.items():
        V = G.look_at_viewmat(eye, (0.0, 0.0, 0.03))
        rgb, depth, seg = [], [], []
        for r in range(len(configs)):
            with torch.no_grad():
                o = render(gt, part_T[r], V[None], K[None], size, size, background=torch.zeros(3),
                           features=gt.part_onehot(), backend="dense")
            a = o["alpha"][0, ..., 0]
            rgb.append((o["rgb"][0].clamp(0, 1) * 255).round().to(torch.uint8))
            depth.append(torch.where(a > 0.5, o["depth"][0, ..., 0] * 1000, torch.zeros_like(a)).round().short())
            seg.append(torch.where(a > 0.5, o["features"][0].argmax(-1), torch.full_like(a, -1).long()).to(torch.int8))
        cams[uid] = {"rgb": torch.stack(rgb), "depth_mm": torch.stack(depth), "seg": torch.stack(seg),
                     "viewmat": V[None].expand(len(configs), 4, 4).clone(), "K": K,
                     "valid": torch.ones(len(configs), dtype=torch.bool)}
    return {"part_names": ["cube", "ground"], "part_kinds": ["object", "background"], "part_T": part_T,
            "split": ["train", "train", "train", "heldout"], "cameras": cams,
            "meta": {"task": "Synthetic-v0", "rings": [{"name": "ring"}]}}


# ---------------------------------------------------------------- geometry + EnvMap
def test_pixel_rays_hit_pixel_centres():
    V = G.look_at_viewmat((0.4, -0.3, 0.5), (0.0, 0.1, 0.0))
    K = G.fov_to_K(math.radians(70), 20, 12)
    d = G.pixel_rays(V, K, 20, 12)
    assert d.shape == (12, 20, 3)
    assert torch.allclose(d.norm(dim=-1), torch.ones(12, 20), atol=1e-6)
    uv, z = G.project(G.viewmat_to_campos(V) + 2.0 * d.reshape(-1, 3), V, K)
    jj, ii = torch.meshgrid(torch.arange(20) + 0.5, torch.arange(12) + 0.5, indexing="xy")
    assert torch.allclose(uv, torch.stack([jj, ii], -1).reshape(-1, 2).float(), atol=1e-4)
    assert (z > 0).all()


def test_envmap_init_wrap_and_save_load(tmp_path):
    env = EnvMap(n_el=16, n_az=8, init_rgb=(0.2, 0.4, 0.6))
    dirs = torch.randn(50, 3)
    assert torch.allclose(env(dirs), torch.tensor([0.2, 0.4, 0.6]).expand(50, 3), atol=1e-5)
    with torch.no_grad():
        env.tex.normal_()
    eps = 1e-4  # just either side of azimuth +-180 deg: the texture must wrap, not jump
    a = env(torch.tensor([[math.cos(math.pi - eps), math.sin(math.pi - eps), 0.1]]))
    b = env(torch.tensor([[math.cos(-math.pi + eps), math.sin(-math.pi + eps), 0.1]]))
    assert torch.allclose(a, b, atol=1e-3)
    env(dirs).sum().backward()
    assert env.tex.grad is not None and env.tex.grad.abs().sum() > 0
    env.save(tmp_path / "bg.pt")
    assert torch.allclose(EnvMap.load(tmp_path / "bg.pt")(dirs), env(dirs))


def test_render_composites_envmap_behind_splats():
    s = _gt_scene()
    V = G.look_at_viewmat((0.5, 0.2, 0.6), (0.0, 0.0, 0.0))[None]
    K = G.fov_to_K(math.radians(60), 16, 16)[None]
    env = EnvMap(n_el=8, n_az=8)
    with torch.no_grad():
        env.tex.normal_()
        a = render(s, torch.eye(4).expand(2, 4, 4), V, K, 16, 16, env=env, backend="dense")
        b = render(s, torch.eye(4).expand(2, 4, 4), V, K, 16, 16, backend="dense")
    want = b["rgb"] + (1 - b["alpha"]) * env.render(V, K, 16, 16)
    assert torch.allclose(a["rgb"], want, atol=1e-6)


def test_depth_normals_of_a_plane_point_up():
    V = G.look_at_viewmat((0.3, 0.2, 0.5), (0.0, 0.0, 0.0))
    K = G.fov_to_K(math.radians(60), 32, 32)
    d = G.pixel_rays(V, K, 32, 32)
    c = G.viewmat_to_campos(V)
    t = -c[2] / d[..., 2]  # ray hits z = 0
    z_cam = ((c + t[..., None] * d) @ V[:3, :3].T + V[:3, 3])[..., 2]
    n, ok = depth_normals(z_cam, V, K)
    assert ok[1:-1, 1:-1].all() and not ok[0].any()
    assert torch.allclose(n[ok], torch.tensor([0.0, 0.0, 1.0]).expand(int(ok.sum()), 3), atol=1e-3)


def test_cap_per_group_limits_each_group():
    group = torch.tensor([0] * 10 + [1] * 3 + [5] * 7)
    keep = _cap_per_group(group, 4, torch.Generator().manual_seed(0))
    counts = torch.bincount(group[keep], minlength=6)
    assert counts.tolist() == [4, 3, 0, 0, 0, 4]
    assert len(set(keep.tolist())) == len(keep)


def test_ssim_identity_and_noise():
    img = torch.rand(20, 20, 3)
    assert abs(float(ssim(img, img)) - 1) < 1e-5
    assert float(ssim(img, (img + 0.3 * torch.randn_like(img)).clamp(0, 1))) < 0.9


# ---------------------------------------------------------------- views + pose refinement
def test_prepare_views_relabels_far_ground_and_flags_policy_cameras():
    cap = _synthetic_captures(size=16)
    views = prepare_views(cap, [0], "cpu", ground_radius=0.3, ground_center=(0.0, 0.0))
    assert {v.camera for v in views if v.policy} == {"base_camera"}
    for v in views:
        raw = cap["cameras"][v.camera]["seg"][0].long()
        rows, cols = (raw == 1).nonzero(as_tuple=True)
        xw = G.backproject(v.depth, v.viewmat, v.K, rows, cols)
        far = xw[:, :2].norm(dim=-1) > 0.3
        assert (v.label[rows[far], cols[far]] == IGNORE).all()
        assert (v.label[rows[~far], cols[~far]] == 1).all()
        assert torch.equal(v.label[raw == 0], raw[raw == 0])


def test_pose_refiner_identity_fixed_configs_and_recovery():
    ref = PoseRefiner(3, 2, fixed=(0,))
    T = torch.stack([_pose(0.3, (0.1, 0.0, 0.05)), torch.eye(4)])
    assert torch.allclose(ref(1, T), T, atol=1e-6)
    with torch.no_grad():
        ref.delta.normal_(0, 0.1)
    assert torch.allclose(ref(0, T), T)  # config 0 anchors the frame
    T1 = ref(1, T)
    assert torch.allclose(T1[:, :3, :3] @ T1[:, :3, :3].transpose(1, 2), torch.eye(3).expand(2, 3, 3), atol=1e-5)

    # photometric recovery: the cube's captured pose is 1.5 cm off; refine it against the true image
    s = _gt_scene()
    for p in s.parameters():
        p.requires_grad_(False)
    V = G.look_at_viewmat((0.35, 0.25, 0.4), (0.0, 0.0, 0.05))[None]
    K = G.fov_to_K(math.radians(60), 32, 32)[None]
    true_T = torch.stack([_pose(0.0, (0.0, 0.0, 0.05)), torch.eye(4)])
    with torch.no_grad():
        target = render(s, true_T, V, K, 32, 32, backend="dense")["rgb"]
    wrong_T = true_T.clone()
    wrong_T[0, :3, 3] += torch.tensor([0.015, -0.01, 0.0])
    ref = PoseRefiner(2, 2, fixed=(0,))
    opt = torch.optim.Adam(ref.parameters(), lr=2e-3)
    for _ in range(150):
        opt.zero_grad()
        loss = (render(s, ref(1, wrong_T), V, K, 32, 32, backend="dense")["rgb"] - target).abs().mean()
        loss.backward()
        opt.step()
    err = (ref(1, wrong_T)[0, :3, 3] - true_T[0, :3, 3]).norm()
    assert err < 0.004, err


# ---------------------------------------------------------------- end to end
def test_reconstruct_synthetic_scene_generalises_to_heldout_config(tmp_path):
    cap = _synthetic_captures(size=24)
    cfg = ReconConfig(iters=300, densify=False, backend="dense", max_points=250, max_pixels_per_view_part=64,
                      ground_radius=0.45, ground_center=(0.0, 0.0), log_every=100, color_mode="lit")
    logs = []
    splats, env, info = reconstruct(cap, cfg, "cpu", log=logs.append)
    assert info["train_configs"] == [0, 1, 2] and len(info["history"]) == 3
    assert info["history"][-1]["loss"] < info["history"][0]["loss"]
    summary = summarize(evaluate(splats, env, cap, [3], "cpu", cfg, backend="dense"))
    assert summary["base_camera"]["views"] == 1 and summary["rings"]["views"] == 6
    for g in ("base_camera", "rings"):
        assert summary[g]["psnr"] > 24, summary
        assert summary[g]["label_acc"] > 0.9, summary
    splats.save(tmp_path / "s.pt")
    assert PartSplats.load(tmp_path / "s.pt").counts() == splats.counts()
    assert splats.shape_points("cube").shape == (512, 6)


def test_surface_points_lie_on_the_rendered_surface():
    cube = _cube_points(3000)  # half-size 0.05, fine 2-4 mm Gaussians; top half red, bottom half blue
    rgb = torch.where(cube[:, 2:3] > 0, torch.tensor([0.9, 0.2, 0.1]), torch.tensor([0.2, 0.3, 0.9]))
    s = init_from_points(cube, rgb, torch.zeros(len(cube)).long(), ["cube"], ["object"], init_opacity=0.95,
                         scale_clamp=(0.002, 0.004))
    pts = surface_points(s, "cube", n=256, n_views=8, size=32, backend="dense")
    assert pts.shape == (256, 6)
    linf = pts[:, :3].abs().max(-1).values
    assert (linf - 0.05).abs().median() < 0.006, linf.median()
    top, bottom = pts[pts[:, 2] > 0.03], pts[pts[:, 2] < -0.03]
    assert top[:, 3].mean() > top[:, 5].mean() and bottom[:, 5].mean() > bottom[:, 3].mean()  # red up, blue down
