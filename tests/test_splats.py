"""B3 checks: PartSplats composition, colour models (local-frame SH, world-lit), io, and the rasterisers.
CPU tests use the dense PyTorch rasteriser; tests marked gpu compare it (and our SH) with gsplat on CUDA."""
import importlib.util
import math

import pytest
import torch

from gaussbridge import geometry as G
from gaussbridge.gs.raster import rasterize
from gaussbridge.gs.render import render
from gaussbridge.gs.splats import PartSplats, eval_sh, init_from_points, rgb_to_sh, sh_to_rgb

HAS_GSPLAT = importlib.util.find_spec("gsplat") is not None
needs_cuda = pytest.mark.skipif(not (HAS_GSPLAT and torch.cuda.is_available()), reason="needs gsplat + CUDA")

W = H = 40
K = G.fov_to_K(math.radians(60), W, H)


def _rot(axis, angle):
    q = torch.cat([torch.tensor([math.cos(angle / 2)]), math.sin(angle / 2) * torch.nn.functional.normalize(
        torch.tensor(axis, dtype=torch.float32), dim=0)])
    return G.quat_to_matrix(q)


def _T(R=None, t=(0.0, 0.0, 0.0)):
    T = torch.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = torch.as_tensor(t, dtype=torch.float32)
    return T


def _random_splats(n_per_part=(60, 40), color_mode="sh", sh_degree=2, seed=0):
    """Random Gaussians in a 6 cm ball per part, strongly view-dependent SH, lit fields set."""
    g = torch.Generator().manual_seed(seed)
    n = sum(n_per_part)
    pid = torch.cat([torch.full((k,), i) for i, k in enumerate(n_per_part)])
    Kc = (sh_degree + 1) ** 2
    params = {
        "means": 0.06 * torch.randn(n, 3, generator=g),
        "scales": torch.log(0.004 + 0.01 * torch.rand(n, 3, generator=g)),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(0.2 + 0.7 * torch.rand(n, generator=g)),
        "sh0": rgb_to_sh(torch.rand(n, 1, 3, generator=g)),
        "shN": 0.3 * torch.randn(n, Kc - 1, 3, generator=g),
        "part_id": pid,
        "albedo": torch.randn(n, 3, generator=g),
        "normals": torch.randn(n, 3, generator=g),
    }
    names = [f"part{i}" for i in range(len(n_per_part))]
    return PartSplats(params, names, ["object"] * len(names), sh_degree, color_mode)


def _camera(eye, target=(0.0, 0.0, 0.0)):
    return G.look_at_viewmat(torch.tensor(eye), torch.tensor(target))


# ---------------------------------------------------------------- composition
def test_world_moves_means_and_covariances_rigidly():
    s = _random_splats()
    part_T = torch.stack([_T(_rot([1, 2, 3], 0.7), (0.1, -0.2, 0.3)), _T(_rot([0, 0, 1], -2.0), (0.0, 0.5, 0.0))])
    w = s.world(part_T)
    from gaussbridge.gs.raster import quat_scale_to_covar

    for p in range(2):
        m = s.part_id == p
        R, t = part_T[p, :3, :3], part_T[p, :3, 3]
        torch.testing.assert_close(w["means"][m], s.params["means"][m] @ R.T + t)
        cov_local = quat_scale_to_covar(s.params["quats"][m], torch.exp(s.params["scales"][m]))
        torch.testing.assert_close(quat_scale_to_covar(w["quats"][m], w["scales"][m]), R @ cov_local @ R.T,
                                   atol=1e-7, rtol=1e-4)
    # identity poses leave everything in place
    w0 = s.world(torch.eye(4).expand(2, 4, 4))
    torch.testing.assert_close(w0["means"], s.params["means"])


def test_part_T_gets_gradients_for_pose_refinement():
    s = _random_splats()
    part_T = torch.stack([_T(t=(0, 0, 0)), _T(t=(0.05, 0, 0))]).requires_grad_()
    vm = _camera([0.0, -0.4, 0.2])[None]
    out = render(s, part_T, vm, K[None], W, H, backend="dense")
    out["rgb"].sum().backward()
    assert part_T.grad is not None and part_T.grad[:, :3].abs().sum() > 0


# ---------------------------------------------------------------- colour models
def test_sh_eval_matches_gsplat_reference():
    if not HAS_GSPLAT:
        pytest.skip("gsplat not installed")
    from gsplat.cuda._torch_impl import _spherical_harmonics

    g = torch.Generator().manual_seed(1)
    dirs, coeffs = torch.randn(200, 3, generator=g), torch.randn(200, 16, 3, generator=g)
    for deg in range(4):
        torch.testing.assert_close(eval_sh(deg, dirs, coeffs), _spherical_harmonics(deg, dirs, coeffs))


def test_rotating_part_and_camera_together_keeps_sh_image():
    """B3 acceptance (2): with view directions in the part frame, a rigid motion of part + camera about the part
    origin is invisible."""
    s = _random_splats(n_per_part=(120,), sh_degree=2)
    T0 = _T(_rot([0.3, -1, 0.5], 0.4), (0.05, 0.02, 0.1))
    vm0 = _camera([0.35, -0.3, 0.3], (0.05, 0.02, 0.1))
    o = T0[:3, 3]
    A = _T(t=o) @ _T(_rot([1, 1, 0.2], 1.9)) @ _T(t=-o)  # rotation about the part origin
    img0 = render(s, T0[None], vm0[None], K[None], W, H, backend="dense")
    img1 = render(s, (A @ T0)[None], (vm0 @ G.se3_inverse(A))[None], K[None], W, H, backend="dense")
    assert img0["alpha"].mean() > 0.05  # the part is in view
    torch.testing.assert_close(img1["rgb"], img0["rgb"], atol=2e-5, rtol=0)
    torch.testing.assert_close(img1["alpha"], img0["alpha"], atol=2e-5, rtol=0)
    torch.testing.assert_close(img1["depth"], img0["depth"], atol=2e-5, rtol=0)


def test_lit_colours_follow_world_lights():
    """'lit': the same rigid motion changes the image (lights are world-fixed), unless the lights move too."""
    s = _random_splats(n_per_part=(120,), color_mode="lit", sh_degree=1)
    T0 = _T(_rot([0.3, -1, 0.5], 0.4), (0.05, 0.02, 0.1))
    vm0 = _camera([0.35, -0.3, 0.3], (0.05, 0.02, 0.1))
    o = T0[:3, 3]
    Rm = _rot([1, 1, 0.2], 1.9)
    A = _T(t=o) @ _T(Rm) @ _T(t=-o)
    img0 = render(s, T0[None], vm0[None], K[None], W, H, backend="dense")["rgb"]
    img1 = render(s, (A @ T0)[None], (vm0 @ G.se3_inverse(A))[None], K[None], W, H, backend="dense")["rgb"]
    assert (img1 - img0).abs().max() > 0.05
    s.light_dirs.copy_(s.light_dirs @ Rm.T)  # rotate the lights with the scene: identical again
    img2 = render(s, (A @ T0)[None], (vm0 @ G.se3_inverse(A))[None], K[None], W, H, backend="dense")["rgb"]
    torch.testing.assert_close(img2, img0, atol=2e-5, rtol=0)


def test_lit_colour_matches_sapien_diffuse_formula():
    """One Gaussian, no SH residual: colour = (albedo (0.3 + sum_i max(0, n.l_i)/pi))^(1/2.2), lights from
    ManiSkill (travelling along [1,1,-1] and [0,0,-1])."""
    n_local = torch.nn.functional.normalize(torch.tensor([[0.2, -0.5, 1.0]]), dim=-1)
    albedo = torch.tensor([[0.6, 0.2, 0.1]])
    params = {"means": torch.zeros(1, 3), "scales": torch.zeros(1, 3), "quats": torch.tensor([[1.0, 0, 0, 0]]),
              "opacities": torch.zeros(1), "sh0": torch.zeros(1, 1, 3), "shN": torch.zeros(1, 3, 3),
              "part_id": torch.zeros(1, dtype=torch.long), "albedo": torch.logit(albedo), "normals": n_local}
    s = PartSplats(params, ["cube"], ["object"], 1, "lit")
    R = _rot([0, 0, 1], 0.8)
    w = s.world(_T(R)[None])
    got = s.colors(w, torch.tensor([[0.0, -1.0, 0.5]]))[0, 0]
    n_w = R @ n_local[0]
    lam = sum(max(0.0, float(n_w @ -torch.nn.functional.normalize(torch.tensor(d), dim=0))) / math.pi
              for d in ([1.0, 1, -1], [0.0, 0, -1]))
    torch.testing.assert_close(got, (albedo[0] * (0.3 + lam)) ** (1 / 2.2))


# ---------------------------------------------------------------- init, labels, io
def test_init_from_points_and_shape_points():
    g = torch.Generator().manual_seed(2)
    a = torch.rand(300, 3, generator=g) * 0.04 - 0.02  # a 4 cm "cube" of points
    b = torch.rand(100, 3, generator=g) * 0.2
    pts, pid = torch.cat([a, b]), torch.cat([torch.zeros(300), torch.ones(100)]).long()
    rgb = torch.cat([torch.tensor([[0.8, 0.1, 0.1]]).expand(300, 3), 0.3 + 0.6 * torch.rand(100, 3, generator=g)])
    s = init_from_points(pts, rgb, pid, ["cubeA", "table"], ["object", "background"])
    assert len(s) == 400 and s.counts() == {"cubeA": 300, "table": 100}
    d = torch.cdist(a, a).topk(4, largest=False).values[:, 1:].mean(-1)
    torch.testing.assert_close(torch.exp(s.params["scales"][:300, 0]), d.clamp(1e-4, 0.05))
    torch.testing.assert_close(torch.sigmoid(s.params["opacities"]), torch.full((400,), 0.1))
    torch.testing.assert_close(sh_to_rgb(s.params["sh0"][:, 0]), rgb)
    # shape tokens need opacity > 0.5
    with pytest.raises(ValueError):
        s.shape_points("cubeA")
    s.params["opacities"].data[:] = 3.0
    sp = s.shape_points("cubeA", n=512)
    assert sp.shape == (512, 6) and (sp[:, :3].abs() <= 0.02 + 1e-6).all()
    torch.testing.assert_close(sp[:, 3:], torch.tensor([0.8, 0.1, 0.1]).expand(512, 3))

    # lit init reproduces the captured colour under the initial lights (pose used for the init)
    # round trip: true albedo + normals -> rendered colours -> init recovers the albedo
    T = torch.stack([_T(_rot([0, 1, 0], 0.5)), _T()])
    normals = torch.nn.functional.normalize(torch.randn(400, 3, generator=g), dim=-1)
    albedo = 0.05 + 0.85 * torch.rand(400, 3, generator=g)
    names, kinds = ["cubeA", "table"], ["object", "background"]
    sl = init_from_points(pts, rgb, pid, names, kinds, color_mode="lit", normals=normals, part_T=T)
    with torch.no_grad():
        sl.params["albedo"].copy_(torch.logit(albedo))
        seen = sl.colors(sl.world(T), torch.tensor([[0.0, 0.0, 1.0]]))[0]  # no SH residual: view-independent
    reach = (seen < 1).all(-1)  # saturated colours cannot be inverted
    sl2 = init_from_points(pts, seen, pid, names, kinds, color_mode="lit", normals=normals, part_T=T)
    assert reach.float().mean() > 0.8
    torch.testing.assert_close(sl2.albedo()[reach], albedo[reach], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(sl2.colors(sl2.world(T), torch.tensor([[1.0, 0.0, 0.0]]))[0], seen, atol=1e-4, rtol=0)


def test_subset_merge_save_load(tmp_path):
    s = _random_splats(n_per_part=(30, 20, 10), color_mode="lit")
    sub = s.subset(["part2", "part0"])
    assert sub.part_names == ["part2", "part0"] and sub.counts() == {"part2": 10, "part0": 30}
    torch.testing.assert_close(sub.params["means"][sub.part_id == 0], s.params["means"][s.part_id == 2])
    merged = PartSplats.merge(sub, s.subset(["part1", "part0"]))
    assert merged.part_names == ["part2", "part0", "part1"]
    assert merged.counts() == {"part2": 10, "part0": 60, "part1": 20}
    with torch.no_grad():
        s.lights["ambient"].fill_(0.25)
    s.save(tmp_path / "s.pt")
    s2 = PartSplats.load(tmp_path / "s.pt")
    assert s2.part_names == s.part_names and s2.color_mode == "lit" and s2.sh_degree == s.sh_degree
    for k in s.params:
        torch.testing.assert_close(s2.params[k], s.params[k])
    torch.testing.assert_close(s2.lights["ambient"], torch.full((3,), 0.25))
    assert not s2.params["part_id"].requires_grad and s2.params["means"].requires_grad


def test_part_labels_render_one_hot():
    s = _random_splats(n_per_part=(80, 80))
    part_T = torch.stack([_T(t=(-0.08, 0, 0)), _T(t=(0.08, 0, 0))])
    out = render(s, part_T, _camera([0.0, -0.45, 0.15])[None], K[None], W, H, features=s.part_onehot(),
                 backend="dense")
    torch.testing.assert_close(out["features"].sum(-1, keepdim=True), out["alpha"], atol=1e-5, rtol=0)
    lab = out["features"].argmax(-1)[0]
    cols = torch.arange(W)
    fg = out["alpha"][0, ..., 0] > 0.5
    # part0 sits at -x, which the camera (looking along +y) sees on the left half of the image
    assert (lab[fg & (cols < W // 2 - 4)] == 0).float().mean() > 0.95
    assert (lab[fg & (cols > W // 2 + 4)] == 1).float().mean() > 0.95
    groups = s.part_onehot([["part1"]])
    assert groups.shape == (160, 1) and groups.sum() == 80


def test_carried_through_gsplat_densification_ops():
    """part_id (non-trainable) and the lit fields survive gsplat's duplicate / split / remove."""
    if not HAS_GSPLAT:
        pytest.skip("gsplat not installed")
    from gsplat.strategy.ops import duplicate, remove, split

    s = _random_splats(n_per_part=(30, 20), color_mode="lit")
    opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in s.params.items() if v.requires_grad}
    mask = torch.zeros(len(s), dtype=torch.bool)
    mask[[0, 35]] = True
    duplicate(s.params, opts, {}, mask)
    split(s.params, opts, {}, torch.cat([mask, torch.zeros(2, dtype=torch.bool)]))
    remove(s.params, opts, {}, torch.arange(len(s)) == 1)
    n = len(s)
    assert all(v.shape[0] == n for v in s.params.values()) and not s.params["part_id"].requires_grad
    assert s.counts() == {"part0": 30 + 1 + 1 - 1, "part1": 20 + 1 + 1}


# ---------------------------------------------------------------- rasterisers
def test_dense_rasteriser_single_gaussian_analytic():
    """Centre pixel of one isotropic Gaussian straight ahead: alpha = o·exp(-0.5 d^T (Σ2d + 0.3 I)^-1 d)."""
    z, s, o = 0.5, 0.01, 0.8
    vm = torch.eye(4)[None]
    rgb, depth, alpha, _ = rasterize(torch.tensor([[0.0, 0, z]]), torch.tensor([[1.0, 0, 0, 0]]),
                                     torch.full((1, 3), s), torch.tensor([o]), torch.tensor([[0.2, 0.4, 0.6]]),
                                     vm, K[None], W, H, backend="dense")
    f = K[0, 0]
    var = (f * s / z) ** 2 + 0.3
    dx = 0.5  # pixel (20, 20) centre is (20.5, 20.5); the mean projects to (20, 20)
    expect = o * math.exp(-0.5 * 2 * dx * dx / var)
    torch.testing.assert_close(alpha[0, 20, 20, 0], torch.tensor(expect))
    torch.testing.assert_close(rgb[0, 20, 20], torch.tensor([0.2, 0.4, 0.6]) * expect)
    torch.testing.assert_close(depth[0, 20, 20, 0], torch.tensor(z))
    assert alpha[0, 0, 0, 0] == 0


@needs_cuda
@pytest.mark.gpu
def test_dense_matches_gsplat_on_random_scene():
    """B3 acceptance (3): max abs difference < 1e-3 between the dense rasteriser and gsplat."""
    s = _random_splats(n_per_part=(150, 150), sh_degree=1, seed=3)
    part_T = torch.stack([_T(_rot([1, 0, 0], 0.3), (-0.05, 0, 0.02)), _T(_rot([0, 1, 1], -1.0), (0.06, 0.01, 0))])
    vms = torch.stack([_camera([0.0, -0.35, 0.2]), _camera([0.3, 0.2, 0.25]), _camera([-0.2, -0.1, 0.4])])
    Ks = K[None].expand(3, 3, 3)
    feats = s.part_onehot()
    bg = torch.tensor([0.1, 0.2, 0.3])
    dense = render(s, part_T, vms, Ks, W, H, background=bg, features=feats, backend="dense")
    sc = s.to("cuda")
    gs = render(sc, part_T.cuda(), vms.cuda(), Ks.cuda(), W, H, background=bg.cuda(), features=feats.cuda(),
                backend="gsplat")
    for k in ("rgb", "features", "alpha"):
        diff = (gs[k].cpu() - dense[k]).abs().max().item()
        assert diff < 1e-3, (k, diff)
    fg = dense["alpha"] > 0.5
    assert fg.float().mean() > 0.05
    assert (gs["depth"].cpu() - dense["depth"])[fg].abs().max() < 1e-3


@needs_cuda
@pytest.mark.gpu
def test_identity_poses_match_gsplat_native_sh():
    """B3 acceptance (1): identity part poses render like plain gsplat with its own SH evaluation."""
    from gsplat import rasterization

    s = _random_splats(n_per_part=(150, 150), sh_degree=2, seed=4).to("cuda")
    vms = torch.stack([_camera([0.0, -0.35, 0.2]), _camera([0.3, 0.2, 0.25])]).cuda()
    Ks = K[None].expand(2, 3, 3).cuda()
    ours = render(s, torch.eye(4, device="cuda").expand(2, 4, 4), vms, Ks, W, H, backend="gsplat")
    p = s.params
    ref, ref_alpha, _ = rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                      s.sh_coeffs(), vms, Ks, W, H, sh_degree=2, packed=False, near_plane=0.01,
                                      far_plane=10.0)
    assert (ours["rgb"] - ref).abs().max() < 1e-5
    assert (ours["alpha"] - ref_alpha).abs().max() < 1e-6
