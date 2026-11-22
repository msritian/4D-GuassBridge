"""B3 (extended in B4/B5): render a PartSplats scene state from a batch of cameras (CLAUDE.md §7.3)."""
import math

import torch

from gaussbridge import geometry as G
from gaussbridge.gs.raster import rasterize


def render(splats, part_T, viewmats, Ks, width, height, background=None, features=None, env=None,
           backend="auto", near_plane=0.01, far_plane=10.0):
    """Render one scene state (T_world_part [P,4,4]) into C cameras (viewmats [C,4,4], Ks [C,3,3]).

    background: RGB [3] or [C,3] (black if None); env: a `gs.background.EnvMap`, composited per pixel behind the
    splats instead of `background` (rgb + (1 - alpha) * env(ray)); features: extra per-Gaussian channels [N,F]
    (e.g. splats.part_onehot()), composited with a zero background and NOT divided by alpha.
    Returns {"rgb" [C,H,W,3], "features" [C,H,W,F] or None, "depth" [C,H,W,1] (expected z), "alpha" [C,H,W,1],
    "meta"}. RGB is not clamped (clamp to [0,1] only when saving)."""
    if env is not None and background is not None:
        raise ValueError("pass either background or env, not both")
    w = splats.world(part_T)
    viewmats, Ks = viewmats.to(w["means"]), Ks.to(w["means"])
    C = viewmats.shape[0]
    colors = splats.colors(w, G.viewmat_to_campos(viewmats))  # [C,N,3]
    F_ = 0
    if features is not None:
        F_ = features.shape[-1]
        colors = torch.cat([colors, features.to(colors)[None].expand(C, -1, -1)], -1)
    bg = None
    if background is not None or F_:
        rgb_bg = torch.zeros(C, 3, device=colors.device) if background is None else \
            torch.as_tensor(background).to(colors).expand(C, 3)
        bg = torch.cat([rgb_bg, torch.zeros(C, F_, device=colors.device)], -1)
    out, depth, alpha, meta = rasterize(w["means"], w["quats"], w["scales"], w["opacities"], colors, viewmats, Ks,
                                        width, height, backgrounds=bg, near_plane=near_plane, far_plane=far_plane,
                                        backend=backend)
    rgb = out[..., :3]
    if env is not None:
        rgb = rgb + (1 - alpha) * env.render(viewmats, Ks, width, height)
    return {"rgb": rgb, "features": out[..., 3:] if F_ else None, "depth": depth, "alpha": alpha, "meta": meta}


def fibonacci_sphere(n):
    """n roughly uniform unit directions [n, 3]."""
    i = torch.arange(n, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / n
    phi = math.pi * (1 + 5 ** 0.5) * i
    r = torch.sqrt(1 - z ** 2)
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], -1)


@torch.no_grad()
def surface_points(splats, name, n=512, n_views=32, size=128, alpha_min=0.5, seed=0, backend="auto"):
    """Shape-token input [n, 6] = canonical-frame [xyz, rgb] samples of one part's rendered SURFACE (CLAUDE.md §6.4).

    The part is rendered alone (identity pose) from n_views directions around it; pixels with alpha > alpha_min are
    back-projected with the rendered depth, coloured with the rendered base colour, then n are drawn at random.
    Unlike `PartSplats.shape_points` (centres of opaque Gaussians) this does not depend on how optimisation arranged
    the Gaussians: after B4 a cube is a shell of faint surface Gaussians around large opaque interior blobs, so opaque
    centres lie ~1 cm inside a 2 cm half-size cube while the rendered surface is within ~1.5 mm of it."""
    sub = splats.subset([name])
    dev = sub.device
    m = torch.sigmoid(sub.params["opacities"]) > 0.1
    mu = sub.params["means"][m] if m.any() else sub.params["means"]
    c = mu.mean(0)
    r = float((mu - c).norm(dim=-1).max()) + 3 * float(torch.exp(sub.params["scales"]).max())
    K = G.fov_to_K(2 * math.asin(0.25) * 1.2, size, size).to(dev)  # sphere of radius r fills ~80% at distance 4r
    pts, cols = [], []
    for d in fibonacci_sphere(n_views).to(dev):
        up = (0.0, 1.0, 0.0) if abs(float(d[2])) > 0.95 else (0.0, 0.0, 1.0)
        V = G.look_at_viewmat(c + d * 4 * r, c, up)
        o = render(sub, torch.eye(4, device=dev)[None], V[None], K[None], size, size, features=sub.base_rgb(),
                   backend=backend)
        a = o["alpha"][0, ..., 0]
        rows, colsi = (a > alpha_min).nonzero(as_tuple=True)
        pts.append(G.backproject(o["depth"][0, ..., 0], V, K, rows, colsi))
        cols.append((o["features"][0][rows, colsi] / a[rows, colsi, None]).clamp(0, 1))
    pts, cols = torch.cat(pts), torch.cat(cols)
    if len(pts) == 0:
        raise ValueError(f"part {name!r} renders nothing with alpha > {alpha_min}")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pts), generator=g)[:n] if len(pts) >= n else torch.randint(len(pts), (n,), generator=g)
    idx = idx.to(dev)
    return torch.cat([pts[idx], cols[idx]], -1)
