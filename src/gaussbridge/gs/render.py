"""B3 (extended in B5): render a PartSplats scene state from a batch of cameras (CLAUDE.md §7.3)."""
import torch

from gaussbridge import geometry as G
from gaussbridge.gs.raster import rasterize


def render(splats, part_T, viewmats, Ks, width, height, background=None, features=None, backend="auto",
           near_plane=0.01, far_plane=10.0):
    """Render one scene state (T_world_part [P,4,4]) into C cameras (viewmats [C,4,4], Ks [C,3,3]).

    background: RGB [3] or [C,3] (black if None); features: extra per-Gaussian channels [N,F] (e.g.
    splats.part_onehot()), composited with a zero background and NOT divided by alpha.
    Returns {"rgb" [C,H,W,3], "features" [C,H,W,F] or None, "depth" [C,H,W,1] (expected z), "alpha" [C,H,W,1],
    "meta"}. RGB is not clamped (clamp to [0,1] only when saving)."""
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
    return {"rgb": out[..., :3], "features": out[..., 3:] if F_ else None, "depth": depth, "alpha": alpha,
            "meta": meta}
