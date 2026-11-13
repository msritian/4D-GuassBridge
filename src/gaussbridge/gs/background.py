"""B4: what lies behind all Gaussians, as a function of the world viewing direction (an environment map).

ManiSkill's ground plane reaches the horizon: beyond a few metres it is 22-25% of every base_camera frame (B3
check) and too far to cover with Gaussians. It is an untextured grey under world-fixed lights, so its colour depends
on the viewing direction only (parallax between cameras changes which floor point is seen, not its colour). Above
the horizon the sky renders black. An equirectangular texture over (elevation, azimuth), composited behind the
splats as rgb + (1 - alpha) * env(ray), models both for any camera, including the moving hand camera.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from gaussbridge import geometry as G


class EnvMap(nn.Module):
    """Learnable background colour per world ray direction. `tex` holds logits [3, n_el, n_az]; rows span
    elevation [-90°, 90°] (fine rows: the horizon is a sharp edge), columns azimuth [-180°, 180°) with wrap-around."""

    def __init__(self, n_el=512, n_az=128, init_rgb=(0.5, 0.5, 0.5), tex=None):
        super().__init__()
        if tex is None:
            rgb = torch.as_tensor(init_rgb, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
            tex = torch.logit(rgb)[:, None, None].expand(3, n_el, n_az).clone()
        self.tex = nn.Parameter(torch.as_tensor(tex, dtype=torch.float32))

    def forward(self, dirs):
        """sRGB colour [..., 3] in [0, 1] for world directions dirs [..., 3] (any length)."""
        shape = dirs.shape[:-1]
        d = F.normalize(dirs.reshape(-1, 3).to(self.tex), dim=-1)
        el = torch.asin(d[:, 2].clamp(-1, 1))
        az = torch.atan2(d[:, 1], d[:, 0])
        # columns 0..n_az-1 cover [-pi, pi); append column 0 at the end so az = +pi blends back into it
        tex = torch.cat([self.tex, self.tex[..., :1]], -1)
        grid = torch.stack([az / math.pi, el / (math.pi / 2)], -1)[None, None]  # x = azimuth, y = elevation
        out = F.grid_sample(torch.sigmoid(tex)[None], grid, mode="bilinear", padding_mode="border",
                            align_corners=True)  # [1, 3, 1, M]
        return out[0, :, 0].T.reshape(*shape, 3)

    def render(self, viewmats, Ks, width, height):
        """Background images [C, H, W, 3] for C cameras."""
        return torch.stack([self(G.pixel_rays(v, k, width, height)) for v, k in zip(viewmats, Ks)])

    def save(self, path):
        torch.save({"tex": self.tex.detach().cpu()}, path)

    @staticmethod
    def load(path, device="cpu"):
        return EnvMap(tex=torch.load(path, map_location="cpu", weights_only=True)["tex"]).to(device)
