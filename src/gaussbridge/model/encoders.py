"""C3: encoders that turn scene information into tokens the motion model can attend to (CLAUDE.md §8.1).

PointSetEncoder: an object's shape, given as P surface points [x, y, z, r, g, b] in the object's own frame
(dataset.box_shape_points now, PartSplats.shape_points after B4), -> one d-dimensional "shape token".

The architecture is a small PointNet:
    each point on its own:  MLP 6 -> 128 -> 256          (what is at this point?)
    pool over all points:   [max, mean] -> 512           (summary that does not depend on point order)
    MLP 512 -> d -> d                                     (the shape token)
Pooling is symmetric, so the token does not depend on the order of the points (tested).

The height-map encoder of §8.1 is intentionally not built: in both v1 tasks the table is flat and empty, so the
height map would be the same constant for every sample and carry no information (decision 2026-11-12, recorded in
CLAUDE.md §6.4). The model's context is a list of tokens, so one can be added later without other changes.
"""
import torch
from torch import nn

# Input scaling, so the network sees numbers of order 1: tabletop objects are a few cm, so positions are divided
# by 5 cm (a 4 cm cube's points lie in [-0.4, 0.4]); colours in [0, 1] are mapped to [-1, 1].
SHAPE_POS_SCALE = 0.05


class PointSetEncoder(nn.Module):
    """[..., P, 6] points -> [..., d] shape token."""

    def __init__(self, d_model, d_point=256):
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, d_point), nn.GELU())
        self.out_mlp = nn.Sequential(nn.Linear(2 * d_point, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, points):
        x = torch.cat([points[..., :3] / SHAPE_POS_SCALE, points[..., 3:] * 2 - 1], -1)
        h = self.point_mlp(x)                                        # [..., P, d_point]
        pooled = torch.cat([h.max(-2).values, h.mean(-2)], -1)       # [..., 2 d_point]
        return self.out_mlp(pooled)                                  # [..., d_model]
