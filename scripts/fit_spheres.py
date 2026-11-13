"""C6: fit sphere proxies to the Panda's collision geometry and save them (src/gaussbridge/guidance/spheres/<robot>.json).

Uses each link's COLLISION geometry from the URDF, i.e. the shapes the simulator itself uses for physics: meshes for
links 0-7 and the hand, small boxes for the fingers and the wrist camera. Spheres are fitted to the geometry's
VOLUME (filled voxels of the mesh, points inside the boxes), not its surface: centres then sit inside the part and
the spheres bulge less outside it (measured on the arm links, 32 spheres: ~15 mm mean bulge from the volume vs
~30 mm from the surface; 64 spheres ~11 mm). Points are in the link frame, the frame forward kinematics returns.

Needs ManiSkill (URDF + meshes) and trimesh, so run it on a laptop; jobs only read the small JSON.
Later (task B4) spheres can be refitted to the reconstructed Gaussians' centres with the same fit_spheres.

usage: python scripts/fit_spheres.py [--robots panda_wristcam panda] [--spheres 32]
"""
import argparse
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch_kinematics.urdf_parser_py.urdf import URDF
from transforms3d.euler import euler2mat

from gaussbridge.guidance.collision import SPHERE_DIR, fit_spheres, save_spheres
from gaussbridge.sim.kinematics import URDFS, _quiet_urdf_parser

VOXEL = 0.004  # m, voxel size for filling collision meshes
BOX_STEP = 0.002  # m, spacing of points inside collision boxes


def _origin(o):
    """URDF <origin> -> 4x4 (rpy = fixed-axis roll, pitch, yaw = euler2mat(r, p, y, "sxyz"))."""
    T = np.eye(4)
    if o is not None:
        T[:3, :3] = euler2mat(*(o.rpy or [0, 0, 0]), "sxyz")
        T[:3, 3] = o.xyz or [0, 0, 0]
    return T


def link_volume_points(robot_uid):
    """{link name: [P, 3] points filling the link's collision geometry, in the link frame}."""
    assets = Path(importlib.util.find_spec("mani_skill").submodule_search_locations[0]) / "assets"
    urdf = assets / URDFS[robot_uid]
    with _quiet_urdf_parser():
        robot = URDF.from_xml_string(urdf.read_text())
    out = {}
    for link in robot.links:
        parts = []
        for c in link.collisions:
            g, T = c.geometry, _origin(c.origin)
            if getattr(g, "filename", None):
                m = trimesh.load(urdf.parent / g.filename, force="mesh")
                if g.scale is not None:
                    m.apply_scale(g.scale)
                pts = m.voxelized(VOXEL).fill().points
            elif getattr(g, "size", None) is not None:
                axes = [np.arange(-s / 2, s / 2 + 1e-9, BOX_STEP) for s in g.size]
                pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            else:
                raise ValueError(f"{link.name}: unsupported collision geometry {type(g).__name__}")
            parts.append(pts @ T[:3, :3].T + T[:3, 3])
        if parts:
            out[link.name] = torch.tensor(np.concatenate(parts), dtype=torch.float32)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--robots", nargs="+", default=sorted(URDFS))
    p.add_argument("--spheres", type=int, default=32, help="spheres per link")
    args = p.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True).stdout.strip()
    for uid in args.robots:
        spheres = {name: fit_spheres(pts, args.spheres) for name, pts in link_volume_points(uid).items()}
        path = SPHERE_DIR / f"{uid}.json"
        save_spheres(path, spheres, {"robot_uid": uid, "source": "URDF collision geometry, volume fit",
                                     "voxel_m": VOXEL, "spheres_per_part": args.spheres, "git": commit})
        print(uid, "->", path, {k: f"{len(r)} spheres, r {r.min():.3f}-{r.max():.3f} m" for k, (c, r) in spheres.items()})


if __name__ == "__main__":
    main()
