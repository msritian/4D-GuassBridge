"""C6: penetration measurement and the collision energy (CLAUDE.md §8.3), robot as spheres, objects as exact boxes.

Geometry
--------
* Robot links: spheres in each link's frame (32 per link), fitted by scripts/fit_spheres.py to the VOLUME of the
  link's URDF collision geometry (the shapes the simulator uses for physics) and saved in
  guidance/spheres/<robot_uid>.json, so jobs need no mesh files. k-means on points filling the volume, radius = the
  farthest cluster member, so the spheres cover the link. Volume rather than surface (§8.3 says surface Gaussian
  centres): surface-fitted centres sit on the skin and the spheres bulge out by up to their radius (measured ~30 mm
  on the arm links vs ~15 mm from the volume).
* Objects: their EXACT box (every v1 object is a 4 cm cube), not spheres. Sphere proxies of a cube bulge ~6 mm, and two
  such bulges made planner references show up to 7 mm of fake penetration (finger vs cubeB while stacking).
* Table: the plane z = 0.

Penetration per part pair and step (metres, >= 0)
--------------------------------------------------
    link vs object   max over the link's spheres of  r - signed_distance(sphere centre, box)
    object vs object max over points filling each cube of  -signed_distance(point, other box), both ways
                     (= how deep the deepest point of one cube sits inside the other. Exact while the overlap is
                     below the half size, since the faces are on the grid; for deeper overlaps it is within half a
                     grid spacing (2.5 mm) of the half size, 2 cm. Surface points alone would read 0 for two cubes
                     in the same place.)
    link vs table    max over the link's spheres of  r - z   (all links except panda_link0, bolted to the table)
    object vs table  -(lowest corner height)

Why EXCESS penetration
----------------------
Even with exact boxes, the robot's spheres bulge, and real contacts exist at the endpoints (a cube resting on the
table, cubeA on cubeB at the end). Start and end are real, physically valid states, so whatever a pair of parts shows
there is approximation error or legitimate contact. Filters, metrics and guidance use the excess:

    excess(pair, n) = max(0, pen(pair, n) - max(pen(pair, start), pen(pair, end)))

(This generalises §8.3's "skip pairs resting on each other at start/end" and keeps checking those pairs in between.)
Calibrated on the dev planner references (physically valid), which must score ~0; see tests and progress.md.

Skipped: gripper links (hand, fingers) vs an object whose centre is within 3 cm of the TCP, i.e. the object being
grasped (§8.3).
"""
import json
from pathlib import Path

import torch

from gaussbridge.trajectory import decode_motion

GRIPPER_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
GRASP_RADIUS = 0.03            # m, object centre to TCP below which gripper-object contact is a grasp, not a collision
OBJECT_HALF_SIZE = 0.02        # m, every v1 object is a 4 cm cube
SPHERE_DIR = Path(__file__).resolve().parent / "spheres"   # inside the package, so CHTC bundles ship it


# ---------------------------------------------------------------------------------------------------- fitting
def kmeans(points, k, iters=50, seed=0):
    """k-means on points [P, 3] with k-means++ initialisation (deterministic in seed) -> centres [k, 3], labels [P]."""
    g = torch.Generator().manual_seed(seed)
    centres = points[torch.randint(len(points), (1,), generator=g)]
    for _ in range(1, k):  # k-means++: next centre far from the chosen ones
        d2 = torch.cdist(points, centres).min(1).values.square()
        centres = torch.cat([centres, points[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
    for _ in range(iters):
        labels = torch.cdist(points, centres).argmin(1)
        new = torch.stack([points[labels == j].mean(0) if (labels == j).any() else centres[j] for j in range(k)])
        if torch.allclose(new, centres):
            break
        centres = new
    return centres, torch.cdist(points, centres).argmin(1)


def fit_spheres(points, k=32, seed=0):
    """Covering spheres of a point set (points filling a part): k-means centres, radius = farthest member."""
    centres, labels = kmeans(points, k, seed=seed)
    radii = torch.stack([(points[labels == j] - centres[j]).norm(dim=-1).max() if (labels == j).any()
                         else points.new_zeros(()) for j in range(k)])
    return centres, radii


def save_spheres(path, spheres, meta):
    """{part: (centres [k,3], radii [k])} -> JSON (small, human-readable, committed with the code)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"meta": meta, "parts": {
        name: {"centres": c.tolist(), "radii": r.tolist()} for name, (c, r) in spheres.items()}}, indent=1))


def load_spheres(path):
    d = json.loads(Path(path).read_text())
    return {name: (torch.tensor(v["centres"]), torch.tensor(v["radii"])) for name, v in d["parts"].items()}


# ---------------------------------------------------------------------------------------------------- box geometry
def box_signed_distance(q, half):
    """Signed distance of points q [..., 3] given in a box's own frame to the box with half sizes `half`
    (positive outside, negative inside: minus the distance to the nearest face)."""
    d = q.abs() - half
    outside = d.clamp_min(0).norm(dim=-1)
    inside = d.amax(-1).clamp_max(0)
    return outside + inside


def box_grid(half, n=9):
    """A regular n x n x n grid of points filling a box, surface included (corners, edges, faces, interior)."""
    t = torch.linspace(-1, 1, n)
    return torch.stack(torch.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3) * torch.as_tensor(half)


# ---------------------------------------------------------------------------------------------------- the scene
class SceneSpheres:
    """Collision geometry of one task's scene: robot link spheres (from a sphere file), objects as exact boxes, table.

    `penetration(x)` decodes motion grids x [B, N, E, F], places everything with forward kinematics / the object poses
    and returns the excess penetration per part pair and step. Differentiable in x (for guidance)."""

    def __init__(self, fk, link_spheres, n_objects, object_half_size=OBJECT_HALF_SIZE, device="cpu"):
        self.fk, self.n_objects, self.device = fk, n_objects, device
        self.links = list(link_spheres)
        self.link_c = torch.stack([link_spheres[k][0] for k in self.links]).to(device)       # [L, s, 3]
        self.link_r = torch.stack([link_spheres[k][1] for k in self.links]).to(device)       # [L, s]
        self.half = torch.full((3,), object_half_size, device=device)
        self.obj_points = box_grid(self.half.cpu()).to(device)                                 # [Q, 3]
        corners = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
        self.obj_corners = (corners * self.half.cpu()).to(device)                              # [8, 3]
        self.gripper = torch.tensor([name in GRIPPER_LINKS for name in self.links], device=device)
        self.on_table_check = torch.tensor([name != "panda_link0" for name in self.links], device=device)

    @classmethod
    def for_robot(cls, fk, n_objects, device="cpu", sphere_file=None):
        path = sphere_file or SPHERE_DIR / f"{fk.robot_uid}.json"
        return cls(fk, load_spheres(path), n_objects, device=device)

    def place(self, x):
        """World geometry for motion grids x [B, N, E, F]: link sphere centres [B, N, L, s, 3], object rotations
        [B, N, K, 3, 3] and positions [B, N, K, 3], TCP [B, N, 3]."""
        d = decode_motion(x, 0.0)
        poses = self.fk.link_poses(d["qpos"].to(self.fk.dtype), links=self.links + ["panda_hand_tcp"])
        T = torch.stack([poses[k] for k in self.links], -3).to(x.dtype)                     # [B, N, L, 4, 4]
        links = torch.einsum("bnlij,lsj->bnlsi", T[..., :3, :3], self.link_c) + T[..., None, :3, 3]
        k = self.n_objects
        return links, d["obj_rot"][..., :k, :, :], d["obj_pos"][..., :k, :], poses["panda_hand_tcp"][..., :3, 3].to(x.dtype)

    def raw_penetration(self, x):
        """Penetration per part pair and step before the endpoint correction:
        {"link_obj" [B, N, L, K], "obj_obj" [B, N, K, K] (diagonal 0), "link_table" [B, N, L], "obj_table" [B, N, K]}.
        Grasp pairs (gripper link vs an object within GRASP_RADIUS of the TCP) are 0."""
        links, R, p, tcp = self.place(x)
        K = R.shape[2]
        # link spheres in every object's box frame: q = R^T (c - p)   [B, N, L, s, K, 3]
        rel = links[:, :, :, :, None, :] - p[:, :, None, None, :, :]
        q = torch.einsum("bnkji,bnlskj->bnlski", R, rel)
        sd = box_signed_distance(q, self.half)                                                 # [B, N, L, s, K]
        link_obj = (self.link_r[:, :, None] - sd).clamp_min(0).amax(3)                        # [B, N, L, K]
        grasped = (p - tcp[..., None, :]).norm(dim=-1) < GRASP_RADIUS                         # [B, N, K]
        link_obj = link_obj * ~(self.gripper[:, None] & grasped[:, :, None, :])
        # points filling each cube (surface + interior) inside the other boxes, both directions
        surf = torch.einsum("bnkij,qj->bnkqi", R, self.obj_points) + p[..., None, :]           # [B, N, K, Q, 3]
        rel = surf[:, :, :, None] - p[:, :, None, :, None, :]                                  # [B, N, K, K', Q, 3]
        q = torch.einsum("bnmji,bnkmqj->bnkmqi", R, rel)                                       # in box m's frame
        obj_obj = (-box_signed_distance(q, self.half)).clamp_min(0).amax(-1)                  # [B, N, K, K']
        obj_obj = obj_obj * ~torch.eye(K, dtype=torch.bool, device=x.device)
        obj_obj = torch.maximum(obj_obj, obj_obj.transpose(-1, -2))                            # symmetric
        # table plane z = 0
        link_table = (self.link_r - links[..., 2]).clamp_min(0).amax(-1) * self.on_table_check
        corner_z = (torch.einsum("bnkij,cj->bnkci", R, self.obj_corners) + p[..., None, :])[..., 2]
        obj_table = (-corner_z.amin(-1)).clamp_min(0)
        return {"link_obj": link_obj, "obj_obj": obj_obj, "link_table": link_table, "obj_table": obj_table}

    def penetration(self, x):
        """Excess penetration (metres) per part pair and step: raw penetration minus the larger of the same pair's
        penetration at the start and at the end (module docstring). Same keys/shapes as raw_penetration."""
        out = {}
        for key, pen in self.raw_penetration(x).items():
            allowed = torch.maximum(pen[:, :1], pen[:, -1:])
            out[key] = (pen - allowed).clamp_min(0)
        return out

    def max_penetration(self, x):
        """[B] largest excess penetration over all steps and part pairs (metres): the filter / metric quantity."""
        return torch.stack([v.flatten(1).amax(1) for v in self.penetration(x).values()], -1).amax(-1)

    def energy(self, x):
        """[B] collision energy for guidance: sum over steps and part pairs of excess penetration squared.
        Its gradient comes from the deepest point / sphere of each pair (max); at a perfectly symmetric configuration
        (a cube's grid point exactly at the other cube's centre) that point's gradient is 0, which real samples never
        hit exactly."""
        return sum(v.square().flatten(1).sum(1) for v in self.penetration(x).values())
