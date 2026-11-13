"""B3: PartSplats, the part-structured Gaussian scene (CLAUDE.md §7.1, §7.3).

Every Gaussian belongs to one rigid part (robot link, object, background entity) and is stored in that part's
canonical frame. A scene state is T_world_part [P, 4, 4] in `part_names` order (the capture part table order);
`world(part_T)` moves the Gaussians: mu_w = R_p mu + t_p, q_w = q(R_p) ⊗ q (covariance R_p Σ R_pᵀ), scales and
opacities unchanged.

Parameters are stored like gsplat's trainer (pre-activation: `scales` log, `opacities` logit) in `self.params`, a
ParameterDict that gsplat's DefaultStrategy can densify directly. `part_id` is a non-trainable Parameter so the
strategy carries it through duplicate/split/prune.

Two colour models (`color_mode`):

* "sh":  the plan's baseline. colour = clamp_min(SH(d_local) + 0.5, 0) with the view direction in the part frame,
  d_local = R_pᵀ (mu_w − campos) (gsplat does not rotate SH, so we evaluate it ourselves).
* "lit": ManiSkill's lights are fixed in the world (B1 lighting check: 21 dB cap for "sh" on a yawed cube), so
  colour = clamp(lin^(1/2.2), 0, 1) with lin = albedo · (ambient + Σ_i e_i · max(0, (R_p n)·l_i)) + SH_res(d_local).
  This is the diffuse part of SAPIEN's `minimal` shader (gbuffer.frag: ambient·albedo + Σ albedo·e·max(0,n·l)/π,
  then pow(1/2.2) and clamp); its specular/IBL terms are left to the small part-frame SH residual (init 0).
  Light directions are ManiSkill's (BaseEnv._load_lighting); ambient [3] and intensities [L] are learnable.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from gaussbridge import geometry as G

SH_C0 = 0.28209479177387814
# ManiSkill BaseEnv._load_lighting: ambient 0.3, white directional lights travelling along these directions.
MANISKILL_LIGHT_DIRS = ((1.0, 1.0, -1.0), (0.0, 0.0, -1.0))
MANISKILL_AMBIENT = 0.3
LAMBERT_SCALE = 1.0 / math.pi  # SAPIEN's diffuse(NoL) = NoL / pi with emission 1
GAMMA = 2.2
_LIN_EPS = 1e-4  # keeps d/dx x^(1/2.2) finite


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh0):
    return sh0 * SH_C0 + 0.5


def srgb_to_linear(c):
    return c.clamp(0, 1) ** GAMMA


def sh_bases(degree, dirs):
    """Real SH basis [..., (degree+1)^2] at unit dirs [..., 3], same ordering/signs as gsplat (Sloan 2013)."""
    out = [torch.full_like(dirs[..., 0], 0.2820947917738781)]
    if degree >= 1:
        x, y, z = dirs.unbind(-1)
        a = -0.48860251190292
        out += [a * y, -a * z, a * x]
    if degree >= 2:
        z2, fC1, fS1 = z * z, x * x - y * y, 2 * x * y
        b = -1.092548430592079 * z
        out += [0.5462742152960395 * fS1, b * y, 0.9461746957575601 * z2 - 0.3153915652525201, b * x,
                0.5462742152960395 * fC1]
    if degree >= 3:
        c = -2.285228997322329 * z2 + 0.4570457994644658
        b = 1.445305721320277 * z
        fC2, fS2 = x * fC1 - y * fS1, x * fS1 + y * fC1
        out += [-0.5900435899266435 * fS2, b * fS1, c * y, z * (1.865881662950577 * z2 - 1.119528997770346),
                c * x, b * fC1, -0.5900435899266435 * fC2]
    if degree > 3:
        raise ValueError("SH degree > 3 is not supported")
    return torch.stack(out, -1)


def eval_sh(degree, dirs, coeffs):
    """sum_k basis_k(normalize(dirs)) * coeffs[..., k, :] -> [..., 3] (= gsplat spherical_harmonics, no +0.5)."""
    K = (degree + 1) ** 2
    return (sh_bases(degree, F.normalize(dirs, dim=-1))[..., None] * coeffs[..., :K, :]).sum(-2)


def knn_mean_dist(points, k=3, chunk=4096):
    """Mean distance to the k nearest other points [M] (brute force, chunked)."""
    out = []
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i:i + chunk], points)
        out.append(d.topk(min(k + 1, len(points)), largest=False).values[:, 1:].mean(-1))
    return torch.cat(out) if out else points.new_zeros(0)


def estimate_normals(points, k=8, chunk=4096):
    """PCA normals [M,3] from k nearest neighbours, oriented away from the point set's centroid (fine for the
    convex-ish rigid parts we have). Prefer depth-based normals with a known camera side when available."""
    normals = []
    for i in range(0, len(points), chunk):
        idx = torch.cdist(points[i:i + chunk], points).topk(min(k, len(points)), largest=False).indices
        nb = points[idx]  # [m, k, 3]
        nb = nb - nb.mean(1, keepdim=True)
        _, vecs = torch.linalg.eigh(nb.transpose(1, 2) @ nb)
        normals.append(vecs[..., 0])  # smallest-eigenvalue direction
    n = torch.cat(normals) if normals else points.new_zeros(0, 3)
    flip = ((points - points.mean(0)) * n).sum(-1) < 0
    return torch.where(flip[:, None], -n, n)


class PartSplats(nn.Module):
    """Per-part canonical Gaussians. See the module docstring for the conventions."""

    def __init__(self, params, part_names, part_kinds, sh_degree=1, color_mode="sh",
                 light_dirs=MANISKILL_LIGHT_DIRS, ambient=MANISKILL_AMBIENT, light_intensity=LAMBERT_SCALE,
                 meta=None):
        super().__init__()
        if color_mode not in ("sh", "lit"):
            raise ValueError(f"color_mode must be 'sh' or 'lit', got {color_mode!r}")
        need = {"means", "scales", "quats", "opacities", "sh0", "shN", "part_id"}
        if color_mode == "lit":
            need |= {"albedo", "normals"}
        missing = need - set(params)
        if missing:
            raise KeyError(f"missing params {sorted(missing)}")
        if len(part_names) != len(part_kinds):
            raise ValueError("part_names and part_kinds differ in length")
        self.part_names, self.part_kinds = list(part_names), list(part_kinds)
        self.sh_degree, self.color_mode, self.meta = int(sh_degree), color_mode, dict(meta or {})
        self.params = nn.ParameterDict({
            k: nn.Parameter(v.detach().clone() if isinstance(v, torch.Tensor) else torch.as_tensor(v),
                            requires_grad=(k != "part_id")) for k, v in params.items()})
        self.params["part_id"] = nn.Parameter(self.params["part_id"].data.long(), requires_grad=False)
        dev = self.params["means"].device
        dirs = F.normalize(torch.as_tensor(light_dirs, dtype=torch.float32, device=dev), dim=-1)
        self.register_buffer("light_dirs", dirs)  # direction the light travels (ManiSkill), [L, 3]
        amb = torch.as_tensor(ambient, dtype=torch.float32, device=dev).expand(3).clone()
        inten = torch.as_tensor(light_intensity, dtype=torch.float32, device=dev).expand(len(dirs)).clone()
        self.lights = nn.ParameterDict({"ambient": nn.Parameter(amb), "intensity": nn.Parameter(inten)})

    # ------------------------------------------------------------------ basic info
    def __len__(self):
        return self.params["means"].shape[0]

    @property
    def num_parts(self):
        return len(self.part_names)

    @property
    def part_id(self):
        return self.params["part_id"]

    @property
    def device(self):
        return self.params["means"].device

    def part_index(self, name):
        return self.part_names.index(name)

    def part_mask(self, names):
        ids = torch.tensor([self.part_index(n) for n in ([names] if isinstance(names, str) else names)],
                           device=self.device)
        return torch.isin(self.part_id, ids)

    def counts(self):
        """Number of Gaussians per part, {name: n}."""
        c = torch.bincount(self.part_id, minlength=self.num_parts).tolist()
        return dict(zip(self.part_names, c))

    def part_T_from_dict(self, poses, default=None):
        """{part name: T_world_part [4,4]} -> [P,4,4]. Parts without Gaussians may be missing (identity);
        others raise unless `default` [4,4] is given."""
        counts = self.counts()
        out = []
        for n in self.part_names:
            if n in poses:
                out.append(torch.as_tensor(poses[n], dtype=torch.float32))
            elif default is not None or counts[n] == 0:
                out.append(torch.eye(4) if default is None else torch.as_tensor(default, dtype=torch.float32))
            else:
                raise KeyError(f"no pose for part {n!r} ({counts[n]} Gaussians)")
        return torch.stack(out).to(self.device)

    # ------------------------------------------------------------------ composition + colour
    def world(self, part_T):
        """Pose every Gaussian with T_world_part [P,4,4] (differentiable in part_T, for pose refinement).
        Returns {means, quats (unit), scales, opacities (activated), R [N,3,3] (part rotation per Gaussian)}."""
        p = self.params
        if part_T.shape != (self.num_parts, 4, 4):
            raise ValueError(f"part_T must be [{self.num_parts}, 4, 4], got {tuple(part_T.shape)}")
        part_T = part_T.to(p["means"])
        Rp, tp = part_T[:, :3, :3], part_T[:, :3, 3]
        R, t = Rp[self.part_id], tp[self.part_id]
        means = (R @ p["means"][:, :, None]).squeeze(-1) + t
        quats = G.quat_mul(G.matrix_to_quat(Rp)[self.part_id], F.normalize(p["quats"], dim=-1))
        return {"means": means, "quats": quats, "scales": torch.exp(p["scales"]),
                "opacities": torch.sigmoid(p["opacities"]), "R": R}

    def sh_coeffs(self):
        return torch.cat([self.params["sh0"], self.params["shN"]], 1)

    def albedo(self):
        """Linear albedo in [0, 1] [N,3] ("lit" only)."""
        return torch.sigmoid(self.params["albedo"])

    def shading(self, R):
        """ambient + Σ e_i max(0, n_w · l_i) [N,3] with world normals n_w = R n ("lit" only)."""
        n_w = F.normalize((R @ self.params["normals"][:, :, None]).squeeze(-1), dim=-1)
        to_light = -self.light_dirs.to(n_w)  # [L,3]
        lam = (n_w @ to_light.T).clamp_min(0) * self.lights["intensity"].clamp_min(0)  # [N,L]
        return self.lights["ambient"].clamp_min(0) + lam.sum(-1, keepdim=True)

    def colors(self, world, campos):
        """Per-camera RGB [C,N,3] in [0,1] (sRGB) for cameras at campos [C,3] (world)."""
        R = world["R"]
        d_world = world["means"][None] - campos[:, None].to(world["means"])  # [C,N,3]
        d_local = torch.einsum("nji,cnj->cni", R, d_world)  # R^T d
        sh = eval_sh(self.sh_degree, d_local, self.sh_coeffs()[None].expand(len(campos), -1, -1, -1))
        if self.color_mode == "sh":
            return (sh + 0.5).clamp_min(0)
        lin = self.albedo() * self.shading(R) + sh
        return lin.clamp_min(_LIN_EPS).pow(1 / GAMMA).clamp_max(1)

    def part_onehot(self, groups=None):
        """Label features [N, P] (one-hot part index), or [N, len(groups)] for groups = [[part names], ...]."""
        if groups is None:
            return F.one_hot(self.part_id, self.num_parts).float()
        lut = torch.full((self.num_parts,), -1, dtype=torch.long, device=self.device)
        for g, names in enumerate(groups):
            for n in names:
                lut[self.part_index(n)] = g
        gid = lut[self.part_id]
        out = torch.zeros(len(self), len(groups), device=self.device)
        ok = gid >= 0
        out[ok, gid[ok]] = 1
        return out

    def base_rgb(self):
        """View-independent base colour [N,3] in [0,1]: SH DC term ("sh") or gamma-encoded albedo ("lit")."""
        if self.color_mode == "sh":
            return sh_to_rgb(self.params["sh0"][:, 0]).clamp(0, 1)
        return self.albedo().pow(1 / GAMMA)

    def shape_points(self, name, n=512, min_opacity=0.5, seed=0):
        """Shape-token input [n, 6] = canonical centres + base colour of one part's Gaussians with opacity >
        min_opacity (CLAUDE.md §6.4), sampled without replacement when possible."""
        with torch.no_grad():
            m = self.part_mask(name) & (torch.sigmoid(self.params["opacities"]) > min_opacity)
            pts = torch.cat([self.params["means"][m], self.base_rgb()[m]], -1)
            if len(pts) == 0:
                raise ValueError(f"part {name!r} has no Gaussians with opacity > {min_opacity}")
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(pts), generator=g)[:n] if len(pts) >= n else \
                torch.randint(len(pts), (n,), generator=g)
            return pts[idx.to(pts.device)]

    # ------------------------------------------------------------------ subset / merge / io
    def _config(self):
        return {"part_names": self.part_names, "part_kinds": self.part_kinds, "sh_degree": self.sh_degree,
                "color_mode": self.color_mode, "light_dirs": self.light_dirs.tolist(),
                "ambient": self.lights["ambient"].detach().cpu(),
                "light_intensity": self.lights["intensity"].detach().cpu(), "meta": self.meta}

    def subset(self, names):
        """New PartSplats with only the listed parts, in that order (part_id re-indexed)."""
        old = torch.tensor([self.part_index(n) for n in names], dtype=torch.long)
        lut = torch.full((self.num_parts,), -1, dtype=torch.long)
        lut[old] = torch.arange(len(names))
        lut = lut.to(self.device)
        keep = lut[self.part_id] >= 0
        params = {k: v.detach()[keep] for k, v in self.params.items()}
        params["part_id"] = lut[self.part_id[keep]]
        cfg = self._config()
        cfg["part_names"], cfg["part_kinds"] = list(names), [self.part_kinds[i] for i in old.tolist()]
        return PartSplats(params, **cfg)

    @staticmethod
    def merge(*splats):
        """Concatenate splats into one; parts are unioned by name in first-seen order. Settings (colour model,
        SH degree, lights) come from the first; all must share colour model and SH degree."""
        first = splats[0]
        for s in splats[1:]:
            if (s.color_mode, s.sh_degree) != (first.color_mode, first.sh_degree):
                raise ValueError("merge needs equal color_mode and sh_degree")
        names, kinds = [], []
        for s in splats:
            for n, k in zip(s.part_names, s.part_kinds):
                if n not in names:
                    names.append(n)
                    kinds.append(k)
        params = {}
        for k in first.params:
            parts = []
            for s in splats:
                v = s.params[k].detach()
                if k == "part_id":
                    lut = torch.tensor([names.index(n) for n in s.part_names], device=v.device)
                    v = lut[v]
                parts.append(v.to(first.device))
            params[k] = torch.cat(parts)
        cfg = first._config()
        cfg["part_names"], cfg["part_kinds"] = names, kinds
        return PartSplats(params, **cfg)

    def save(self, path):
        torch.save({"params": {k: v.detach().cpu() for k, v in self.params.items()}, **self._config()}, path)

    @staticmethod
    def load(path, device="cpu"):
        d = torch.load(path, map_location="cpu", weights_only=True)
        params = d.pop("params")
        return PartSplats(params, **d).to(device)


def init_from_points(points, rgb, part_id, part_names, part_kinds, sh_degree=1, color_mode="sh", normals=None,
                     part_T=None, point_R=None, init_opacity=0.1, scale_clamp=(1e-4, 0.05), meta=None,
                     **light_kwargs):
    """PartSplats from canonical-frame points (CLAUDE.md §7.2 init).

    points [M,3] are in their part's frame, rgb [M,3] in [0,1] (sRGB as captured), part_id [M] indexes part_names.
    scale = mean distance to the 3 nearest points of the same part, clamped to scale_clamp (m); isotropic, identity
    rotation, opacity init_opacity. "sh": sh0 = rgb_to_sh(rgb), shN = 0. "lit": SH residual = 0, normals given
    (part frame, pointing out of the surface) or estimated by PCA, albedo = linear(rgb) / shading(normal) under the
    initial lights with the part rotation each colour was seen at: point_R [M,3,3] (points gathered from several
    configurations), else part_T [P,4,4] (one configuration), else identity; clipped to [0.001, 0.999]."""
    points = torch.as_tensor(points, dtype=torch.float32)
    dev = points.device
    rgb = torch.as_tensor(rgb, dtype=torch.float32, device=dev)
    part_id = torch.as_tensor(part_id, dtype=torch.long, device=dev)
    M = len(points)
    scales = torch.empty(M, device=dev)
    est_normals = torch.zeros(M, 3, device=dev)
    for p in part_id.unique().tolist():
        m = part_id == p
        if m.sum() > 1:
            scales[m] = knn_mean_dist(points[m])
            if color_mode == "lit" and normals is None:
                est_normals[m] = estimate_normals(points[m])
        else:
            scales[m] = scale_clamp[1]
            est_normals[m] = torch.tensor([0.0, 0.0, 1.0], device=dev)
    scales = scales.clamp(*scale_clamp)
    K = (sh_degree + 1) ** 2
    params = {
        "means": points,
        "scales": torch.log(scales)[:, None].repeat(1, 3),
        "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(M, 1),
        "opacities": torch.logit(torch.full((M,), init_opacity, device=dev)),
        "sh0": rgb_to_sh(rgb)[:, None] if color_mode == "sh" else torch.zeros(M, 1, 3, device=dev),
        "shN": torch.zeros(M, K - 1, 3, device=dev),
        "part_id": part_id,
    }
    if color_mode == "lit":
        params["normals"] = est_normals if normals is None else torch.as_tensor(normals, dtype=torch.float32,
                                                                                device=dev)
        params["albedo"] = torch.zeros(M, 3, device=dev)
    splats = PartSplats(params, part_names, part_kinds, sh_degree, color_mode, meta=meta, **light_kwargs).to(dev)
    if color_mode == "lit":
        with torch.no_grad():
            if point_R is not None:
                R = torch.as_tensor(point_R, dtype=torch.float32, device=dev)
            elif part_T is None:
                R = torch.eye(3, device=dev).expand(M, 3, 3)
            else:
                R = torch.as_tensor(part_T, dtype=torch.float32, device=dev)[part_id, :3, :3]
            albedo = (srgb_to_linear(rgb) / splats.shading(R)).clamp(1e-3, 0.999)
            splats.params["albedo"].copy_(torch.logit(albedo))
    return splats
