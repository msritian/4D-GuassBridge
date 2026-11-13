"""B3: Gaussian rasterisation backends (CLAUDE.md §7.3).

`rasterize` renders world-space Gaussians with per-camera colours of any channel count D and returns
(colors [C,H,W,D], expected depth [C,H,W,1], alpha [C,H,W,1], meta). Two backends:

* "gsplat": gsplat 1.5.3 `rasterization` (CUDA), render_mode "RGB+ED", packed=False, sh_degree=None.
* "dense":  a small, differentiable PyTorch re-implementation for CPU unit tests. It follows gsplat's
  classic-mode kernels (projection_ewa_3dgs_fused / persp_proj / rasterize_to_pixels_3dgs_fwd): EWA
  projection with the Jacobian clamped to 1.3x the field of view, eps2d blur added to the 2D covariance,
  Gaussians culled outside [near, far] or with opacity < 1/255, alpha = min(0.999, o·exp(-σ)) evaluated at
  pixel centres (j + 0.5, i + 0.5), alphas < 1/255 skipped, front-to-back compositing by camera z that stops
  (exclusively) once transmittance would fall to 1e-4. Memory is O(H·W·N) per camera: tests only.

"auto" picks gsplat for CUDA tensors and dense otherwise.
"""
import torch

ALPHA_THRESHOLD = 1.0 / 255.0  # gsplat Common.h
T_EPS = 1e-4                   # gsplat rasterize_to_pixels: a pixel is done once T would drop to this


def rasterize(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
              backgrounds=None, near_plane=0.01, far_plane=10.0, eps2d=0.3, backend="auto"):
    """means [N,3], quats [N,4] (wxyz, any norm), scales [N,3] and opacities [N] (activated),
    colors [N,D] or [C,N,D], viewmats [C,4,4] (world->camera, OpenCV), Ks [C,3,3], backgrounds [C,D] or None.
    Returns colors [C,H,W,D], depth [C,H,W,1] (expected camera z; 0/alpha where empty), alpha [C,H,W,1], meta."""
    if backend == "auto":
        backend = "gsplat" if means.is_cuda else "dense"
    C = viewmats.shape[0]
    if colors.dim() == 2:
        colors = colors[None].expand(C, -1, -1)
    if backend == "gsplat":
        return _rasterize_gsplat(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                                 backgrounds, near_plane, far_plane, eps2d)
    if backend == "dense":
        return _rasterize_dense(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                                backgrounds, near_plane, far_plane, eps2d)
    raise ValueError(f"unknown backend {backend!r}")


def _rasterize_gsplat(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                      backgrounds, near_plane, far_plane, eps2d):
    from gsplat import rasterization

    out, alpha, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane, eps2d=eps2d, sh_degree=None, packed=False,
        backgrounds=backgrounds, render_mode="RGB+ED", rasterize_mode="classic")
    meta["backend"] = "gsplat"
    return out[..., :-1], out[..., -1:], alpha, meta


def quat_scale_to_covar(quats, scales):
    """Sigma = R diag(s^2) R^T [N,3,3] (quats are normalised first, as gsplat)."""
    from gaussbridge.geometry import quat_to_matrix

    M = quat_to_matrix(quats) * scales[..., None, :]
    return M @ M.transpose(-1, -2)


def project_gaussians(means, covars, viewmat, K, width, height, eps2d=0.3):
    """gsplat's pinhole EWA projection for one camera. Returns mean2d [N,2], cov2d [N,2,2] (incl. eps2d), z [N]."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    pc = means @ R.T + t
    covc = R @ covars @ R.T
    x, y, z = pc.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tan_fovx, tan_fovy = 0.5 * width / fx, 0.5 * height / fy
    lim_x_pos, lim_x_neg = (width - cx) / fx + 0.3 * tan_fovx, cx / fx + 0.3 * tan_fovx
    lim_y_pos, lim_y_neg = (height - cy) / fy + 0.3 * tan_fovy, cy / fy + 0.3 * tan_fovy
    tx = z * torch.minimum(lim_x_pos, torch.maximum(-lim_x_neg, x / z))
    ty = z * torch.minimum(lim_y_pos, torch.maximum(-lim_y_neg, y / z))
    J = torch.zeros(*z.shape, 2, 3, dtype=means.dtype, device=means.device)
    J[..., 0, 0] = fx / z
    J[..., 0, 2] = -fx * tx / z ** 2
    J[..., 1, 1] = fy / z
    J[..., 1, 2] = -fy * ty / z ** 2
    cov2d = J @ covc @ J.transpose(-1, -2) + eps2d * torch.eye(2, dtype=means.dtype, device=means.device)
    mean2d = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)
    return mean2d, cov2d, z


def _rasterize_dense(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                     backgrounds, near_plane, far_plane, eps2d):
    covars = quat_scale_to_covar(quats, scales)
    D = colors.shape[-1]
    jj, ii = torch.meshgrid(torch.arange(width, dtype=means.dtype, device=means.device) + 0.5,
                            torch.arange(height, dtype=means.dtype, device=means.device) + 0.5, indexing="xy")
    pix = torch.stack([jj, ii], -1).reshape(-1, 2)  # [HW, 2] pixel centres (x, y)
    outs, depths, alphas = [], [], []
    for c in range(viewmats.shape[0]):
        mean2d, cov2d, z = project_gaussians(means, covars, viewmats[c], Ks[c], width, height, eps2d)
        det = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
        keep = (z >= near_plane) & (z <= far_plane) & (det > 0) & (opacities >= ALPHA_THRESHOLD)
        order = torch.argsort(torch.where(keep, z, torch.full_like(z, float("inf"))))
        order = order[keep[order]]
        m2, cv, zz, op, col = mean2d[order], cov2d[order], z[order], opacities[order], colors[c][order]
        dt = det[order]
        conic = torch.stack([cv[:, 1, 1] / dt, -cv[:, 0, 1] / dt, cv[:, 0, 0] / dt], -1)  # inverse (a, b, c)
        d = m2[None] - pix[:, None]  # [HW, n, 2], gsplat: delta = mean - pixel
        sigma = 0.5 * (conic[:, 0] * d[..., 0] ** 2 + conic[:, 2] * d[..., 1] ** 2) + conic[:, 1] * d[..., 0] * d[..., 1]
        a = torch.clamp(op * torch.exp(-sigma), max=0.999)
        a = torch.where((sigma >= 0) & (a >= ALPHA_THRESHOLD), a, torch.zeros_like(a))
        T_incl = torch.cumprod(1 - a, -1)
        a = torch.where(T_incl > T_EPS, a, torch.zeros_like(a))  # exclusive stop: drop the one that crosses
        T_incl = torch.cumprod(1 - a, -1)
        T_excl = torch.cat([torch.ones_like(T_incl[:, :1]), T_incl[:, :-1]], -1)
        w = a * T_excl  # [HW, n]
        T_final = T_incl[:, -1] if T_incl.shape[1] else torch.ones(pix.shape[0], dtype=means.dtype,
                                                                     device=means.device)
        img = w @ col
        if backgrounds is not None:
            img = img + T_final[:, None] * backgrounds[c]
        alpha = 1 - T_final
        depth = (w @ zz) / alpha.clamp(min=1e-10)
        outs.append(img.reshape(height, width, D))
        depths.append(depth.reshape(height, width, 1))
        alphas.append(alpha.reshape(height, width, 1))
    return torch.stack(outs), torch.stack(depths), torch.stack(alphas), {"backend": "dense"}
