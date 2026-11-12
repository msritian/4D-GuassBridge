"""B1 diagnostic: does ManiSkill's appearance of a part depend on its world orientation (world-fixed lights)?

Rotate a part by theta about a vertical axis AND move the camera by the same theta about the same axis. In the
part's own frame nothing changed (same geometry, same viewpoint), so a colour model stored in the part frame
(CLAUDE.md §7.3: SH with the view direction in the part frame) predicts identical pixels. Any difference comes from
the world-fixed lights. Tested on cubeA (yaw) and on the arm (joint 1 rotates links 1-8 about the base axis).

usage: python scripts/check_lighting.py --out results/lighting  [--step-deg 15]
Writes <out>/lighting.json and <out>/lighting.png (cube and arm crops per angle + per-face colour curves).
"""
import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

from gaussbridge import geometry as G
from gaussbridge.sim.capture import capture_state, part_table
from gaussbridge.sim.envs import Ring

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUBE_XY = (0.12, 0.15)
BASE_XY = (-0.615, 0.0)


def masked_stats(img_a, img_b, mask):
    """Mean abs difference (0-255) and PSNR (dB) of two uint8 images on a boolean mask."""
    a, b = img_a[mask].float(), img_b[mask].float()
    mse = ((a - b) ** 2).mean().item()
    return float((a - b).abs().mean()), float(10 * math.log10(255.0 ** 2 / max(mse, 1e-12)))


def cube_face_colours(cam, part_T, part_idx):
    """Mean RGB of each visible cube face, faces named by the axis in the CUBE frame (+x, -x, +y, -y, +z)."""
    rows, cols = torch.nonzero((cam["seg"] == part_idx) & (cam["depth"] > 0), as_tuple=True)
    X = G.backproject(cam["depth"].double(), cam["viewmat"].double(), cam["K"].double(), rows, cols)
    T = G.se3_inverse(part_T.double())
    x = X @ T[:3, :3].T + T[:3, 3]
    axis = x.abs().argmax(-1)
    positive = x.gather(-1, axis[:, None])[:, 0] > 0
    rgb = cam["rgb"][rows, cols].float()
    out = {}
    for a in range(3):
        for pos in (True, False):
            sel = (axis == a) & (positive == pos)
            if sel.sum() > 20:
                out[f"{'+' if pos else '-'}{'xyz'[a]}"] = rgb[sel].mean(0).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-deg", type=float, default=15.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = round(360 / args.step_deg)
    angles = [k * args.step_deg for k in range(n)]
    cube_ring = Ring("cubering", target=(*CUBE_XY, 0.02), radius=0.3, elevations_deg=(35,), n_azimuth=n,
                     size=128, fovy_deg=25.0)
    arm_ring = Ring("armring", target=(*BASE_XY, 0.45), radius=1.6, elevations_deg=(25,), n_azimuth=n,
                    size=128, fovy_deg=45.0)
    env = gym.make("StackCubeCapture-v1", obs_mode="rgb+depth+segmentation", sim_backend="physx_cpu",
                   render_backend="sapien_cpu", capture_rings=(cube_ring, arm_ring))
    env.reset(seed=0)
    u = env.unwrapped
    parts = part_table(env)
    names = [p.name for p in parts]
    u.cubeB.set_pose(Pose.create_from_pq(p=torch.tensor([[0.35, -0.35, 0.02]])))  # out of the way
    q0 = u.agent.robot.get_qpos().clone()

    cube_imgs, arm_imgs, cube_masks, arm_masks, faces = [], [], [], [], []
    a_idx = names.index("cubeA")
    arm_idx = torch.tensor([names.index(f"panda_link{i}") for i in range(1, 8)]
                           + [names.index(x) for x in ("panda_hand", "panda_leftfinger", "panda_rightfinger",
                                                       "camera_link")])
    for k, deg in enumerate(angles):
        th = math.radians(deg)
        u.cubeA.set_pose(Pose.create_from_pq(p=torch.tensor([[*CUBE_XY, 0.02]]),
                                             q=torch.tensor([[math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]])))
        q = q0.clone()
        q[0, 0] = q0[0, 0] + th if q0[0, 0] + th <= 2.8973 else q0[0, 0] + th - 2 * math.pi  # joint 1 range ±166°
        u.agent.robot.set_qpos(q)
        cams = [f"cubering_e35_a{k:02d}", f"armring_e25_a{k:02d}"]
        s = capture_state(env, parts, cameras=cams)
        c, a = s["cameras"][cams[0]], s["cameras"][cams[1]]
        cube_imgs.append(c["rgb"])
        cube_masks.append(c["seg"] == a_idx)
        arm_imgs.append(a["rgb"])
        arm_masks.append(torch.isin(a["seg"].long(), arm_idx))
        faces.append(cube_face_colours(c, s["part_T"][a_idx], a_idx))
    env.close()

    def compare(imgs, masks, valid):
        rows = []
        for k in range(len(imgs)):
            if not valid[k]:
                continue
            both = masks[0] & masks[k]
            iou = float(both.sum() / (masks[0] | masks[k]).sum())
            mad, psnr = masked_stats(imgs[0], imgs[k], both)
            rows.append({"angle_deg": angles[k], "mask_iou": iou, "mean_abs_diff": mad, "psnr_db": psnr})
        return rows

    # joint 1 cannot reach every angle (±166°); angles that wrap are a different arm pose, skip them for the arm
    arm_valid = [abs(math.radians(d)) + float(q0[0, 0]) <= 2.8973 for d in angles]
    report = {"angles_deg": angles, "cube": compare(cube_imgs, cube_masks, [True] * n),
              "arm": compare(arm_imgs, arm_masks, arm_valid), "cube_face_rgb": faces,
              "note": "identical part-frame geometry and viewpoint at every angle; differences = world-fixed lighting"}
    (out / "lighting.json").write_text(json.dumps(report, indent=1))

    fig = plt.figure(figsize=(16, 7))
    show = list(range(0, n, max(1, n // 8)))[:8]
    for i, k in enumerate(show):
        ax = fig.add_subplot(3, 8, i + 1)
        ax.imshow(cube_imgs[k].numpy())
        ax.set_title(f"cube {angles[k]:.0f}°", fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(3, 8, 9 + i)
        ax.imshow(arm_imgs[k].numpy())
        ax.set_title(f"arm {angles[k]:.0f}°" + ("" if arm_valid[k] else " (skip)"), fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(3, 2, 5)
    for face in ["+x", "-x", "+y", "-y", "+z"]:
        ys = [f[face][0] if face in f else np.nan for f in faces]
        ax.plot(angles, ys, marker=".", label=f"face {face}")
    ax.set_xlabel("cube yaw = camera azimuth (deg)")
    ax.set_ylabel("mean red channel (0-255)")
    ax.legend(fontsize=7, ncol=5)
    ax = fig.add_subplot(3, 2, 6)
    ax.plot([r["angle_deg"] for r in report["cube"]], [r["mean_abs_diff"] for r in report["cube"]], label="cube")
    ax.plot([r["angle_deg"] for r in report["arm"]], [r["mean_abs_diff"] for r in report["arm"]], label="arm")
    ax.set_xlabel("rotation (deg)")
    ax.set_ylabel("mean |pixel diff| vs 0° (0-255)")
    ax.legend(fontsize=8)
    fig.suptitle("Same part-frame geometry and viewpoint, different world orientation: colour change = lighting")
    fig.tight_layout()
    fig.savefig(out / "lighting.png", dpi=110)
    worst = {k: max(report[k], key=lambda r: r["mean_abs_diff"]) for k in ("cube", "arm")}
    print(json.dumps({"worst": worst}, indent=1))


if __name__ == "__main__":
    main()
