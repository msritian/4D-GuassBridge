# GaussBridge — project brief for Claude Code agents

> **Read this whole file before doing anything.** It is the single source of truth for what we are
> building, why, how every stage works, the exact versions/APIs we verified, and who does what.
> The task board is `TASKS.md` (pick a task, put your name on it, follow its acceptance test).
> If you change a decision, convention or interface, update this file in the same commit.

Team (CS771 course project, UW–Madison): **Shivam Mittal (smittal39), Amogh Dixit (asdixit3), Aditya Kumar (akumar323)**.
Plan written 2026-11-05. Assumed schedule: 6 weeks, Mon 2026-11-09 → Fri 2026-12-18 (**confirm the course deadline and adjust §12**).

---

## 0. TL;DR

**Title:** *GaussBridge: Gaussian World Models for Diverse Start-to-Goal 4D Trajectory Generation*

**Problem.** Given multi-view images of a scene at a **start** time and at a later **end** time, generate *many*
different, realistic trajectories of what happened in between. Each trajectory is a complete 3D world at every time
step (Gaussian splats), renderable from any camera with exact labels (RGB, depth, flow, 3D tracks, masks, robot actions).

**Core idea.** Reconstruct the scene **once** as 3D Gaussians split into rigid **parts** (static background, each robot
link, each object), each stored in its own canonical frame. A learned **goal-conditioned motion model** (a diffusion
transformer trained with flow matching) generates the *poses* of the moving parts (robot joint angles + object poses)
for all intermediate steps **jointly**, with start and end **held fixed by inpainting**, so every sample lands
exactly on the end state. The world at step *t* is the same Gaussians with each part moved to its generated pose
(no re-reconstruction, no drift). Sample many times → many diverse trajectories → render → training data.

**v1 testbed (decided).** ManiSkill 3 tabletop, Franka Panda: **StackCube-v1** (primary, 2 objects) and
**PickCube-v1** (fast dev task). Training motions come from a randomised motion planner. Downstream proof:
ManiSkill's **Diffusion Policy** baseline trained on real demos vs. real + our generated demos, success rate in sim.

**What is novel (claims to defend).**
1. First *goal-conditioned* Gaussian world model: existing Gaussian world models (ManiGaussian, GWM, 4DGS-WAM) only
   predict forward; goal-conditioned world models (Act2Goal) work in 2D video; 4D inbetweening (In-2-4D) is single
   object, single output.
2. A *learned, stochastic* motion prior conditioned on observed start and end: recent Gaussian demo-synthesis
   systems (RoboSplat 2025, single-scan synthesis 2609.21112, robot-aligned 3DGS 2610.35626) use planners or
   equivariant edits of an existing demo, not a generative model that samples diverse valid motions.
3. Exact endpoints + explicit 3D state ⇒ free dense labels and physically checkable trajectories.

---

## 1. Ground rules for agents

1. **Never guess an API.** Versions below are pinned and were checked against source. If something disagrees with
   this file, read the installed package source (`python -c "import x, inspect; print(inspect.getsource(...))"`),
   fix the code, and update this file.
2. **Conventions in §4 are law** (quaternion order, frames, units, camera convention). Most silent bugs in this
   project will be frame/convention bugs. Every geometry function gets a unit test.
3. Small, tested steps. Every task in `TASKS.md` has an acceptance test; a task is done only when it passes.
4. CPU-first where possible: everything except Gaussian rendering/training and model training runs on a laptop or
   CHTC CPU node (ManiSkill runs headless on CPU with software Vulkan, §3.3).
5. Reproducibility: Hydra configs, fixed seeds, git commit hash saved with every artifact, W&B logging.
6. Keep `TASKS.md` current: owner, status, link to PR/commit, measured numbers.
7. Don't rewrite or delete teammates' work without asking in the task's thread.

---

## 2. System overview (end to end)

```
            ┌────────────────────── trained ONCE per task (simulation) ───────────────────────┐
            │  (A) randomised motion planner ──► reference trajectories (h5)  ──► Episodes     │
            │                                                     │                            │
            │  (B) asset captures (random robot/object configs) ──► articulated 3DGS ──► PartSplats
            │                                                     │                            │
            │  (C) motion model (DiT + flow matching, inpainted endpoints) ◄── Episodes + shape tokens
            └──────────────────────────────────────────────────────────────────────────────────┘
 per scene: start capture + end capture ─► part poses at start/end (known in sim / registered in real)
          ─► sample S trajectories (guidance: collisions, grasp consistency) ─► project + filter + pick diverse
          ─► compose Gaussians per step ─► render any camera (RGB, depth, labels, flow, tracks)
          ─► export as ManiSkill-format demos ─► train Diffusion Policy ─► evaluate in sim
```

Workstreams (one owner each, see §12): **A = Simulation & data** (planner data, extraction, captures, DP export,
policy experiments) · **B = Gaussians & rendering** (reconstruction, composition, renderer, labels) ·
**C = Motion model & evaluation** (dataset, model, sampler, guidance, baselines, metrics).

---

## 3. Tech stack (pinned, verified 2026-11-05)

### 3.1 Versions

| Component | Version | Notes (verified) |
|---|---|---|
| Python | **3.11** | `mplib 0.1.1` (required by ManiSkill) ships wheels only for cp37–cp312; sapien 3.0.3 has cp310–cp314 |
| PyTorch | 2.5.x, CUDA 12.x build | CPU build fine for data work |
| ManiSkill | **mani_skill==3.0.1** (latest on PyPI) | pulls sapien>=3.0.3, mplib==0.1.1, **pytorch_kinematics==0.7.6** (pinned by ManiSkill; do not upgrade) |
| gsplat | **1.5.3** (latest PyPI release) | PyPI wheel is pure Python; CUDA kernels JIT-compile on first use → needs `nvcc` (use a `-devel` CUDA image). GitHub `main` is 1.6.0-dev with a *different* `rasterization` signature — don't read main's docs |
| pytorch_kinematics | 0.7.6 | differentiable FK from URDF |
| h5py, numpy<2 | — | ManiSkill trajectories |
| hydra-core, wandb | — | configs, logging |
| SAM 2 / SAM 3 | facebookresearch/sam2, facebookresearch/sam3 | only for real captures (stretch) |
| VGGT or COLMAP | — | only for real captures (stretch) |
| LPIPS (`lpips`), torchmetrics | — | render metrics |

### 3.2 Install (local / container)

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate      # or conda create -n gb python=3.11
pip install torch==2.5.1                                  # CUDA 12.4 wheel from PyPI
pip install mani_skill==3.0.1 "numpy<2" h5py
pip install gsplat==1.5.3 ninja jaxtyping rich            # first CUDA call compiles kernels (~5-10 min)
pip install hydra-core wandb lpips torchmetrics trimesh transforms3d
export MS_ASSET_DIR=$HOME/.maniskill/data MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
python -m mani_skill.examples.demo_random_action -e StackCube-v1   # smoke test (needs Vulkan)
```

### 3.3 Vulkan (ManiSkill rendering) — important

* SAPIEN needs **a Vulkan device even with `render_backend="none"`** (URDF material loading fails with
  `RuntimeError: failed to find a rendering device`). Verified.
* **GPU nodes:** install `libvulkan1` and the NVIDIA ICD files (copy from ManiSkill's `docker/` folder):
  `/usr/share/vulkan/icd.d/nvidia_icd.json`, `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`,
  `/etc/vulkan/implicit_layer.d/nvidia_layers.json` (needed on A100). Container env: `NVIDIA_DRIVER_CAPABILITIES=all`.
* **CPU-only nodes / laptops:** `apt-get install mesa-vulkan-drivers libvulkan1` (lavapipe), then
  `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json` and create envs with
  `render_backend="sapien_cpu"`. Verified: motion planning, state extraction and small captures work this way.
  Use this for all planner data generation on CHTC **CPU** jobs (no GPU queue wait).
  Needs a recent Mesa: Debian bookworm's Mesa 22.3 is rejected by SAPIEN ("Failed to find a supported physical
  device"); Debian trixie (Mesa 25) works. The ICD file name is distro-specific (`lvp_icd.json` on trixie,
  `lvp_icd.x86_64.json` on bookworm). Ready-made image: `docker/Dockerfile.cpu` (verified 2026-11-06).
* **Macs:** `mplib` (the planner backend) has Linux x86_64 wheels only, so the motion planner cannot run natively
  on macOS. Run `docker/Dockerfile.cpu` with `--platform linux/amd64` (works on Apple Silicon under emulation).

### 3.4 Dockerfile (one image for everything)

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04          # CHTC recommends nvidia/cuda tags starting "12.1.1-devel"
ENV NVIDIA_DRIVER_CAPABILITIES=all DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip git \
    libvulkan1 vulkan-tools mesa-vulkan-drivers libegl1 libglvnd-dev libxext6 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN python3.11 -m venv /opt/gb && /opt/gb/bin/pip install torch==2.5.1 mani_skill==3.0.1 "numpy<2" h5py \
    gsplat==1.5.3 ninja jaxtyping rich hydra-core wandb lpips torchmetrics trimesh transforms3d
ENV PATH=/opt/gb/bin:$PATH TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
# Compile gsplat's CUDA kernels at build time (no GPU needed when TORCH_CUDA_ARCH_LIST is set), so jobs don't
# JIT-compile on every start. This replaces the PyPI gsplat installed above:
RUN pip uninstall -y gsplat && pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
```
The torch 2.5.1 PyPI wheel bundles its own CUDA 12.4 runtime; nvcc 12.1 from the base image is only used to
compile gsplat (same CUDA major version). If the build complains about the version mismatch, switch the base image
to `12.4.1-devel-ubuntu22.04`.
JSON contents for the three NVIDIA files (from ManiSkill `docker/`):
`nvidia_icd.json` = `{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.155"}}`;
`10_nvidia.json` = `{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}`;
`nvidia_layers.json` = the `VK_LAYER_NV_optimus` layer file in ManiSkill `docker/nvidia_layers.json`.
Push with a **new tag per change** (`<dockerhub_user>/gaussbridge:0.1`, `0.2`, …). Never use `latest` (CHTC caches).
Recommended: build from source with the arch list so jobs don't JIT-compile gsplat each time.

### 3.5 Compute: CHTC (UW–Madison) + Colab

Verified from CHTC docs (2026):
* Submit file basics: `universe = container`, `container_image = docker://<user>/gaussbridge:0.3`,
  `request_gpus = 1`, `+WantGPULab = true`, `+GPUJobLength = "short"` (12 h) | `"medium"` (24 h, default) | `"long"` (7 days),
  `gpus_minimum_memory = 40000` (MB), `gpus_minimum_capability = 8.0`, `+is_resumable = true` (backfill; jobs < 4–6 h or checkpointing).
* Per-user GPU limits: short ≤ 2/3 of GPU Lab GPUs, medium ≤ 1/3, long ≤ 4 GPUs. Three accounts = 3× quota.
* GPU Lab (open capacity): H200 141 GB, H100 80 GB, A100 80/40 GB, L40/L40S 44 GB, RTX PRO 6000 94 GB, … L40S are
  plentiful and fine for rendering and model training.
* `condor_ssh_to_job` is **disabled on shared GPU machines** (since 2026-04-22): debug with
  `condor_submit -i job.sub` (interactive) and watch logs with `condor_tail`.
* Data: `/staging/<letter>/<netid>` (default 100 GB / 1000 files). Input 1–30 GB:
  `transfer_input_files = osdf:///chtc/staging/s/smittal39/file.tar`; 30–100 GB: `file:///staging/...`; add
  `Requirements = (Target.HasCHTCStaging == true)`. Output: `transfer_output_remaps = "out.tar = file:///staging/s/smittal39/out.tar"`.
  **Few big files** (tar shards), never thousands of small files. Upload/download via `transfer.chtc.wisc.edu`.
  Submit only from `/home`. Ask CHTC for a **group staging dir** so all three accounts share data.
* CPU jobs (planner data generation) start much faster than GPU jobs: use them.
* Colab Pro: notebooks for visualisation and quick GPU debugging only (sessions die; no long training).

Template GPU submit file (`chtc/train.sub`):
```
universe              = container
container_image       = docker://<user>/gaussbridge:0.3
executable            = chtc/run.sh
arguments             = python scripts/train_motion.py +experiment=stackcube_v1 resume=auto
transfer_input_files  = osdf:///chtc/staging/s/smittal39/gb/episodes_stackcube.tar, chtc/run.sh
transfer_output_files = ckpt.tar
transfer_output_remaps = "ckpt.tar = file:///staging/s/smittal39/gb/ckpt_$(Cluster).tar"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
request_gpus = 1
+WantGPULab = true
+GPUJobLength = "short"
gpus_minimum_memory = 40000
+is_resumable = true
request_cpus = 8
request_memory = 48GB
request_disk = 60GB
Requirements = (Target.HasCHTCStaging == true)
log = logs/$(Cluster).log
output = logs/$(Cluster).$(Process).out
error = logs/$(Cluster).$(Process).err
queue 1
```
`chtc/run.sh` clones the repo at a pinned commit (or ships a tarball), untars inputs, runs the command, tars outputs.
Checkpoint every ≤ 1 h so short/resumable jobs can be resumed.

---

## 4. Conventions (do not change)

* **Units:** metres, radians, seconds. **World frame:** ManiSkill world, z up, table top at z = 0, workspace centred
  near the origin, Panda base at `(-0.615, 0, 0)` with identity rotation (both tasks).
* **Quaternions:** `(w, x, y, z)` (SAPIEN/ManiSkill and gsplat agree). **Pose7** = `(px,py,pz,qw,qx,qy,qz)` =
  ManiSkill `Pose.raw_pose`. 4×4 `T_world_part` maps part-local → world.
* **6D rotation** = first two *columns* of R, `[R[:,0], R[:,1]]`, decoded by Gram–Schmidt (Zhou et al. 2019).
* **Cameras:** OpenCV convention (x right, y down, z forward). `viewmat` = world→camera 4×4 (what gsplat wants).
  ManiSkill gives `sensor_param[cam]["extrinsic_cv"]` as **[3,4]** (pad to 4×4) and `intrinsic_cv` [3,3].
  `CameraConfig.fov` is the **vertical** FoV (`set_fovy`). Depth from ManiSkill is uint16 **millimetres** (0 = invalid).
  Intrinsics: `f = (H/2)/tan(fovy/2)`, principal point `(W/2, H/2)`, so pixel j covers u ∈ [j, j+1): index images
  with `floor(uv)` (verified against ManiSkill `intrinsic_cv` and segmentation, 2026-11-06).
* **SAPIEN camera/link frame → OpenCV:** `R_cv_from_sapien = [[0,-1,0],[0,0,-1],[1,0,0]]`;
  `viewmat = inv(T_world_cam_sapien · diag(R_cv_from_sapienᵀ, 1))`. Verified against ManiSkill's hand camera
  extrinsic to 1e-7.
* **Robot joint order** (ManiSkill active joints = pytorch_kinematics order, verified):
  `panda_joint1..7, panda_finger_joint1, panda_finger_joint2` → `qpos` [9].
* **Time:** control rate 20 Hz (sim 100 Hz). On disk: physical time. In the model: **N = 128 normalised steps** + a
  scalar log-duration.
* **Gripper command:** +1 open, −1 close (ManiSkill normalised gripper action).

---

## 5. Verified facts about ManiSkill 3.0.1 (from source + local runs)

Tasks
* **StackCube-v1**: default robot `panda_wristcam` (URDF `robots/panda/panda_v3.urdf`, adds `camera_base_link`,
  `camera_link`). Cameras: `base_camera` 128×128, fov π/2, `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`, and
  `hand_camera` 128×128, fov π/2, mounted on `camera_link` with identity local pose. Objects `cubeA` (red),
  `cubeB` (green), half-size 0.02. `extra` obs (rgb mode): `tcp_pose`. Success: A on B (xy offset ≤ ‖half_xy‖+5 mm,
  z offset 0.04±5 mm), A static, A not grasped. `robot_init_qpos_noise=0.02`.
* **PickCube-v1**: default robot `panda` (`panda_v2.urdf`), only `base_camera` (same pose as above).
  Object `cube` (half 0.02), `goal_site` (hidden marker, appears in env_states as an actor).
  `extra` obs (rgb): `is_grasped`, `tcp_pose`, `goal_pos`. Success: cube within 0.025 m of goal **and** robot static.
  `max_episode_steps=50` by default (too short for our demos; override). **StackCube-v1 also defaults to 50.**
* Shadows are **off** by default (`enable_shadow=False`) → no baked-shadow problem in sim.
* Panda `rest` keyframe qpos `[0, π/8, 0, −5π/8, 0, 3π/4, π/4, 0.04, 0.04]`. Joint limits (rad):
  low `[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]`, high `[2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973]`;
  fingers `[0, 0.04]` m each. TCP link: `panda_hand_tcp`.
* `pd_joint_pos` action = 8-D: 7 **absolute** arm joint targets (not normalised) + gripper ∈ [−1,1]
  (normalised, maps to finger target [−0.01, 0.04]).

Trajectory files (`trajectory.{obs_mode}.{control_mode}.{backend}.h5` + `.json`)
* Per `traj_{i}`: `actions [L,8]`, `terminated/truncated/success [L]`, `env_states/actors/<name> [L+1,13]`
  (= p3, q4, linvel3, angvel3), `env_states/articulations/<robot_uid> [L+1,31]`
  (= root p3 q4 v3 w3, qpos9, qvel9), `obs` only if recorded with an obs mode.
* JSON: `env_info{env_id, env_kwargs}`, `episodes[{episode_id, episode_seed, reset_kwargs{seed}, control_mode,
  elapsed_steps, success}]`. `RecordEpisode.flush_trajectory(save=False)` discards; `episode_id` increments only on save.
* Replay/convert: `python -m mani_skill.trajectory.replay_trajectory --traj-path X.h5 -o rgb -c pd_joint_pos
  --save-traj --use-env-states -b physx_cpu` (adds observations). `--use-env-states` reproduces the scene exactly.

Motion planner (`mani_skill/examples/motionplanning/panda/`)
* `PandaArmMotionPlanningSolver(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info,
  joint_vel_limits=0.9, joint_acc_limits=0.9)`; methods `move_to_pose_with_screw(pose, dry_run=False, refine_steps=0)`,
  `move_to_pose_with_RRTConnect(...)`, `open_gripper()`, `close_gripper()`, `close()`. Failure returns `-1`.
  Paths are time-parameterised at `env.control_timestep` (0.05 s) with the scaled joint vel/acc limits.
  Env must use `control_mode="pd_joint_pos"`. Official solutions exist for PickCube, StackCube, PushCube, PullCube,
  PlaceSphere, PegInsertionSide, PlugCharger, LiftPegUpright, StackPyramid, DrawTriangle, PullCubeTool, DrawSVG.
* Grasp construction: `compute_grasp_info_by_obb(get_actor_obb(actor), approaching=[0,0,-1], target_closing=tcp_y, depth=0.025)`
  then `agent.build_grasp_pose(approaching, closing, center)`.
* Measured: official StackCube demos are 94–108 steps (~5 s); our randomised demos 150–260 steps (7–13 s);
  planning ≈ 0.5 s per trajectory on one CPU core; ~80–100 % of attempts succeed.

Grasp ("attached") facts from demos
* While holding a 4 cm cube the finger joint reads ≈ 0.0183 m and the cube centre stays 1–3 mm from the TCP.
* Rule used everywhere (sim data and generated data): `attached_k(t) = (gripper_cmd(t) < 0) and (finger(t) < 0.039)
  and ‖p_obj_k(t) − p_tcp(t)‖ < 0.02`.
* `agent.is_grasping(obj)` uses contact forces, which are stale after `set_state_dict` without a physics step →
  **don't** use it on replayed states; use the geometric rule.

Kinematics
* `pytorch_kinematics.build_chain_from_urdf(open(urdf,'rb').read())`; `T_world_link = T_world_base @ FK(qpos)[link].get_matrix()`
  matches simulator link poses to < 1e-6 m for all 15 links (verified). FK accepts `[B, 9]`; flatten extra batch dims.

Segmentation (StackCube example) — `env.unwrapped.segmentation_id_map`: 0 = sky/empty, 1–13 = Panda links
(`panda_link0..8, panda_hand, panda_hand_tcp, panda_leftfinger, panda_rightfinger`), 14 `camera_base_link`,
15 `camera_link`, 16 `table-workspace`, 17 `ground`, 18 `cubeA`, 19 `cubeB`. Build the id→part map from this dict;
never hard-code ids. `panda_hand_tcp` and `camera_base_link` have no visual geometry.

Round-trip check (verified): planner episode → 128-step motion tensor → back to 20 Hz → replay `pd_joint_pos`
actions in sim: **21/21 successes**, max joint error 0.009 rad, max object end error 3.6 mm. So N = 128 is enough.
Re-verified 2026-11-06 with `trajectory.py` on 12 StackCube + 12 PickCube randomised demos (+10 hold steps): 24/24,
joint error ≤ 0.0015 rad, object end error ≤ 3.9 mm (StackCube) / 5.6 mm (PickCube; replaying measured-qpos actions
without resampling already gives up to 3.3 mm).

Custom cameras: subclass the task and override the `_default_sensor_configs` **property**, keep the policy camera
first, register with `@register_env("StackCubeCapture-v1", max_episode_steps=400)`. After `robot.set_qpos(...)` or
`actor.set_pose(...)`, `env.unwrapped.get_obs()` re-renders with the new state (verified).

Diffusion Policy baseline (`examples/baselines/diffusion_policy/`, clone ManiSkill repo; `pip install -e .` there)
* `train_rgbd.py --env-id StackCube-v1 --demo-path X.h5 --control-mode pd_joint_pos --sim-backend physx_cpu
  --num-demos N --max_episode_steps 400 --total_iters 100000 --obs-mode rgb` (defaults `obs_horizon=2`,
  `act_horizon=8`, `pred_horizon=16`; images assumed 128×128).
* Loader reads h5 `obs` and `actions`, reorders `obs` keys to the env observation space, so **the h5 must contain
  `obs/agent/{qpos,qvel}`, `obs/extra/<task keys>`, `obs/sensor_data/<cam>/rgb`, `obs/sensor_param/<cam>/{extrinsic_cv,
  cam2world_gl,intrinsic_cv}`** for every camera in the env. State = concat(agent values, extra values);
  images of all cameras are concatenated on channels (StackCube: base + hand → 6 channels).
* The dataset is loaded **entirely into GPU memory**: 128×128×3 uint8 × 2 cams × ~200 steps ≈ 20 MB per episode
  → keep total demos ≲ 600 per run on a 40 GB GPU, or use 80 GB GPUs.
* README advice: set `--max_episode_steps` ≈ 2× mean demo length.

gsplat 1.5.3 facts
* `rasterization(means[N,3], quats[N,4] (wxyz, unnormalised ok), scales[N,3] (activated), opacities[N] (activated),
  colors, viewmats[C,4,4], Ks[C,3,3], width, height, near_plane=0.01, far_plane=1e10, sh_degree=None, packed=True,
  backgrounds=[C,D], render_mode in {"RGB","D","ED","RGB+D","RGB+ED"}, rasterize_mode, ...) -> (render [C,H,W,D(+1)],
  alpha [C,H,W,1], meta)`. With `sh_degree=None`, `colors` are post-activation `[N,D]` or `[C,N,D]` with **any D**
  (extra channels for labels/flow; chunked by `channel_chunk=32`).
* Internally with SH: colour = `clamp_min(SH(dirs) + 0.5, 0)`, dirs = means − campos. If you evaluate SH yourself
  (we do, see §7.3) add 0.5 and clamp. `rgb_to_sh(rgb) = (rgb − 0.5) / 0.28209479177387814`.
  Helper: `gsplat.spherical_harmonics(degree, dirs, coeffs)` (CUDA) or `gsplat.cuda._torch_impl._spherical_harmonics` (CPU).
* Trainer defaults (examples/simple_trainer.py): lr means 1.6e-4·scene_scale, scales 5e-3, quats 1e-3, opacities 5e-2,
  sh0 2.5e-3, shN 2.5e-3/20; init opacity 0.1; scales = log(mean 3-NN distance); loss = 0.8·L1 + 0.2·(1−SSIM).
  Params stored pre-activation (`scales` log, `opacities` logit). One `Adam` per parameter name.
* `DefaultStrategy(refine_start_iter=500, refine_stop_iter=15000, refine_every=100, reset_every=3000, ...)`;
  `check_sanity` compares only **trainable** params with optimizers, so a **non-trainable per-Gaussian tensor**
  (our `part_id`, `nn.Parameter(..., requires_grad=False)`) is carried through duplicate/split/prune automatically.
  Call `strategy.step_pre_backward(params, opts, state, step, meta)` before `loss.backward()` and
  `step_post_backward(..., packed=...)` after the optimiser step. With `packed=False`, meta must contain `means2d`,
  `radii [C,N,2]`, `width`, `height`, `n_cameras`, and the key `gaussian_ids` (None).

---

## 6. Data: generation, formats, splits

### 6.1 Randomised planner demos (Workstream A)
Wrap the official solutions with diversity **knobs**, sampled per trajectory from `rng = default_rng([seed, ref, attempt])`:

| Knob | Range | Effect |
|---|---|---|
| `grasp_yaw_k` | {0,1,2,3} × π/2 extra yaw on the top grasp (cube is 4-fold symmetric); fall back to next reachable k via `dry_run` | different grasps / wrist motions |
| `reach_h` | U[0.03, 0.10] m | pre-grasp height |
| `lift_h` | U[0.06, 0.20] m | lift height |
| `place_h` | 0 (p=.5) or U[0, 0.08] m | extra pre-place waypoint |
| `via` | none (p=.5) or (U±0.08, U±0.08, U[0,0.08]) m mid-transport offset | path shape |
| `speed` | U[0.4, 0.9] | scales joint vel AND acc limits → timing |
| `rrt_transport` | p = 0.15 | RRTConnect instead of screw for transport (fallback to screw) |

StackCube sequence: reach (grasp·Pose([0,0,−reach_h])) → grasp → close → lift (Pose([0,0,lift_h])·grasp) →
[via] → [pre-place] → place at `cubeB.pose·Pose([0,0,0.04])` expressed as TCP offset (as in the official solution)
→ open. PickCube: reach → grasp → close → lift → [via] → goal (`goal_site` position, grasp orientation).
Discard failures (`flush_trajectory(save=False)`); save a sidecar `<shard>.knobs.json` indexed by `episode_id`
with `{seed, ref_idx, attempt, knobs, num_steps}`.

Reference structure: **for one start seed, M references** share the same start state; StackCube/PickCube object end
positions are (near-)identical across refs (A lands on B / cube at goal), robot end differs by grasp choice.

Planned sizes (CPU, ~0.5 s each):
| Split | StackCube | PickCube |
|---|---|---|
| train | 1,500 seeds × 8 refs ≈ 12k | 1,000 × 8 ≈ 8k |
| val | 50 seeds × 8 | 50 × 8 |
| test (metrics) | 100 seeds × **16** refs | 100 × 16 |
| "real" demos for policy | 100 seeds × 1 ref (separate seed range) | same |
Seed ranges must not overlap: train 0–1499, val 10000–10049, test 20000–20099, policy-real 30000–30099,
policy-eval 40000+ (DP eval uses its own seeds; keep ours disjoint). Shard = 50 seeds per CHTC CPU job.

### 6.2 Episode format (`episodes/<env_id>/<split>/<id>.npz`)
| Key | Shape / type | Meaning |
|---|---|---|
| `robot_qpos` | [L+1, 9] f32 | joint state per step (ManiSkill order) |
| `gripper_cmd` | [L+1] f32 | ±1; entry t = command applied after state t (last = repeat) |
| `obj_pose` | [L+1, K, 7] f32 | pose7 of each object, order = `OBJECTS[env_id]` (`["cubeA","cubeB"]`, `["cube"]`) |
| `attached` | [L+1, K] bool | geometric rule §5 |
| `base_pose` | [7] f32 | robot root pose |
| `control_freq` | f32 | 20 |
| `obj_names`, `meta` | JSON strings | env_id, robot_uid, seed, ref_idx, knobs, success, source |
`actions_pd_joint_pos = concat(qpos[1:, :7], gripper_cmd[:-1])` → [L, 8].
Extraction: `robot_qpos = env_states/articulations/<uid>[:, 13:22]`, `base_pose = [..][0, :7]`,
`obj_pose = env_states/actors/<name>[:, :7]`, `gripper_cmd = actions[:, -1]` (+ repeat last).

### 6.3 Motion tensor (model space)
* N = 128 normalised steps (linear resampling for qpos/positions, slerp for rotations, nearest sample for
  gripper/attached; endpoints preserved exactly). E = 1 + K_MAX entities (K_MAX = 3; robot = slot 0). F = 10:
  * robot token: `[q_norm(7) = 2(q−lo)/(hi−lo)−1, finger_norm = finger/0.04·2−1, gripper_cmd, 0]`
  * object token: `[(p − c)/s (3) with c=(0,0,0.1), s=0.3, rot6d (6), attached·2−1]`
* duration: `log_dur = (log(L/20) − μ)/σ`; recompute μ, σ from the training set (expected μ ≈ log 8 s, σ ≈ 0.35).
* `ent_mask [E]`, `known [N,E]`: start (n=0) and end (n=N−1) of every valid entity. Modes:
  *full goal* (robot end known) and *object-only goal* (robot end unknown, `known[-1,0]=False`).
* Decode for physics: qpos from q_norm, finger, object p/R; back to 20 Hz with L = round(exp(dur)·20).

### 6.4 Scene conditioning
* **Shape tokens:** per object, P = 512 canonical-frame points `[xyz, rgb]` = centres + base colour of that object's
  Gaussians (opacity > 0.5). Before Gaussians exist, use analytic box-surface samples (half 0.02, red/green).
* **Height map:** 64×64 over x,y ∈ [−0.4, 0.4] m of the **static** background (table) from background Gaussians or
  sim; constant (flat) for v1 tasks, kept for generality.
* Endpoint-noise augmentation on the *known copy* only: objects 5 mm / 2°, joints 0.01 rad.

---

## 7. Gaussian world representation (Workstream B)

### 7.1 Parts and the `PartSplats` container
Every Gaussian has `part_id` (non-trainable) and lives in its part's canonical frame. Parts = background
(`table-workspace`, `ground`), robot links (all visual links incl. `camera_link`), objects. Stored like gsplat
(`means, scales(log), quats, opacities(logit), sh0 [N,1,3], shN [N,K−1,3]`), SH degree 1 by default.
A state = dict `{part: T_world_part}`; robot link poses come from FK(qpos) (§5), object poses from the trajectory,
background = its captured pose.

### 7.2 Articulated multi-configuration reconstruction (one optimiser for every case)
A **configuration** = one static arrangement captured from C cameras: `{rgb [C,H,W,3], depth [C,H,W] m, seg [C,H,W]
part index (−1 ignore), viewmats [C,4,4], Ks [C,3,3], part_T [P,4,4]}`. Optimise one canonical Gaussian set per part:
for each sampled training view, pose all parts with that configuration's `part_T`, render, compute loss.
* **Init:** back-project valid depth pixels of each part into its canonical frame (`x_can = T_part⁻¹ x_world`),
  subsample ≤ 20k points per part, colour from RGB, scale = mean 3-NN distance (clamp 0.1 mm–5 cm), opacity 0.1.
  Ignore ground pixels farther than 3 m.
* **Loss:** 0.8·L1 + 0.2·(1−SSIM) + 0.1·|depth − D_gt| (valid pixels) + 0.05·CE(part label) where part labels are
  rendered as one-hot feature channels normalised by alpha (keeps each part's Gaussians on its own pixels).
  Learnable background colour (sigmoid of a 3-vector).
* Densify with `DefaultStrategy` (`scene_scale=1.0`, refine 500→5000 every 100, opacity reset 3000), 7k iterations,
  1 view per iteration. Expected: a few minutes on one A100/L40S.
* Optional `refine_poses`: learnable se(3) delta per (config, part), lr 1e-4, config 0 fixed (real captures).

Uses:
1. **Asset mode (sim, built once per task):** R = 8 configurations: robot = rest + U(±0.6) per arm joint (clipped
   to limits−0.05, reject if any link origin below z = 2 cm), fingers U[0,0.04]; objects 50% resting with random yaw,
   50% floating 8–25 cm with random rotation (so bottoms are seen), ≥ 12 cm from robot links. Cameras: 12 azimuths ×
   elevations {20°,45°,70°} = 36 views, radius 1.1 m around `[-0.15, 0, 0.15]`, 256×256, vfov 60°. Policy cameras of
   two extra configurations = held-out test views. Output: robot links + objects + background `PartSplats`
   (`assets/<env_id>/scene.splats.pt`) and per-object shape points (`assets/<env_id>/<obj>.points.npy`).
2. **Scene mode (the real use case):** only the START and END captures of one episode (2 configurations). Used to
   evaluate how good a reconstruction from two snapshots is. Option: merge the pre-built robot asset (a robot is a
   known, fixed asset in practice) and reconstruct only background + objects.
3. **Real captures (stretch):** objects-only tabletop, phone walk-around before/after; SAM 2/3 masks; VGGT or COLMAP
   poses (mask moving objects); AprilTag for metric scale and table frame; end poses via ICP on Gaussian centres +
   photometric `refine_poses`.

### 7.3 Composition and rendering
For part p with `T_p`: `μ_w = R_p μ + t_p`, `q_w = q(R_p) ⊗ q` (covariance `R_p Σ R_pᵀ`), scales/opacities unchanged.
**Colour:** evaluate SH with the view direction in the part frame, `d_local = R_pᵀ(μ_w − campos)`, add 0.5, clamp,
and pass `[C,N,3]` colours with `sh_degree=None` (gsplat does not rotate SH). Render `render_mode="RGB+ED"`,
`packed=False`, near 0.01, far 10.
Outputs per frame and camera:
* RGB (clamp to [0,1] only when saving), expected depth, alpha;
* part/group labels: one-hot group channels → argmax where alpha > 0.5;
* scene flow: render `(μ_w(t+1) − μ_w(t))` as 3 channels / alpha; optical flow by projecting `x + flow`;
* 3D point tracks: query points = Gaussian centres (subsample per part); position at t = `T_p(t)·μ`; 2D tracks by
  projection; visibility = |projected depth − rendered depth| < 1 cm;
* cameras: `base_camera` (fixed, from `sensor_param`), `hand_camera` = `viewmat(FK(qpos_t)[camera_link])` (§4),
  plus optional jittered/novel cameras for augmentation.

### 7.4 Validation of the representation
On held-out configurations and on planner trajectories (where ManiSkill can render ground truth via
`--use-env-states` replay), report PSNR / SSIM / LPIPS of our renders vs ManiSkill renders at `base_camera` and
`hand_camera`. Targets: ≥ 28 dB PSNR at 128×128 for asset mode; report scene mode honestly.

---

## 8. Motion model (Workstream C)

### 8.1 Architecture (`MotionDiT`)
* Tokens: one per (step n, entity e): 128 × 4 = 512, plus 1 duration token. Token embedding =
  `Linear(F)(x_τ) + Linear(F+1)([x_known·known, known]) + entity-slot embedding + sinusoidal(n)` + (objects) shape token.
* Context (cross-attention): a learned null token, object shape tokens (PointNet: per-point MLP 6→128→256, concat
  max & mean pool → MLP → d), height-map tokens (4 stride-2 convs → 4×4 = 16 tokens).
* 8 blocks, d = 384, 6 heads, MLP ratio 4: self-attention over all motion tokens (+ padding mask for missing
  objects), cross-attention to context, MLP; flow time τ through **adaLN-Zero** (DiT). Output heads zero-initialised:
  per-token velocity [F] and duration velocity. ≈ 30 M parameters. (If attention cost matters: factorise into
  time-attention per entity + entity-attention per step; at 513 tokens it does not.)

### 8.2 Training (flow matching with inpainting)
* `τ ~ sigmoid(N(0,1))` (logit-normal), `x_τ = (1−τ)x₀ + τε`, target `v = ε − x₀`; τ = 0 data, τ = 1 noise.
* **Known entries are never noised:** `x_τ[known] = x_known` at every τ (train and test); the loss masks them and
  padded entities. Duration is generated (loss weight 0.1).
* Conditioning dropout: whole end unknown with p = 0.1 (gives the forward-only baseline for free), robot end unknown
  with p = 0.3 (object-only goal mode).
* AdamW lr 2e-4, wd 0.01, 2k warmup + cosine, batch 256, 150k steps, bf16, grad-clip 1.0, EMA 0.999 (sample from EMA).
  Checkpoint every 2k steps (resumable CHTC jobs). Expect ~6–8 h on one A100/L40S.
* Monitor: val loss; every 10k steps sample 16 per val pair and log endpoint error, collision rate, coverage, a video.

### 8.3 Sampling
* 40 Euler steps τ: 1 → 0; after every step re-impose known entries and zero padded entities.
* **Guidance** (for τ < 0.6): predicted clean sample `x̂₀ = x_τ − τv`; energy on decoded physics; step
  `x_τ ← x_τ − λ ∇_{x̂₀}E` (DPS-style, no backprop through the network). E = collision energy + attachment energy.
  Tune λ ∈ {0, 0.25, 0.5, 1, 2} on val.
  * **Collision:** each part approximated by ~16 k-means spheres fitted to its canonical Gaussian centres
    (radius = max member distance). Penetration² summed over robot-link↔object, object↔object (skip pairs resting
    on each other at start/end), everything↔table plane (z = 0, skip `panda_link0`); skip gripper links
    (`panda_hand`, fingers, tcp) ↔ an object within 3 cm of the TCP (grasping). Robot link poses via
    pytorch_kinematics FK (differentiable).
  * **Attachment:** while `att` is on, the object's TCP-relative position is constant:
    `Σ σ(4·att_n)σ(4·att_{n+1})‖r_{n+1} − r_n‖²`, `r = (T_tcp⁻¹ T_obj).t`.
* **Grasp-consistency projection (post-hoc, hard):** per object take the longest `att>0` interval [n₀,n₁];
  `G = T_tcp(n₀)⁻¹ T_obj(n₀)`; set `T_obj(n) = T_tcp(n)·G` on [n₀,n₁], hold start pose before n₀ and release pose
  after n₁, final step = known end. **Reject** if release error ‖T_obj(n₁) − T_end‖ > 1 cm, or an object whose end
  differs from start by > 1 cm was never held. (Objects only move when carried — true for all v1 tasks.)
* **Filter:** max penetration > 5 mm → reject; jerk above the 99th percentile of training data → reject;
  joint limits violated → reject. Log rejection rates (they are metrics).
* **Diversity selection:** sample 64–200 candidates per pair, keep k (e.g. 16) by farthest-point selection on
  the track feature (TCP + object positions at 8 evenly spaced steps).
* Output → Episode at 20 Hz → optional physics replay check (`pd_joint_pos` from the start state).

### 8.4 Baselines (all evaluated identically)
1. **Straight line:** joints/finger linear, gripper = start value then end value at N/2, objects linear + slerp,
   never attached. Reaches endpoints, ignores grasping/obstacles.
2. **Forward-only:** our model with the end masked (dropout-trained); endpoints not enforced.
3. **Step-by-step (autoregressive):** same architecture trained on windows: 8 known past steps + 16 to predict +
   the end state as extra known tokens (with a "goal" time embedding); roll out to 128, end forced in the last
   window. Shows drift and end-correction jerk.
4. **Planner (oracle, upper bound):** the randomised planner itself; needs full simulator state and meshes.

---

## 9. Exporting generated data for Diffusion Policy (Workstream A + B)

Write a ManiSkill-format h5 (`traj_i` groups) + JSON, control mode `pd_joint_pos`:
* `actions [L,8]` = Episode actions; `terminated`, `truncated` = False, `success` = True at the last step.
* `obs/agent/qpos [L+1,9]`; `obs/agent/qvel [L+1,9]` = central finite differences × 20 Hz (note: a small gap vs. sim
  velocities; ablation: drop qvel from the state for all arms).
* `obs/extra`: StackCube `tcp_pose [L+1,7]` (FK); PickCube `is_grasped [L+1]` (attached), `tcp_pose`, `goal_pos [L+1,3]`.
* `obs/sensor_data/<cam>/rgb [L+1,128,128,3] uint8` rendered from Gaussians for every policy camera.
* `obs/sensor_param/<cam>/{extrinsic_cv [L+1,3,4], cam2world_gl [L+1,4,4], intrinsic_cv [L+1,3,3]}` (fixed camera:
  copy from sim; hand camera: from FK; cam2world_gl = inverse of the extrinsic with OpenGL axes — DP does not read
  them but key presence is required).
* JSON: `env_info` copied from a real demo file of the same task (env kwargs!), `episodes` with `episode_id`,
  `reset_kwargs.seed`, `control_mode`, `elapsed_steps`.
**Acceptance test:** DP's `load_demo_dataset` + `reorder_keys` load a mixed real+generated file without error and
the state/rgb shapes equal those of a real demo replayed with `-o rgb`.
To mix datasets, concatenate groups into one h5 (renumber `traj_i`, merge JSON) — DP takes one `--demo-path`.

---

## 10. Evaluation protocol

### 10.1 Motion quality (test split: 100 seeds × 16 refs per task)
For each test seed: condition on start + the (shared) object end (object-only goal) and, separately, on one
reference's full end (full goal). Draw S = 32 samples after filtering. Report mean ± std over seeds:
| Metric | Definition |
|---|---|
| Endpoint error | object pos (cm) and rot (deg, min over cube symmetry group, 24 rotations); TCP pos (cm) |
| Penetration rate / max depth | % trajectories with any penetration > 5 mm; mean max depth (mm), sphere proxies |
| Smoothness | mean squared jerk of TCP + object positions (m²/s⁶), ratio to reference distribution |
| Coverage / precision @3 cm | refs with a sample within ADE 3 cm / samples within 3 cm of some ref |
| minADE@S, APD | best-of-S ADE to a reference; mean pairwise ADE among samples (diversity) |
| Physics replay success | % of samples whose `pd_joint_pos` actions solve the task in ManiSkill from the start state |
| Rejection rate | % candidates removed by projection/filters |
| Speed | samples/s on one GPU |
ADE = mean over steps and tracked points (TCP + objects) of Euclidean distance, on the 128-step grid.

### 10.2 Rendering quality
PSNR/SSIM/LPIPS vs ManiSkill renders (policy cameras) for: asset mode, scene mode (start+end only), scene mode +
robot asset. 20 test episodes, all frames.

### 10.3 Downstream (the headline result)
ManiSkill Diffusion Policy (`train_rgbd.py`, rgb, `pd_joint_pos`, `max_episode_steps 400`, 3 training seeds,
evaluate 100 episodes each on disjoint seeds). StackCube primary, PickCube secondary.
| Arm | Training data |
|---|---|
| R-N | N ∈ {10, 25, 50} real planner demos, ManiSkill-rendered |
| R-N + G-M (ours) | + M ∈ {100, 300} generated demos (new start seeds, object-only goal from task definition, Gaussian-rendered) |
| R-N + G-M filtered | same, only physics-replay-successful samples |
| R-N + P-M (oracle motion) | + M planner demos rendered by our Gaussians (same renderer as ours: isolates motion quality) |
| R-N + G-M sim-rendered | + our motions rendered by ManiSkill via env-state replay (isolates the appearance gap) |
| R-N + SL-M | + straight-line motions (sanity: bad motion should not help) |
Generated starts: `env.reset(seed)` for new seeds gives start poses; end = task goal (StackCube: cubeA at
cubeB + 4 cm, yaw = start yaw; PickCube: cube at `goal_site`), robot end unknown. No planner is used.
**Success criterion:** R-N + G-M beats R-N, approaches R-N + P-M.

### 10.4 Ablations
no collision guidance · no attachment projection · no shape tokens · no endpoint-noise augmentation ·
whole-sequence vs step-by-step · training data 2k/6k/12k · N = 64 vs 128 · guidance scale sweep.

---

## 11. Planned repository layout and interfaces

```
gaussbridge/
  CLAUDE.md  TASKS.md  README.md  pyproject.toml
  docker/ (Dockerfile, nvidia_icd.json, nvidia_layers.json, 10_nvidia.json)
  chtc/   (run.sh, gen_demos.sub, capture.sub, recon.sub, train.sub, sample_render.sub, dp_train.sub, interactive.sub)
  configs/ (hydra: data/, model/, train/, sample/, recon/, experiment/)
  src/gaussbridge/
    geometry.py          quats (wxyz), 6D, pose7<->4x4, se3_inverse, transform_gaussians, camera conversions
    trajectory.py        Episode (save/load), resampling, episode_to_motion / motion_to_episode, known_mask, decode_motion
    sim/envs.py          *Capture-v1 envs (capture ring + policy cams), task tables (objects, cameras, robot uid)
    sim/planner_gen.py   Knobs, sample_knobs, solve_stack_cube, solve_pick_cube
    sim/extract.py       ManiSkill h5 -> Episodes, attached rule
    sim/kinematics.py    PandaFK (link poses, tcp, hand camera viewmat)
    sim/capture.py       part_table (seg id -> part), capture_state, random_configs
    sim/replay.py        physics replay of pd_joint_pos actions
    gs/splats.py         PartSplats (compose, local-frame SH colours, save/load/subset/merge), init_from_points
    gs/recon.py          articulated multi-config reconstruction (+ pose refinement)
    gs/render.py         render (rgb, depth, alpha, labels, scene flow), optical_flow, tracks
    gs/raster.py         gsplat on CUDA; tiny dense PyTorch rasteriser for CPU unit tests
    model/encoders.py    PointSetEncoder, HeightMapEncoder
    model/dit.py         MotionDiT
    model/flow.py        flow_loss, sample (inpainting + guidance)
    model/dataset.py     MotionDataset
    guidance/collision.py  fit_spheres, scene_energy
    guidance/attach.py     project_attachment, attach_energy
    eval/metrics.py      ADE, APD, coverage/precision, endpoint errors (+cube symmetries), jerk
    eval/baselines.py    straight_line (+ AR wrapper)
    export/maniskill_h5.py  generated Episodes + renders -> DP-ready h5/json
  scripts/  gen_demos.py extract_episodes.py capture_assets.py capture_scene.py train_recon.py train_motion.py
            sample.py render_episodes.py export_dp.py merge_h5.py eval_motion.py eval_render.py run_dp.sh
  tests/    unit tests per module (CPU) + tests/gpu/ (run on a GPU node)
  notebooks/ viz_episode, viz_splats, viz_samples
```
Data layout on staging: `gb/raw/<env>/shard_XXX.{h5,json,knobs.json}` → `gb/episodes/<env>/<split>.tar` →
`gb/assets/<env>/` → `gb/ckpt/` → `gb/samples/<run>/` → `gb/dp_data/<arm>.h5` → `gb/dp_runs/`.

---

## 12. Team plan, milestones and gates (details per task in TASKS.md)

Owners (proposal — swap if you prefer): **A = Simulation & data — Aditya**, **B = Gaussians & rendering — Amogh**,
**C = Motion model & evaluation — Shivam**. Writing is shared; C assembles results tables.

| Week | Dates | A (sim & data) | B (Gaussians) | C (model & eval) | Gate (end of week) |
|---|---|---|---|---|---|
| 1 | Nov 9–15 | env + Docker + CHTC; randomised planner; extraction; data shards | capture envs; part table; asset captures; recon v0 | geometry/trajectory utils + tests; dataset; DiT + flow; straight-line baseline | **G1:** 12k StackCube episodes on staging; 1 recon renders; model overfits 10 episodes |
| 2 | Nov 16–22 | test/val splits; physics replay tool; DP baseline reproduced on real demos | asset recon (robot + objects + bg); renderer + policy cams + labels; render-quality eval | full training run v1; sampler; metrics; val dashboard | **G2:** model beats straight line on endpoint+collision; asset renders ≥ 28 dB |
| 3 | Nov 23–29 (Thanksgiving 26th) | DP export + mixed-data loader test | tracks/flow; scene-mode recon (start+end) | guidance + projection + filters; forward-only; AR baseline | **G3:** first full metrics table (§10.1) |
| 4 | Nov 30–Dec 6 | downstream DP runs (all arms, StackCube) | render generated episodes at scale; sim-rendered control | ablations; PickCube | **G4:** downstream table, StackCube |
| 5 | Dec 7–13 | PickCube DP runs; videos | figures; (stretch) real objects-only capture | remaining ablations; final numbers | **G5:** all tables frozen |
| 6 | Dec 14–18 | report + presentation (all) | | | submit |

Cut order if behind: real-world stretch → PickCube downstream → AR baseline → scene-mode recon (keep asset mode) →
fewer DP arms (keep R-N, R-N+G-M, R-N+P-M).

---

## 13. Risks and mitigations

| Risk | Early sign | Mitigation |
|---|---|---|
| Vulkan/rendering broken on CHTC | `vk::createInstanceUnique: ErrorIncompatibleDriver` | NVIDIA ICD files (§3.3); lavapipe on CPU nodes; test day 1 |
| gsplat JIT compile per job | jobs spend minutes compiling | build gsplat from source in the image with `TORCH_CUDA_ARCH_LIST`; or persistent `TORCH_EXTENSIONS_DIR` |
| Frame/convention bugs | renders misaligned, FK off | unit tests vs simulator (FK, hand camera, project points onto seg masks) |
| Generated grasps not physical | low replay success | attachment guidance + projection; filter by replay; report both |
| Gaussian vs ManiSkill appearance gap hurts DP | R+G below R | sim-rendered control arm; mix ratios; colour jitter aug in DP |
| DP OOM (whole dataset on GPU) | CUDA OOM at load | ≤ ~600 episodes per run on 40 GB, or 80 GB GPU |
| Diversity is noise not modes | high APD, low coverage | farthest-point selection; more knob diversity; report coverage/precision |
| "Why not just run the planner?" | reviewer question | planner needs privileged sim state + meshes; ours works from captures; planner reported as oracle |
| Scooped | similar arXiv paper | weekly arXiv check (§14 queries); write early |

---

## 14. Literature survey (what matters for this project)

**Gaussian splatting & tooling**
* Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 2023 — representation.
* Ye et al., *gsplat: An Open-Source Library for Gaussian Splatting*, arXiv 2409.06765 — our rasteriser.

**Gaussian world models (forward prediction, our closest relatives)**
* Lu et al., *ManiGaussian*, ECCV 2024 — dynamic Gaussians for multi-task manipulation.
* Lu et al., *GWM: Towards Scalable Gaussian World Models for Robotic Manipulation*, ICCV 2025.
* Ma, Xu, King, *4DGS-WAM: Bridging Past and Future with an Object-Centric World Action Model based on 4D Gaussian
  Splatting*, arXiv 2608.25956 (Aug 2026). All predict forward from the current state; none condition on a goal state.

**Goal-conditioned generation / planning with diffusion**
* Zhou et al., *Act2Goal: From World Model To General Goal-conditioned Policy*, arXiv 2512.23541 (Dec 2025) —
  goal-image-conditioned 2D video world model guiding a policy.
* Janner et al., *Diffuser* (ICML 2022) — whole-trajectory diffusion; start/goal conditioning by **inpainting** (what we do).
* Chi et al., *Diffusion Policy* (RSS 2023) — action-chunk diffusion; our downstream policy (ManiSkill implementation).
* Nag et al., *In-2-4D: Inbetweening from Two Single-View Images to 4D Generation*, arXiv 2504.08366 — single object,
  one output, per-instance optimisation (baseline for cost/diversity only).
* Lipman et al., *Flow Matching* (ICLR 2023); Esser et al., SD3 (2024, logit-normal τ); Peebles & Xie, *DiT* (ICCV 2023,
  adaLN-Zero); Zhou et al., *On the Continuity of Rotation Representations* (CVPR 2019, 6D rotations).

**Robots in Gaussian scenes / demonstration synthesis (closest applications; position against these)**
* Qureshi et al., *SplatSim* (arXiv 2409.10161, 2024) — robot Gaussians segmented per link (CAD boxes + KNN for the
  gripper), moved with PyBullet FK: `μ' = Rμ + t`, `Σ' = RΣRᵀ`; objects aligned by ICP. Does **not** handle SH
  rotation — we do (local-frame view directions).
* *RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator*, arXiv 2411.11839.
* Yang et al., *RoboSplat: Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot
  Manipulation*, arXiv 2504.13175 — edits Gaussians (object pose via equivariant transforms of ONE demo, appearance,
  lighting, embodiment, viewpoint); no learned motion model.
* Wang et al., *Demonstration Synthesis from a Single Scan via Gaussian Splatting for Visuomotor Policy Learning*,
  arXiv 2609.21112 (Sep 2026) — 3DGS replica + kinematic grasp/trajectory **planning** on the reconstruction, learned
  contact model; DP trained only on synthetic demos: 95.1% sim, 84.2% real (UR10e). Strong related work: same
  "scan → demos → DP" loop, but planner-based, not a generative start-to-goal model.
* Wang & Liang, *Robot-Aligned 3D Gaussian Splatting for Precise Reaching with Wrist-Camera VLA Policies*,
  arXiv 2610.35626 (Oct 2026) — 3DGS wrist-view rendering + smooth reaching trajectories for π0.5.
* Mandlekar et al., *MimicGen* (CoRL 2023) and Xue et al., *DemoGen* (RSS 2025) — demo augmentation by transforming
  source demos; classic data-generation baselines.

**Positioning sentence:** prior Gaussian demo synthesis either transforms existing demos or calls a motion planner
with privileged geometry; GaussBridge learns a stochastic, goal-conditioned motion prior over an explicit
part-structured Gaussian state, producing many physically checked trajectories between two observed snapshots.

Weekly arXiv check queries: "goal-conditioned Gaussian world model", "4D Gaussian inbetweening", "start goal 4D
generation", "Gaussian splatting demonstration synthesis", "trajectory diffusion Gaussian splatting robot".

---

## 15. Open decisions (owner decides, then record here)
* [ ] Confirm course deadline and report format (CVPR template, as for the proposal?) — Shivam, by Nov 9.
* [ ] Group staging directory requested from CHTC — Aditya, by Nov 9.
* [ ] W&B team project name — Shivam.
* [ ] Robot end in conditioning for the downstream data: object-only goal (default) vs full goal.
* [ ] Real-world stretch go/no-go at G4 (needs a tabletop, phone, AprilTag; no robot needed: objects-only mode
      requires training a variant with the robot token masked).
