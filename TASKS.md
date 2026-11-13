# GaussBridge task board

How to use: pick the lowest-numbered task in your workstream whose dependencies are done, set `Status: in progress`
and your name, follow the steps, and mark it done only when the **Acceptance** check passes (paste the numbers).
Read `CLAUDE.md` first; section references (§) point there. Workstreams: **A** sim & data (Aditya), **B** Gaussians &
rendering (Amogh), **C** motion model & evaluation (Shivam). `S` = shared.

Status legend: todo · in progress · blocked · done

---

## Week 0–1: setup (everyone, Nov 9–11)

### S0.1 Repo + conventions — owner: Amogh (was Shivam) — Status: done on Windows + CI (2026-11-11), see progress.md
- Create the GitHub repo with the layout in §11 (empty modules + `pyproject.toml`, `src/` package, pytest, ruff).
- Add `CLAUDE.md`, `TASKS.md`; README with the 5-line quick start.
- Acceptance: `pip install -e . && pytest` runs (0 tests OK) on all three laptops.

### S0.2 Local environment — owner: each person — Status: todo
- Follow §3.2 (Python 3.11). On machines without NVIDIA Vulkan use lavapipe (§3.3).
- Acceptance: `python -m mani_skill.examples.motionplanning.panda.run -e StackCube-v1 -n 3 --only-count-success`
  produces `demos/StackCube-v1/motionplanning/*.h5` (use `-b cpu`; with lavapipe set `VK_ICD_FILENAMES`).

### S0.3 Docker image + CHTC — owner: Aditya — Status: todo
- Build the image in §3.4 (gsplat from source with `TORCH_CUDA_ARCH_LIST`), push `gaussbridge:0.1`.
- Request group staging dir; write `chtc/run.sh` and the submit templates (§3.5): CPU job, GPU short/resumable job,
  interactive job.
- Acceptance: (a) a CPU job runs the planner smoke test with lavapipe; (b) a GPU job runs
  `python -c "import gsplat, torch; ..."` rendering 1k random Gaussians with `gsplat.rasterization` and
  `python -m mani_skill.examples.demo_random_action -e StackCube-v1 --render-mode rgb_array` (NVIDIA Vulkan works).

---

## Workstream A: simulation & data

### A1 Randomised planner generator — deps S0.2 — owner: Aditya — Status: done (2026-11-06), see progress.md
- `sim/planner_gen.py`: `Knobs`, `sample_knobs(rng)`, `solve_stack_cube`, `solve_pick_cube` exactly as §6.1
  (copy the official solutions' structure; grasp yaw fallback via `dry_run`; RRT fallback to screw; wrap in
  try/finally `planner.close()`).
- `scripts/gen_demos.py --env-id --seed-start --num-seeds --refs-per-seed --out`: RecordEpisode (obs none,
  pd_joint_pos, physx_cpu, `max_episode_steps=1000`), seeds `default_rng([seed, ref, attempt])`, up to 3 attempts per
  ref, `flush_trajectory(save=False)` on failure, sidecar `<out>.knobs.json`.
- Acceptance: 3 seeds × 4 refs for both tasks; success/attempt ≥ 0.8; lengths 120–300 steps; refs of a seed visibly
  differ (plot TCP xyz over time for the 4 refs).

### A2 Extraction to Episodes — deps A1, C1 — Status: todo
- `sim/extract.py`: h5 + json + knobs → `Episode` list (§6.2), attached rule (§5) via `PandaFK`.
- `scripts/extract_episodes.py` → `episodes/<env>/<split>/*.npz` packed per split as tar.
- Acceptance: unit test on a generated shard: `attached` is True on a single contiguous interval per moved object,
  finger ≈ 0.018 m inside it, cubeB never attached (StackCube).

### A3 Physics replay tool — deps A2 — Status: todo
- `sim/replay.py`: reset with seed (or `set_state_dict`), execute `actions_pd_joint_pos` + 10 hold steps, return
  success and per-object end error.
- Acceptance (regression test, already verified once): planner episode → `episode_to_motion` (N=128) →
  `motion_to_episode` → replay: 100% success on 20 episodes, joint error < 0.01 rad, object end error < 5 mm.

### A4 Full datasets on CHTC — deps A1–A3, S0.3 — Status: todo
- Shards of 50 seeds per CPU job; seed ranges and sizes in §6.1 (train/val/test/policy-real) for StackCube, then PickCube.
- Extract, compute duration μ/σ and joint/position statistics, tar per split to staging.
- Acceptance: counts within 10% of plan; a stats JSON committed (`data_stats/<env>.json`); 20 random episodes replay
  successfully.

### A5 Reproduce the Diffusion Policy baseline — deps S0.3 — Status: todo
- Clone ManiSkill, `pip install -e examples/baselines/diffusion_policy`.
- Replay our policy-real demos to rgb: `replay_trajectory --use-env-states -o rgb -c pd_joint_pos --save-traj`.
- Train `train_rgbd.py` (rgb, pd_joint_pos, `--max_episode_steps 400`, N ∈ {10, 25, 50}, 3 seeds) on StackCube.
- Acceptance: success-rate curve for R-N (this is the reference line of the main result). Record GPU memory used.

### A6 DP exporter for generated data — deps B5, C6 — Status: todo
- `export/maniskill_h5.py` per §9 (agent qpos/qvel, extra keys per task, rgb per policy camera, sensor_param, JSON
  from a real file); `scripts/merge_h5.py` to concatenate real + generated.
- Acceptance: DP `load_demo_dataset` + `reorder_keys` load a merged file; shapes equal a real replayed demo;
  a 2k-iteration DP run on the merged file starts and logs loss.

### A7 Downstream experiments — deps A5, A6, C9 — Status: todo
- Arms of §10.3 for StackCube (then PickCube): R-N, R-N+G-M, R-N+G-M filtered, R-N+P-M, R-N+G-M sim-rendered,
  R-N+SL-M; N ∈ {10, 25, 50}, M ∈ {100, 300}; 3 seeds; 100 eval episodes.
- Acceptance: table with mean ± std success; plot success vs N per arm.

---

## Workstream B: Gaussians & rendering

### B1 Capture environments — deps S0.2 — owner: Amogh — Status: done (2026-11-11), 99.94% reprojection agreement, see progress.md
- `sim/envs.py`: `StackCubeCapture-v1`, `PickCubeCapture-v1` (§7.2 ring: 12 az × {20,45,70}°, r = 1.1 m, target
  [-0.15,0,0.15], 256², vfov 60°) keeping policy cameras; tables `OBJECTS`, `POLICY_CAMERAS`, `DEFAULT_ROBOT`.
- `sim/capture.py`: `part_table(env)` from `segmentation_id_map`, `part_poses`, `capture_state` (rgb, depth m, seg →
  part index, viewmats 4×4, Ks, part_T).
- Acceptance: test projecting each part's depth-backprojected points with `viewmat`/`K` lands on its seg mask
  (> 99% pixels); hand-camera viewmat from FK equals ManiSkill extrinsic (< 1e-5).

### B2 Asset-mode captures — deps B1 — owner: Amogh — Status: done (2026-11-11), weakest link in 71 views, see progress.md
- `random_configs` (§7.2: 8 configs, robot noise ±0.6, link-height rejection, objects resting/floating) + 2 held-out
  configs; save as torch files to staging.
- Acceptance: a contact sheet image of 4 configs × 4 views; every visual robot link appears in ≥ 20 views.

### B3 PartSplats + composition — deps C1 — Status: todo
- `gs/splats.py`: storage (§7.1), `world(part_T)`, local-frame SH colours (§7.3), save/load/subset/merge,
  `points_for_shape_encoder`; `gs/raster.py` (gsplat on CUDA; dense PyTorch rasteriser for CPU tests).
- Acceptance: CPU tests: (1) identity poses render identical to gsplat-style direct rendering; (2) rotating a part
  by R and the camera by the same R about the part origin gives the same image (checks SH-in-local-frame);
  (3) dense rasteriser vs gsplat on GPU: max abs diff < 1e-3 on a random scene.

### B4 Articulated reconstruction — deps B2, B3 — Status: todo
- `gs/recon.py` per §7.2 (init by back-projection per part, L1+SSIM+depth+label CE, DefaultStrategy with
  non-trainable `part_id`, optional pose refinement); `scripts/train_recon.py`.
- Acceptance: StackCube asset model: held-out config PSNR ≥ 28 dB at base_camera (128²) and ≥ 26 dB at capture res;
  labels accuracy ≥ 95%; save `assets/StackCube-v1/scene.splats.pt` and `cubeA/cubeB.points.npy`. Repeat PickCube.

### B5 Renderer for trajectories — deps B4, A2 — Status: todo
- `gs/render.py` + `scripts/render_episodes.py`: Episode → per-step part poses (FK for links, objects from
  episode, background fixed) → policy cameras (base fixed, hand from FK) → RGB uint8 + optional depth/labels/flow/tracks.
- Acceptance: render 20 planner test episodes and compare with ManiSkill renders (`replay --use-env-states -o rgb`):
  PSNR/SSIM/LPIPS table (§10.2, asset mode); side-by-side video committed.

### B6 Labels: flow and tracks — deps B5 — Status: todo
- Scene flow / optical flow / 3D + 2D tracks + visibility (§7.3).
- Acceptance: tracks of cube Gaussians stay on the cube mask in > 98% visible frames; optical flow warps frame t to
  t+1 with lower photometric error than zero flow.

### B7 Scene-mode reconstruction — deps B4 — Status: todo
- Start + end captures only (2 configs) for 20 test episodes; variant with merged robot asset.
- Acceptance: §10.2 table rows for scene mode; failure cases documented (hidden faces, floaters).

### B8 (Stretch) Real objects-only capture — deps B7, decision at G4 — Status: todo
- Phone walk-around before/after with AprilTag; SAM 2/3 masks; VGGT or COLMAP; ICP + photometric pose refinement.
- Acceptance: qualitative video of generated motions in a real scene; endpoint error vs measured tag-frame poses.

---

## Workstream C: motion model & evaluation

### C1 Geometry + trajectory core — deps S0.1 — owner: Aditya — Status: done (2026-11-06), see progress.md
- `geometry.py` (§4 conventions: quats wxyz, 6D, pose7↔4×4, se3_inverse, transform_gaussians, SAPIEN→OpenCV camera,
  look-at, fov→K, project) and `trajectory.py` (Episode, resampling, motion tensor §6.3, known masks, decode).
- Acceptance: unit tests: quat↔matrix↔6D round trips (1e-6), slerp endpoints, resampling preserves endpoints,
  `episode_to_motion`→`motion_to_episode` preserves L and endpoints, cube symmetry group has 24 elements.

### C2 Kinematics wrapper — deps C1 — owner: Amogh — Status: done (2026-11-11), ≤ 9.1e-7 m, see progress.md
- `sim/kinematics.py`: `PandaFK(robot_uid)` (panda → panda_v2.urdf, panda_wristcam → panda_v3.urdf), link poses,
  TCP, hand camera viewmat; batch-flattening.
- Acceptance: FK vs simulator link poses < 1e-5 m for 50 random configs (both robots).

### C3 Dataset + model — deps C1, A2 (small shard ok) — Status: todo
- `model/dataset.py` (§6.3–6.4, endpoint-noise augmentation, analytic cube shapes until B4), `model/encoders.py`,
  `model/dit.py` (§8.1), `model/flow.py` (§8.2 loss, §8.3 sampler without guidance).
- Acceptance: overfit 10 episodes (loss → ~0); sampled endpoints equal conditioning exactly (max abs diff 0);
  forward pass on batch 256 fits a 40 GB GPU in bf16.

### C4 Training pipeline — deps C3, A4, S0.3 — Status: todo
- `scripts/train_motion.py` (Hydra; AdamW, cosine, EMA, bf16, resumable checkpoints every 2k steps, W&B); CHTC job.
- Acceptance: StackCube v1 model trained 150k steps; val loss curve; 16 sampled trajectories visualised (matplotlib
  3D TCP/object tracks + one Gaussian-rendered video once B5 exists).

### C5 Metrics + evaluation harness — deps C1 — Status: todo
- `eval/metrics.py` (§10.1), `scripts/eval_motion.py` (test split, both goal modes, S = 32), JSON + markdown table output.
- Acceptance: metrics on planner references vs themselves (sanity: coverage 1.0) and on the straight-line baseline.

### C6 Guidance, projection, filters, diversity selection — deps C2, C3, B4 (spheres from Gaussians; use cube/box
  spheres from geometry until then) — Status: todo
- `guidance/collision.py` (k-means spheres per part; energy; max-penetration metric), `guidance/attach.py`
  (energy + hard projection with rejection), filters and farthest-point selection in `scripts/sample.py`.
- Acceptance: on 50 val pairs, guidance lowers penetration rate vs λ=0 without raising endpoint error; projection
  release error < 1 cm on ≥ 80% of samples; physics replay success reported.

### C7 Baselines — deps C3, C5 — Status: todo
- Straight line; forward-only (end dropout model, no inpainting of end); step-by-step AR model (§8.4, separate config);
  planner oracle (test refs).
- Acceptance: all four in the §10.1 table.

### C8 Main metrics table + ablations — deps C4–C7 — Status: todo
- §10.1 for StackCube and PickCube; ablations §10.4.
- Acceptance: tables frozen in `results/` with config hashes.

### C9 Generate data for downstream — deps C6, B5 — Status: todo
- New start seeds (policy range), object-only goal from task definition (§10.3), S candidates → filter → k per start;
  also straight-line and planner (oracle) sets for the control arms; physics-replay flags.
- Acceptance: M = 300 StackCube episodes per arm in Episode format + replay success stats.

---

## Shared: writing (Week 5–6)
### S1 Figures — pipeline diagram, sample grid (same start/end, 8 diverse trajectories), render comparison, success curves.
### S2 Report (CVPR template, as the proposal) and presentation; every number traceable to `results/*.json`.
