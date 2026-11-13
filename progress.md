# Progress

Shared log of what is built, by whom, and what is left. Task IDs follow `TASKS.md` (the detailed board that
`CLAUDE.md` refers to); `TASKS_helper.md` IDs are in brackets where they differ.

**Before starting work:** `git pull`, read this file, and add/claim your row (`in progress` + your name) so nobody
duplicates it. **When you finish:** set the status, list what exists, and paste the acceptance numbers.

Status: todo · in progress · done · blocked

## Task status

| Task | Owner | Status | What exists | Notes / numbers |
|---|---|---|---|---|
| S0.1 Repo scaffold [S1] | Amogh (Shivam's task) | done on Windows (2026-11-11); Linux = CI | `pyproject.toml`, `src/gaussbridge/__init__.py`, `tests/conftest.py`, `.gitattributes`, `.github/workflows/tests.yml`, README quick start | `pip install -e ".[dev]"` keeps the installed torch; see S0.1 below |
| S0.2 Laptops [S1] | each | in progress | Amogh: Windows done for B work (2026-11-10), see "Windows setup" | Macs can't run the planner natively (see Findings); use the CPU image below. Windows can't run the planner either (no mplib) |
| S0.3 Docker + CHTC [S2] | Aditya | in progress | `docker/Dockerfile.cpu` (CPU, lavapipe) | CUDA image (CLAUDE.md §3.4) and CHTC submit files still todo |
| A1 Randomised planner | Aditya | done (2026-11-06) | `src/gaussbridge/sim/planner_gen.py`, `scripts/gen_demos.py`, `scripts/plot_refs.py`, `tests/test_planner_gen.py` | 80% / 100% success per attempt; see A1 below |
| A2 Extraction to Episodes | Aditya | done (2026-11-12) | `src/gaussbridge/sim/extract.py`, `scripts/extract_episodes.py`, `tests/test_extract.py` | 24/24 dev demos: one grasp interval, finger 18.3 mm; see A2 below |
| A3 Physics replay [A2] | Aditya | todo | — | |
| A4–A7 | Aditya | todo | — | |
| B1 Capture environments | Amogh | done (2026-11-11) | `src/gaussbridge/sim/envs.py`, `src/gaussbridge/sim/capture.py`, `geometry.backproject`, `tests/test_capture.py`, `scripts/check_lighting.py`, `results/lighting/` | reprojection agreement 99.94–99.96%; lighting confirmed (21 dB cap); see B1 below |
| B2 Asset-mode captures | Amogh | done (2026-11-11) | `sim/capture.py` (`random_configs`, `apply_config`, `capture_configs`, `view_valid`, save/load), `scripts/capture_assets.py`, `tests/test_configs.py`, `results/b2/` | every visual robot link in ≥ 67 training views (need 20); see B2 below |
| B3 PartSplats + composition | Amogh | done (2026-11-12) | `src/gaussbridge/gs/{splats,raster,render}.py`, `geometry.viewmat_to_campos`, `tests/test_splats.py`, `scripts/check_splats_capture.py`, `results/b3/` | dense rasteriser vs gsplat 2.3e-6; part+camera rotation invariance 2e-5; lit colour model built; see B3 below |
| B4–B8 | Amogh | todo | — | |
| C1 Geometry + trajectory core | Aditya (Shivam's task) | done (2026-11-06) | `src/gaussbridge/geometry.py`, `src/gaussbridge/trajectory.py`, `tests/test_geometry.py`, `tests/test_trajectory.py`, `tests/test_sim_integration.py` | 24/24 planner demos replay after the 128-step round trip; see C1 below |
| C2 Kinematics (PandaFK) [none] | Amogh (Shivam's task) | done (2026-11-11) | `src/gaussbridge/sim/kinematics.py`, `tests/test_kinematics.py` | link poses vs sim ≤ 9.1e-7 m on 50 random configs, both robots; see C2 below |
| C3 Dataset + model | Shivam | in progress (2026-11-13): code done, overfit passed, memory check pending | `src/gaussbridge/model/{dataset,encoders,dit,flow,evaluate}.py`, `scripts/overfit_motion.py`, `scripts/check_memory.py`, `chtc/{overfit,check_memory}.{sub,sh}`, `tests/test_motion_*.py` | overfit on 10 real episodes: 6 mm / 0.9° / 0.026 rad to the nearest episode, endpoints exact; see C3 below |
| C5 Metrics + harness | Shivam | in progress (2026-11-13): metrics, straight line, harness done; sanity passed | `src/gaussbridge/eval/{metrics,baselines}.py`, `scripts/eval_motion.py`, `tests/test_eval_metrics.py`, `results/c5/` | refs vs themselves: coverage 1, precision 1, minADE 0; penetration waits for C6 spheres, replay for A3; see C5 below |
| C4 Training pipeline | Shivam | in progress (2026-11-13): pipeline done, CHTC trial running; real run needs A4 | `src/gaussbridge/model/train.py`, `scripts/train_motion.py`, `configs/train/stackcube_v1.yaml`, `chtc/{train.sub,train.sh,pack.sh}`, `tests/test_motion_train.py` | exact resume verified (CPU); see C4 below |
| C6–C9 | Shivam | todo | — | |

## A1 — randomised planner demos

**Run (any machine with Docker; on Apple Silicon this runs under emulation):**
```bash
docker build --platform linux/amd64 -f docker/Dockerfile.cpu -t gaussbridge-cpu:0.2 .
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work gaussbridge-cpu:0.2 pytest -q tests/test_planner_gen.py
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work gaussbridge-cpu:0.2 \
  python scripts/gen_demos.py --env-id StackCube-v1 --seed-start 0 --num-seeds 3 --refs-per-seed 4 --out data/raw/StackCube-v1/dev
```
Output per shard: `<out>.h5` + `<out>.json` (ManiSkill format, successful episodes only, no observations) and
`<out>.knobs.json` = `{episode_id: {seed, ref_idx, attempt, knobs, num_steps}}`.

Diversity plot: `python scripts/plot_refs.py data/raw/StackCube-v1/dev StackCube-v1 0 refs.png` (inside the image).

**Tests** (`tests/test_planner_gen.py`, 3 tests, ~7 s): knob sampling is reproducible and in range; for both tasks,
2 refs of one seed are saved, share the exact same start state, and have different actions.

**Acceptance (3 seeds × 4 refs, seeds 0–2):**

| Task | Success per attempt | Saved | Length (steps @20 Hz) | Refs differ |
|---|---|---|---|---|
| StackCube-v1 | 12/15 = **0.80** | 12/12 | 134–214, mean 165 | yes (plot) |
| PickCube-v1 | 12/12 = **1.00** | 12/12 | 113–213, mean 157 | yes (plot) |

* StackCube is exactly at the 0.8 bar on a small sample; recheck on a full shard in A4. All 3 failures were seed 2
  with grasp rotation 270°: the grasp is reachable but a later move fails; the retry fixes it.
* PickCube's shortest demo (113) is below the 120-step lower bound; that bound fits StackCube, and PickCube is a
  shorter task (no place, no release).
* Re-running the same shard gives identical files, including RRT demos.

**Decisions made (not settled by CLAUDE.md):**
* A failed attempt = any planner move returns -1, the planner raises, or the task is not solved at the last step.
  Failed attempts are discarded; after 3 failed attempts a reference is skipped (seed keeps fewer refs).
* Grasp yaw fallback: sampled k first, then k+1, k+2, k+3 (mod 4), first one the planner can reach (`dry_run`).
* Via waypoint = midpoint of lift pose and final target + the sampled offset; gripper orientation kept.
* When `rrt_transport` is set, RRTConnect is used for every transport move (via, pre-place, final place), each
  falling back to screw motion if RRT fails to plan. The planner has no table/cube obstacles in either mode.
* PickCube ignores `place_h` (recorded as 0 in the knobs file).

## C1 — geometry + trajectory core

**What's there** (all torch, any leading batch dims, differentiable; `Episode` stores numpy):
* `geometry.py`: `quat_to_matrix`, `matrix_to_quat` (returns w ≥ 0), `quat_mul`, `matrix_to_rot6d`,
  `rot6d_to_matrix`, `slerp`, `cube_symmetries` (24), `pose7_to_matrix`, `matrix_to_pose7`, `se3_inverse`,
  `transform_gaussians`, `sapien_pose_to_viewmat` (hand camera from `camera_link`), `look_at_viewmat`, `fov_to_K`,
  `project`.
* `trajectory.py`: `Episode` (fields of CLAUDE.md §6.2, `.actions()`, `.save()/.load()` npz), `resample`,
  `episode_to_motion` → `{x [128,4,10], ent_mask [4], log_dur}`, `known_mask(ent_mask, goal="full"|"objects"|"none")`,
  `decode_motion` (differentiable, for guidance), `motion_to_episode`.

**Tests** (`pytest -q tests/` in the CPU image: 10 tests, ~15 s; the first two files need only torch):
* `test_geometry.py`: rotation/pose round trips incl. 180° turns and noisy 6D; cube group is 24 elements and closed;
  moving Gaussians rotates their covariance as R Σ Rᵀ.
* `test_trajectory.py`: resampling keeps endpoints exactly, follows a rotation whose quaternion flips sign midway,
  and keeps the gripper command a clean ±1 step at the right time.
* `test_sim_integration.py` (uses ManiSkill + A1): camera helpers reproduce ManiSkill's base/hand extrinsics and
  intrinsics (1e-5) and project cube centres onto the cubes' segmentation pixels; a fresh planner demo (both tasks)
  → Episode → npz → 128-step motion → Episode keeps length and endpoints and still solves the task in physics.

**Round trip on all 24 A1 dev demos** (Episode → motion → Episode → replay `pd_joint_pos` + 10 hold steps):
| Task | Replay success | Max joint path error | Max object end error |
|---|---|---|---|
| StackCube-v1 | 12/12 | 0.0015 rad | 3.9 mm |
| PickCube-v1 | 12/12 | 0.0014 rad | 5.6 mm |

PickCube's end error is mostly not from resampling: replaying measured qpos as targets (the §6.2 action definition)
already gives up to 3.3 mm vs 0.7–2.0 mm with the planner's own actions. **A3: the 5 mm bound is tight for
PickCube** (1/12 above it, success tolerance is 25 mm); consider 1 cm.

**Decisions made:**
* Gripper / attached are resampled by nearest sample (keeps endpoints exact); recorded in CLAUDE.md §6.3.
* The robot token stores finger joint 1 only; decoding sets both fingers to it.
* `matrix_to_quat` returns w ≥ 0 but ManiSkill quaternions can have w < 0: compare rotations, never raw quaternions.
* Duration stats `LOG_DUR_MU = log 8`, `LOG_DUR_SIGMA = 0.35` are placeholders in `trajectory.py` until A4.
* Pixel j covers u ∈ [j, j+1) (principal point W/2, H/2): index images with `floor(uv)`. Added to CLAUDE.md §4.

**For the next tasks:** A3 can lift the replay loop from `tests/test_sim_integration.py`.

## C3 — dataset + motion model (Shivam)

**Code** (`src/gaussbridge/model/`, all pure torch, run anywhere):
* `dataset.py` — `MotionDataset(episodes)`: per sample the clean grid `x0 [128,4,10]`, `known [128,4]`, the given copy
  `x_known` (endpoint noise 5 mm / 2° / 0.01 rad on the given copy only), `log_dur` normalised with `duration_stats`
  of the training episodes, cube shape points `[3,512,6]` (`box_shape_points`, same format as
  `PartSplats.shape_points`). Goal mode per sample: none 0.1 / objects 0.3 / full 0.6.
* `encoders.py` — `PointSetEncoder` (PointNet). No height-map encoder (constant input in v1, CLAUDE.md §6.4).
* `dit.py` — `MotionDiT`: 513 tokens (128 steps × 4 slots + duration), 8 blocks, d 384, adaLN-Zero, padded slots masked;
  30.5 M parameters; predicts 0 at initialisation.
* `flow.py` — `flow_loss` (logit-normal τ, given cells never noised, loss over free real cells, duration weight 0.1) and
  `sample` (40 Euler steps, given cells re-imposed every step). Padded slots and the robot spacer feature are held at 0.
* `evaluate.py` — `motion_errors`: joint rad, finger mm, object mm / deg, duration s.
* `trajectory.decode_motion` / `motion_to_episode` take `dur_stats` (the model's training stats) for generated motions.

**Tests:** 30 in `tests/test_motion_{dataset,dit,flow,evaluate}.py` (CPU, ~20 s): sample layout, goal-mode masks and
frequencies, noise sizes, padding isolation, every input reaches the output, batch independence, all parameters get
gradients, oracle tests (an oracle velocity gives zero loss and the sampler following it lands exactly on the data),
exact endpoints in all goal modes, a tiny model learning two motions. Every test was checked against a planted bug.
Two real bugs found this way: a noisy robot spacer feature (train/sample mismatch) and `known_mask` ignoring the device.

**Overfit check** (`scripts/overfit_motion.py`, CHTC job 8539260, A100 MIG 3g.40gb, 10k steps, batch 64, lr 3e-4
constant, full-goal mode, no endpoint noise; 10 StackCube dev episodes = seeds 0–2; 115 min):

| step | loss | joint (rad) | cube pos (mm) own / nearest | cube rot (deg) | duration (s) own / nearest | endpoints |
|---|---|---|---|---|---|---|
| 0 (untrained) | ~1.4 | ~2.0 | ~485 | ~125 | | |
| 1,000 | 0.172 | 0.050 | 13.9 / 11.8 | 1.89 | 0.53 | exact |
| 5,000 | 0.014 | 0.032 | 8.8 / 5.7 | 1.11 | 0.43 | exact |
| 10,000 | 0.011 | 0.036 / **0.026** | 10.5 / **6.0** | **0.97** / 0.93 | 0.57 / **0.04** | **exact (max diff 0)** |

"Nearest" = against the closest of the 10 training episodes. 5 of 10 samples reproduce their own episode (≈ 6 mm,
duration within 0.03–0.09 s); the other 5 reproduce a sibling with the **same start, same grasp yaw and end states
within 0.023 rad / 7 mm** (smaller than the endpoint noise), i.e. the same request with two valid answers. So the model
learned both modes; this is the diversity we want, not a failure. Loss plateaus at ≈ 0.011 (not → 0) with this simple
recipe (constant lr, no EMA); C4 candidates: lr decay + EMA, and rescaling joint features (arm joints vary with std
0.06–0.13 in model units vs noise std 1). Results: `runs/overfit_8539260/` locally, `~/gaussbridge/overfit/` on CHTC.

**CHTC lessons** (CLAUDE.md §3.5): Blackwell GPUs fail with torch 2.5.1 → request capability 8.0–9.0; job scripts
always return logs; one folder per job type under `~/gaussbridge/`.

## C3–C5 open items (branch `shivam/c3-c4-c5-model-training-eval`, one PR for all three)

Updated as results come in. The PR merges when every **merge blocker** is ticked; the rest stays open in its task.

| # | Task | Item | Merge blocker? | Status / evidence |
|---|---|---|---|---|
| 1 | all | CI green on Linux (planner + simulator tests, which cannot run on a Mac; shared files changed: `trajectory.py`, `kinematics.py`, `CLAUDE.md`) | yes | **done**: PR #10 CI run 43496909955, 86 passed, 5 skipped (GPU), 3m38s |
| 2 | C3 | Batch 256 in bf16 fits a 40 GB GPU (TASKS.md C3 acceptance) | yes | **done**: CHTC 8547380 on A100-SXM4-40GB: bf16 batch 256 peak **26.8 GB** (< 40 GB), 0.31 s/step; fp32 does not fit (OOM). 150k steps ≈ 13 h on this card (plan said 6–8 h) |
| 3 | C4 | GPU trial: full model, bf16, `pytorch_kinematics` install on the node, at least one real CHTC pause → restart → resume, validation metrics, `final_ema.pt` | yes | **done**: CHTC 8549894 (8547386 crashed first: macOS tar's hidden `._*.npz` files; fixed in `chtc/pack.sh` + `trajectory.episode_files`). Full model, bf16, A100; **two real pause → restart → resume cycles** (2143 → 2755 → 3000, exit 85, 85, 0); validation incl. C5 metrics on the node; `final_ema.pt` written. Details: C4 section |
| 4 | C3 | Overfit PickCube (1 object, padded slots on real data) | no (extra check) | running: CHTC 8547379.0 (step 6k: 7.5 mm, endpoints exact; slow GPU) |
| 5 | C3 | Overfit StackCube with all goal modes; object-only evaluation on real data | no (extra check) | **done**: CHTC 8547379.1, 10k steps, trained on all goal modes: full goal 8.0 mm / 0.67° / 0.024 rad (nearest 3.9 mm); **object-only goal 9.7 mm / 0.68° (nearest 3.9 mm)**, joints 0.050 rad (robot end is free, so its joints may legitimately differ); endpoints exact in both modes |
| 6 | C3 | Overfit StackCube, full goal (TASKS.md C3 acceptance) | yes | **done**: CHTC 8539260, 6 mm / 0.9° / 0.026 rad to nearest episode, endpoints exact |
| 7 | C3 | Sampled endpoints equal the conditioning exactly (TASKS.md C3 acceptance) | yes | **done**: tests + every evaluation of 8539260 (max diff 0) |
| 8 | C5 | Sanity on dev data: references vs themselves coverage 1, precision 1, minADE 0; straight-line baseline scored | yes | **done**: `results/c5/*_dev.md` |
| 9 | C5 | Penetration metric (needs C6 sphere proxies) | no → C6 | todo |
| 10 | C5 | Physics-replay success (needs A3) | no → after A3 | todo |
| 11 | C5 | Decide the coverage threshold (3 cm is lenient: references differ by 1.7–2.9 cm) | no → C8 | open question |
| 12 | C4 | Real training run on the A4 split (150k steps) + val curves + 16 samples visualised (TASKS.md C4 acceptance) | no → after A4 | todo (needs A4) |
| 13 | C4 | Try joint-feature rescaling / check joint error plateau (≈0.03 rad in the overfit) | no → C4 follow-up | open question |

## C4 — training pipeline (Shivam)

```bash
python scripts/train_motion.py configs/train/stackcube_v1.yaml --run runs/<name> [--set optim.steps=1000 ...] [--max-minutes 120]
bash chtc/pack.sh <train episodes dir> <val episodes dir> <bundle dir>    # then condor_submit chtc/train.sub
```
* Recipe (CLAUDE.md §8.2) in `model/train.py::Trainer`: all goal modes + endpoint noise, AdamW 2e-4 / wd 0.01, 2k warm-up +
  cosine, grad clip 1.0, bf16 on GPU, EMA 0.999 (validation and `final_ema.pt` use it), checkpoints every 2k steps
  (atomic, last 2 kept). Validation: EMA loss over the val set; every 10k steps 16 samples per val start (object-only
  goal) scored with the C5 metrics (also coverage/precision @1 cm).
* **Exact resume:** all randomness of step n comes from a generator seeded by (seed, n); test: 5 steps + stop + new
  Trainer + resume == 12 uninterrupted steps, bit-identical weights/EMA/optimiser on CPU.
* **CHTC:** `chtc/train.sub` uses self-checkpointing: `--max-minutes` → checkpoint + exit 85 → HTCondor saves the run
  directory and restarts the job (maybe elsewhere, `+is_resumable`), which resumes. Public PyTorch image +
  `pip install pytorch_kinematics==0.7.6` + the two Panda URDFs shipped in the bundle (`GB_URDF_DIR`, kinematics.py).
* Config: plain YAML + strict overrides (decision, CLAUDE.md §1); floats in YAML need a dot (2.0e-4), overrides accept 1e-4.
* Bugs found by the tests: initial weights were unseeded (runs not reproducible), PyYAML read `1e-3` as text, a log
  window was dropped on pause, a second `--set` replaced the first.

**GPU trial (CHTC 8549894, 2026-11-13; StackCube dev, train = val = 12 episodes, 3000 steps, 12-min segments):**
| step | train loss | lr | val loss (EMA) | val samples (object goal, EMA): minADE / cov@3cm / prec@3cm / APD | jerk (m²/s⁶) |
|---|---|---|---|---|---|
| 1000 | 0.024 | 1.6e-4 | 0.875 | 41.4 cm / 0.00 / 0.00 / 46.2 cm | 4.2e7 |
| 2000 | 0.017 | 5.2e-5 | 0.136 | 8.6 cm / 0.00 / 0.00 / 8.8 cm | 2.8e6 |
| 3000 | 0.014 | 2e-7 | 0.049 | 2.9 cm / 0.58 / 0.29 / 3.5 cm | 3.6e5 |
Endpoints exact throughout. Findings for the real run: (1) samples are jittery: jerk 3.6e5 vs 2–8 for planner motions
(the C6 "jerk > 99th percentile of training" filter would reject everything); recheck after full training, else
smooth / more sampling steps. (2) Speed depends on the card: ~180 steps/min (segment 1) vs ~50 (segment 2).
(3) Checkpoints are 488 MB (model + EMA + optimiser): ~1 GB per run; write runs to /staging (home quota 40 GB).
Results: `/staging/s/smittal39/gaussbridge/trial_stackcube_dev.tar.gz`.

## C5 — metrics + evaluation harness (Shivam)

* `eval/metrics.py`: tracks = TCP (FK) + object centres on the 128-step grid; `ade`, `min_ade`, `coverage`, `precision`
  (@3 cm), `apd`, `mean_squared_jerk` (float64), `endpoint_errors` (object cm, object rotation with the 24 cube
  symmetries, TCP cm). Not yet: penetration (needs C6 sphere proxies), physics replay (needs A3).
* `eval/baselines.py`: `straight_line` (§8.4) in grid form; with an unknown robot end the robot stays at its start.
* `scripts/eval_motion.py --episodes <dir> --out results/c5/<name>`: cases = start seeds, goal modes objects + full
  (conditioning = reference 0), methods `refs` (sanity) and `straight`; a model plugs in as one more method.
* Tests (`tests/test_eval_metrics.py`, 8): hand-made tracks with known answers, jerk of t³ = 36, cube symmetry, TCP
  track = FK, baseline masks/half-way/valid rotations, harness sanity; each checked against a planted bug.

**Dev data (3 seeds × 4 refs per task, `results/c5/*_dev.md`), goal = objects:**
| task | method | end obj (cm) | end TCP (cm) | minADE (cm) | coverage@3cm | precision@3cm | APD (cm) | jerk (m²/s⁶) |
|---|---|---|---|---|---|---|---|---|
| StackCube | refs | 0.23 | 0.54 | 0.00 | 1.00 | 1.00 | 1.74 | 3.34 |
| StackCube | straight | 0.00 | 14.05 | 5.27 | 0.00 | 0.00 | 0.00 | 0.00 |
| PickCube | refs | 0.30 | 0.31 | 0.00 | 1.00 | 1.00 | 2.92 | 3.78 |
| PickCube | straight | 0.00 | 12.05 | 7.14 | 0.00 | 0.00 | 0.00 | 0.00 |

**Open question for C8 (threshold):** references from one start differ by only 1.7–2.9 cm ADE on average, below
the 3 cm coverage threshold of §10.1, so one sample can "cover" several references; with the full goal even the
straight line reaches coverage 0.17–0.33. Proposal: also report coverage/precision at 1 cm (and decide in C8).
Jerk alone does not separate the straight line (0) from real motion; coverage and precision do.

## A2 — extraction to Episodes

```bash
python scripts/extract_episodes.py --split train --out data/episodes/StackCube-v1 data/raw/StackCube-v1/shard_*.h5
```
Writes `<out>/<split>/<seed:05d>_<ref:02d>.npz` (one `Episode` each, `meta` incl. knobs and `git_commit`) and packs
them as `<out>/<split>.tar` (one file for CHTC staging). In code: `sim.extract.load_shard(prefix)` → `[Episode]`;
`attached_mask(...)` is the §5 rule (gripper commanded closed, finger < 39 mm, cube centre < 2 cm from the TCP via
`PandaFK.tcp_pose`) and can be reused on generated motions.

**Tests:** `test_extract.py` (both tasks, fresh 1 seed × 2 refs shard): the carried cube is attached on exactly one
contiguous interval, finger 17–21 mm inside it, StackCube releases cubeA (and never attaches cubeB), PickCube still
holds at the end, knobs sidecar matched by episode id. `test_sim_integration.py`'s round-trip test now extracts with
`load_shard`. Full suite in the CPU image: 38 passed, 4 skipped (GPU).

**All 24 A1 dev demos:**
| Task | One grasp interval | Finger while held | Cube centre to TCP while held | Held at last step |
|---|---|---|---|---|
| StackCube-v1 | 12/12 (cubeB never) | 18.3–20.6 mm, median 18.3 | ≤ 4.7 mm | no (released) |
| PickCube-v1 | 12/12 | 18.3–20.5 mm, median 18.3 | ≤ 4.4 mm | yes |
The first held step reads ≈ 20.5 mm (fingers still closing), then 18.3 mm. CLAUDE.md §5 said "1–3 mm" to the TCP;
updated to ≤ 5 mm.

## S0.1 — repo scaffold

* `pip install -e ".[dev]"` (extras: `gs` = gsplat + render metrics, `train` = hydra + wandb, `dev` = pytest/ruff).
  Install torch first from the right index (CLAUDE.md §3); the pins match, so pip leaves it alone (checked with
  `--dry-run` on the Windows CUDA env). No more `PYTHONPATH=src` needed (pytest also sets it).
* `tests/conftest.py` skips `test_planner_gen.py` / `test_sim_integration.py` where mplib is missing (Windows, Mac),
  instead of failing at import. Mark new tests `@pytest.mark.sim` (ManiSkill + Vulkan) or `gpu` (CUDA).
* CI (`.github/workflows/tests.yml`): every PR and push to main runs `pytest` in `python:3.11-slim-trixie` with
  lavapipe (same setup as `docker/Dockerfile.cpu`), so planner + sim tests run on Linux even when the author can't.
* `.gitattributes`: text stored with LF, so `chtc/*.sh` never get Windows line endings.

## C2 — Panda forward kinematics

`PandaFK(robot_uid="panda_wristcam" | "panda", device, dtype)`: `link_poses(qpos, base_pose=None, links=None)` →
`{link: T_world_link [..., 4, 4]}`, `tcp_pose(qpos)`, `hand_camera_viewmat(qpos)` (OpenCV world→camera, wristcam
only). `qpos [..., 9]` in ManiSkill order, any batch dims, differentiable; `base_pose` = pose7 or 4×4, default
`(-0.615, 0, 0)`. Reads ManiSkill's URDF without importing ManiSkill (no SAPIEN/Vulkan needed, e.g. in training jobs).

**Acceptance** (`tests/test_kinematics.py`, 50 random configs within joint limits, all 15 links, float64):

| Robot (task) | Max position error | Max rotation-matrix error | Hand-camera viewmat vs ManiSkill extrinsic |
|---|---|---|---|
| panda_wristcam (StackCube) | 8.6e-7 m | 1.3e-6 | 1.0e-6 |
| panda (PickCube) | 9.1e-7 m | 1.3e-6 | — |

Also tested: batch shapes `[2, 3, 9]`, base-pose shift, gradients w.r.t. joint angles. Speed: 26 ms for 4096
configs (all links, CPU). **A2** can use `tcp_pose` for the attached rule; **B1/B5** use `hand_camera_viewmat`.

## B1 — capture environments

* `import gaussbridge.sim.envs` registers `StackCubeCapture-v1` / `PickCubeCapture-v1` (task env + capture rings;
  `capture_rings=` kwarg to change them). Tables: `OBJECTS`, `POLICY_CAMERAS`, `DEFAULT_ROBOT`, `CAPTURE_ENV`.
  Default rings (`RINGS`): wide 36 cams (CLAUDE.md §7.2) + close 16 cams (r 0.5 m, ≈ 2.3 mm/px) + policy cameras
  = 54 cameras for StackCube, 53 for PickCube; 0.34 s / 0.21 s per capture (RTX 4060, `sapien_cpu`).
* `capture.part_table(env)` → `Part(name, kind, seg_id)` from `segmentation_id_map` (StackCube 19 parts, PickCube 18;
  PickCube's panda has finger pads and no camera links). `capture_state(env)` → dict per camera
  `{rgb uint8, depth float32 m, seg int16 part index (-1 none), viewmat, K}` (policy cameras first; ManiSkill itself
  lists `hand_camera` last) + `part_names, part_kinds, part_T, qpos`. `geometry.backproject` = inverse of `project`.
* Depth from the `minimal` shader is camera-z in whole millimetres (int16), so 1 mm quantisation.

**Acceptance** (`tests/test_capture.py`): points back-projected from one camera, carried with their part's pose and
projected into another camera land on their own part's pixels wherever visible (5 mm depth test):
| Check | Agreement (all parts) | Worst part |
|---|---|---|
| Same configuration, 60 random camera pairs | 99.94% | cubeA 98.3% (edges touching the table) |
| After moving the arm (±0.5 rad) and cubeA (lifted, yawed), 60 pairs | 99.96% | panda_link0 97.9% |
Hand camera viewmat from `PandaFK` = capture viewmat (< 1e-5). Tests assert > 99.5% overall and > 95% per part.

**Lighting check** (`python scripts/check_lighting.py --out results/lighting`): rotate a part and the camera together
(same part-frame geometry and view, mask IoU 1.0): cubeA pixels change up to 12/255 mean (21 dB), arm (joint 1) up to
16/255 (≈ 30 dB within ±60°, 21.5 dB at 150°). So the §7.3 part-frame SH colour model is capped near 21 dB on cubes
→ lighting-aware colour planned for B3/B4 (CLAUDE.md §7.3).

## B2 — asset-mode captures

```bash
python scripts/capture_assets.py --task StackCube-v1 --out data/assets/StackCube-v1/captures.pt     --sheet results/b2/StackCube-v1_sheet.png      # 8 train + 2 held-out configs, seed 0
```
Runs anywhere ManiSkill renders (laptop, CHTC CPU job with lavapipe, GPU job); output is **one file per task**
(fits the "few big files" staging rule) + `<out>.report.json`. Load with `capture.load_captures(path)`;
`capture.config_state(cap, r)` returns configuration r in the B1 `capture_state` format. Format: CLAUDE.md §7.2.

* `random_configs(task, n, seed)` (no simulator needed, FK only): CLAUDE.md §7.2 asset-mode distribution around the
  env's own start pose (`envs.HOME_QPOS`: joint 7 is −π/4 for `panda_wristcam`, +π/4 for `panda`).
* **Invalid views:** with ±0.6 rad noise the arm sometimes swings into a close-ring camera (eye inside a link, the
  frame is one flat grey surface). Each (configuration, camera) gets `valid`; false when one part fills ≥ 90% of the
  view with median depth < 10 cm. Reconstruction (B4) must skip them. Seed 0: StackCube 4/540, PickCube 2/530.

**Acceptance** (`results/b2/<task>.report.json`, contact sheets `results/b2/<task>_sheet.png`):
| Task | Cameras | Capture time (10 configs) | File | Weakest visual robot link (train views ≥ 50 px) | Objects |
|---|---|---|---|---|---|
| StackCube-v1 | 54 | 2.7 s | 197 MB | panda_leftfinger 71 (others 156–338) | cubeA 353, cubeB 364 |
| PickCube-v1 | 53 | 3.0 s | 196 MB | panda_leftfinger 67 (others 69–330) | cube 381 |
Tests: `tests/test_configs.py` (limits, link heights, clearances, resting/floating, determinism, `view_valid`);
`tests/test_capture.py::test_capture_configs_apply_reproject_and_roundtrip` (applied poses equal config + FK,
reprojection between two different configurations > 99.5%, save/load round trip). Full suite: 19 passed.

## B3 — PartSplats + composition

```bash
pytest -q tests/test_splats.py                       # 13 tests; the 2 gsplat comparisons need CUDA (marked gpu)
python scripts/check_splats_capture.py --captures data/assets/StackCube-v1/captures.pt --out results/b3
```
* `gs/splats.py` — `PartSplats` (an `nn.Module`): gsplat-style pre-activation params in `splats.params`
  (a `ParameterDict` that `DefaultStrategy` can densify directly; `part_id` is a non-trainable Parameter),
  `world(part_T [P,4,4])` (composition, differentiable in `part_T` for pose refinement), `colors(world, campos)`,
  `part_onehot(groups=None)`, `shape_points(name, n=512)` (shape-token input, §6.4), `subset`, `merge`, `save`/`load`,
  `part_T_from_dict`. `init_from_points(points, rgb, part_id, names, kinds, color_mode=..., part_T=...)` (§7.2 init,
  works on GPU). Two colour models: `"sh"` (§7.3 part-frame SH) and `"lit"` (world-lit albedo + normal + SH residual,
  matching SAPIEN's `minimal` shader; formula in CLAUDE.md §7.3).
* `gs/raster.py` — `rasterize(...)` → colours [C,H,W,D], expected depth, alpha, meta. `backend="gsplat"` (CUDA,
  `RGB+ED`, `packed=False`) or `"dense"`: a differentiable PyTorch copy of gsplat's classic kernels (Jacobian clamp,
  eps2d 0.3, 1/255 alpha threshold, exclusive 1e-4 transmittance stop, pixel centres at +0.5) for CPU tests.
* `gs/render.py` — `render(splats, part_T, viewmats, Ks, W, H, background, features)` (B5 extends it).
* **Fix in `geometry.matrix_to_quat`:** its backward was NaN at the identity rotation (`sqrt(0)`), which would have
  broken pose refinement and any gradient through part poses. Now takes the root only of positive entries (as
  pytorch3d); forward values unchanged, new test `test_matrix_to_quat_has_finite_gradients_at_identity`.

**Acceptance** (RTX 4060 laptop, gsplat 1.5.3 built from source):
| Check (TASKS.md B3) | Result |
|---|---|
| (1) identity poses vs plain gsplat with its own SH evaluation (degree 2, 300 Gaussians, 2 cameras) | max abs 2.7e-7 (rgb), 2.4e-7 (alpha) |
| (2) rotate part and camera together about the part origin, `"sh"` mode (dense, CPU) | identical to 2e-5 (rgb, alpha, depth) |
| (3) dense rasteriser vs gsplat, random 2-part scene, 3 cameras, background + 2 label channels | max abs 1.5e-6 rgb, 2.3e-6 labels/alpha, 1.8e-7 m depth (target 1e-3) |
Also tested: covariances transform as R Σ Rᵀ; `"lit"` matches the SAPIEN diffuse formula for one Gaussian and changes
under a part+camera rotation unless the lights rotate too; lit albedo init round trip; `part_id`, albedo and normals
survive gsplat's `duplicate`/`split`/`remove`; subset/merge/save/load. Full suite: 34 passed.

**Real-capture check** (`results/b3/<task>.splats_check.{json,png}`): back-project StackCube configuration 0
(all valid cameras, ≤ 20k points per part, opacity 0.9, no optimisation), pose with the **held-out** configurations'
part poses, render the policy cameras:
| View | Mask IoU | Part-label accuracy | PSNR (raw init) |
|---|---|---|---|
| base_camera, held-out cfg 8 / 9 | 0.772 / 0.774 (0.9998 without far ground) | 0.944 / 0.928 | 12.7 / 12.5 dB |
| hand_camera, held-out cfg 8 / 9 | 0.968 / 1.000 | 0.896 / 0.910 | 17.5 / 18.8 dB |
PickCube: same numbers within 0.01. Frames, FK-driven composition and the renderer agree with ManiSkill on real data.
Two findings for B4: (a) the far floor (> 3 m, dropped by §7.2's init rule) is 22–25% of every base_camera frame and
must be modelled; (b) blurry init Gaussians of `panda_hand` (sized for the far rings) bleed into the bottom of the
hand camera from ~3 cm away; label errors there are almost all `panda_hand`, which training on hand-camera views
should fix.

## Windows setup (Amogh, 2026-11-10; RTX 4060 Laptop 8 GB, driver 616.56)

Native Windows works for everything except the motion planner (mplib is Linux-only; ManiSkill skips it).
```powershell
conda create -n gaussbridge python=3.11 -y
$py = "$env:USERPROFILE\anaconda3\envs\gaussbridge\python.exe"
& $py -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124   # PyPI torch is CPU-only on Windows
& $py -m pip install mani_skill==3.0.1 "numpy<2" h5py gsplat==1.5.3 ninja jaxtyping rich hydra-core wandb lpips torchmetrics trimesh transforms3d pytest matplotlib
& $py -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124  # lpips pulls a new CPU torch; re-pin
# gsplat from source (needs MSVC 2022 + CUDA 12.4 nvcc; run inside a vcvars64 shell, CUDA_HOME set,
# CUDA's lib dir on LIB, TORCH_CUDA_ARCH_LIST=8.9 for an RTX 40xx):
& $py -m pip install --no-build-isolation --no-deps --force-reinstall "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
```
* **Use `render_backend="sapien_cpu"` on Windows.** The default GPU backend (`sapien_cuda`) crashes with an access
  violation in `get_picture_cuda`. `sapien_cpu` still renders on the NVIDIA GPU through Vulkan (6 ms per `get_obs`
  for both StackCube cameras), just with a CPU readback.
* **The gsplat 1.5.3 PyPI wheel cannot JIT-compile on Windows:** `gsplat/cuda/_backend.py` passes the GCC flag
  `-Wno-attributes` to MSVC (`cl : D8021 invalid numeric argument`). Build from source instead (above); `setup.py`
  handles Windows flags. After that there is no compile at import time.
* Verified on this machine: StackCube rgb+depth+seg renders, `segmentation_id_map` ids match CLAUDE.md §5,
  re-render after `set_qpos` works; `test_geometry.py` + `test_trajectory.py` pass (4/4); camera-helper check of
  `test_sim_integration.py` passes (run standalone, since that file imports the planner); pytorch_kinematics FK vs
  sim link positions 3.6e-7 m (panda_v3); gsplat `rasterization` with 20 colour channels, `RGB+ED`, `packed=False`
  → render `[1,128,128,21]`, `radii [C,N,2]`, meta has `means2d`, `gaussian_ids`, `width`, `height`, `n_cameras`;
  backward works; 1.3 ms per 128² frame (1k Gaussians).

## Findings that affect everyone

* **CHTC Blackwell GPUs (RTX PRO 6000, GB10) fail with torch 2.5.1** ("no kernel image is available"). Add
  `gpus_minimum_capability = 8.0` and `gpus_maximum_capability = 9.0` to every GPU submit file. CLAUDE.md §3.5.
* `mplib` (ManiSkill's planner backend) only has Linux x86_64 wheels. On a Mac `pip install mani_skill` works
  (mplib is Linux-only in its requirements) but the motion planner cannot run. Use `docker/Dockerfile.cpu`.
* Debian bookworm's software Vulkan (Mesa 22.3) is rejected by SAPIEN 3.0.3 ("Failed to find a supported physical
  device"); Debian trixie (Mesa 25) works, so the image uses it. The driver file name differs by distro
  (`lvp_icd.json` on trixie, `lvp_icd.x86_64.json` on bookworm). Added to CLAUDE.md §3.3.
* Confirmed: env creation fails without a Vulkan device even with `render_backend="none"`.
* **StackCube-v1 also defaults to `max_episode_steps=50`** (CLAUDE.md §5 only says PickCube). Our demos are
  113–260 steps, so always override it. Added to CLAUDE.md §5.
