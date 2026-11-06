# Progress

Shared log of what is built, by whom, and what is left. Task IDs follow `TASKS.md` (the detailed board that
`CLAUDE.md` refers to); `TASKS_helper.md` IDs are in brackets where they differ.

**Before starting work:** `git pull`, read this file, and add/claim your row (`in progress` + your name) so nobody
duplicates it. **When you finish:** set the status, list what exists, and paste the acceptance numbers.

Status: todo · in progress · done · blocked

## Task status

| Task | Owner | Status | What exists | Notes / numbers |
|---|---|---|---|---|
| S0.1 Repo scaffold [S1] | Shivam | todo | — | No `pyproject.toml` yet; code imports via `PYTHONPATH=src` (set in the CPU image) |
| S0.2 Laptops [S1] | each | todo | — | Macs can't run the planner natively (see Findings); use the CPU image below |
| S0.3 Docker + CHTC [S2] | Aditya | in progress | `docker/Dockerfile.cpu` (CPU, lavapipe) | CUDA image (CLAUDE.md §3.4) and CHTC submit files still todo |
| A1 Randomised planner | Aditya | done (2026-11-06) | `src/gaussbridge/sim/planner_gen.py`, `scripts/gen_demos.py`, `scripts/plot_refs.py`, `tests/test_planner_gen.py` | 80% / 100% success per attempt; see A1 below |
| A2 Extraction to Episodes | Aditya | todo | — | needs C1 (Episode format) and C2 (FK for the gripper position) |
| A3 Physics replay [A2] | Aditya | todo | — | |
| A4–A7 | Aditya | todo | — | |
| B1–B8 | Amogh | todo | — | |
| C1 Geometry + trajectory core | Aditya (Shivam's task) | done (2026-11-06) | `src/gaussbridge/geometry.py`, `src/gaussbridge/trajectory.py`, `tests/test_geometry.py`, `tests/test_trajectory.py`, `tests/test_sim_integration.py` | 24/24 planner demos replay after the 128-step round trip; see C1 below |
| C2 Kinematics (PandaFK) [none] | Shivam | todo | — | needed by A2 and B1; no owner on the helper board |
| C3–C9 | Shivam | todo | — | |

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

**For the next tasks:** A2 can lift `_episode_from_shard` from `tests/test_sim_integration.py` (it is the extractor
minus the attached rule); A3 can lift the replay loop from the same test.

## Findings that affect everyone

* `mplib` (ManiSkill's planner backend) only has Linux x86_64 wheels. On a Mac `pip install mani_skill` works
  (mplib is Linux-only in its requirements) but the motion planner cannot run. Use `docker/Dockerfile.cpu`.
* Debian bookworm's software Vulkan (Mesa 22.3) is rejected by SAPIEN 3.0.3 ("Failed to find a supported physical
  device"); Debian trixie (Mesa 25) works, so the image uses it. The driver file name differs by distro
  (`lvp_icd.json` on trixie, `lvp_icd.x86_64.json` on bookworm). Added to CLAUDE.md §3.3.
* Confirmed: env creation fails without a Vulkan device even with `render_backend="none"`.
* **StackCube-v1 also defaults to `max_episode_steps=50`** (CLAUDE.md §5 only says PickCube). Our demos are
  113–260 steps, so always override it. Added to CLAUDE.md §5.
