# GaussBridge — project brief

This file tells any agent or teammate **what we are building, why, how it works at a conceptual level, what
"done" looks like, and what we already learned the hard way**. It deliberately does not prescribe code
structure, class names or function signatures: whoever owns a task designs it, records important choices in
the Decision log at the bottom, and keeps `TASKS.md` up to date.

Team (CS771 course project, UW–Madison): Shivam Mittal, Amogh Dixit, Aditya Kumar.
Schedule assumption: Nov 9 – Dec 18, 2026 (confirm the course deadline).

---

## 1. What we are building

**Title:** *GaussBridge: Gaussian World Models for Diverse Start-to-Goal 4D Trajectory Generation*

**Problem.** Given images of a scene at a **start** time and at a later **end** time, generate *many*
different, realistic ways the scene could have evolved in between. Each trajectory should be a complete 3D world
at every time step that we can render from any camera, with exact labels for free (depth, motion, masks, and,
for robots, the actions).

**Why it matters.** Dynamic 3D data (and robot demonstrations in particular) is expensive to collect and label.
Snapshots of a start and a goal are cheap. If we can turn two snapshots into hundreds of plausible, physically
sensible trajectories, we have a data engine for training policies and perception models.

**Core idea.**
1. Reconstruct the scene **once** with 3D Gaussian splatting, split into rigid **parts** (static background,
   each robot link, each object). Each part can be moved independently without re-reconstructing anything.
2. Learn a **goal-conditioned motion model** that generates how the moving parts move (robot joint angles and
   object poses) across all intermediate steps, **conditioned on both the start and the end**, and sampled many
   times for diversity.
3. The world at any step = the same Gaussians with each part moved to its generated pose. Render it from the
   cameras we need.

**What is new (claims we must defend in the report).**
* Existing Gaussian world models (ManiGaussian, GWM, 4DGS-WAM) only predict forward from the current state;
  none condition on an observed goal.
* Goal-conditioned world models (e.g. Act2Goal) work in 2D video, without an explicit 3D state or exact labels.
* Recent Gaussian demonstration-synthesis systems (RoboSplat 2025; single-scan synthesis, arXiv 2609.21112;
  robot-aligned 3DGS for VLA reaching, arXiv 2610.35626) use motion planners or transform an existing demo.
  We learn a *stochastic* motion prior that produces diverse, valid motions between two observed states.

**The one result that matters most:** a robot policy trained on a few real demos **plus our generated demos**
succeeds more often than one trained on the real demos alone, and comes close to one given extra
"perfect" planner demos.

---

## 2. How it works (conceptual pipeline)

```
 simulator + randomized planner ──► many example motions ──────────────┐
                                                                        ▼
 multi-view captures ──► part-structured Gaussians        goal-conditioned motion model
        (robot links, objects, table)                      (trained once per task)
                │                                                       │
 start + end snapshot ──► start/end pose of every part ──► sample many trajectories
                                                                        │
                                       check physics, keep diverse ones ▼
                     move Gaussian parts per step ──► render robot's cameras + labels
                                                                        │
                                   export as demonstrations ──► train a policy ──► success rate
```

**Stage 1 — Example motions (training data).** We use the ManiSkill simulator and its motion planner to solve
tasks many times *from the same start* with randomized choices (grasp orientation, heights, waypoints, speed),
so the model sees genuinely different valid ways to do the same thing. The planner is our source of
"realistic motion"; later work could add real or human motion data.

**Stage 2 — Gaussian world representation.** Every Gaussian belongs to one rigid part and lives in that part's
own coordinate frame. A scene state is just a pose per part (robot link poses come from forward kinematics of
the joint angles). Two ways to build it:
* *Asset mode* (simulation, efficient): capture the robot and objects in many random poses from many cameras,
  so every side is seen, and reconstruct robot, objects and table once per task.
* *Scene mode* (the real use case): reconstruct from only the start and end captures of one scene.
  Used to measure how well "two snapshots" works.
View-dependent colour must stay correct when parts rotate; that needs care (see §4).

**Stage 3 — Motion model.** A transformer-based generative model (diffusion / flow matching) over the
trajectory of all moving parts. Design principles we agreed on:
* Generate **poses, not pixels or Gaussians**: small, fast, and appearance stays exactly as captured.
* Generate **all time steps jointly** with the start and end **given as fixed inputs**, so samples land exactly
  on the goal and can plan ahead (avoid obstacles early). A step-by-step version is a baseline, not the method.
* Support two goal modes: everything's end pose known, or only the objects' end poses known (robot free).
* Condition on what the objects look like/how big they are (from their Gaussians), so the model generalizes.

**Stage 4 — Sampling with physical sanity.** Draw many candidates per start/end pair, then:
* steer away from collisions (simple sphere approximations of parts are enough),
* make grasps consistent: an object only moves while it is held, and then moves rigidly with the gripper,
* discard invalid samples, and keep a diverse subset.
Optionally verify a sample by executing its actions in the simulator.

**Stage 5 — Rendering and labels.** Compose the Gaussians for each step and render from the cameras the
downstream model uses (for our policy: a fixed camera and, for StackCube, the wrist camera, which moves with
the robot). Depth, masks, flow and point tracks come for free because we know where every Gaussian is.

**Stage 6 — Downstream proof.** Export generated trajectories in the format ManiSkill's Diffusion Policy
baseline reads, train policies with and without them, compare success rates in simulation.

---

## 3. Decisions already made (and why)

| Decision | Why |
|---|---|
| Simulation first: ManiSkill 3 with a Franka Panda; StackCube (main) and PickCube (fast dev) | Free ground truth, a built-in motion planner, and a ready Diffusion Policy baseline; no robot hardware needed |
| Rigid objects; objects move only when grasped | Keeps v1 tractable; true for both tasks |
| Downstream = ManiSkill's Diffusion Policy, RGB input, success rate in sim | Standard, cheap to run, directly measures usefulness of the data |
| Training motions from a randomized planner | Only practical source of many diverse, valid motions per start state |
| Downstream generated data uses object-only goals (robot end left free) | Matches the realistic use case: a user specifies where things should end up |
| Baselines: straight-line interpolation, forward-only model (no end condition), step-by-step model; planner as upper bound | Each isolates one part of our claim (realism, goal conditioning, joint generation) |
| Real-world capture is a stretch goal, decided at the week-4 gate | Core results must not depend on hardware/capture logistics |

Anything else is open: architecture size, losses, guidance details, data sizes, render resolution, etc.
Choose, justify briefly, and log it in the Decision log.

---

## 4. Things we already verified (save yourself the time)

Checked on 2026-11-05 by reading source and running ManiSkill locally.

**Environment**
* Use **Python 3.11**. ManiSkill 3.0.1 depends on `mplib`, which has no wheels for Python 3.13.
* ManiSkill pins `pytorch_kinematics==0.7.6`; don't upgrade it. Its forward kinematics matched the
  simulator's robot link poses to under a micrometre.
* ManiSkill needs a Vulkan device **even when not rendering** (it loads URDF materials). On machines without an
  NVIDIA Vulkan driver (laptops, CHTC CPU nodes), install software Vulkan (`mesa-vulkan-drivers`, set
  `VK_ICD_FILENAMES` to the lavapipe ICD) and use the CPU render backend. On NVIDIA machines you need the
  NVIDIA ICD files (ManiSkill's `docker/` folder has them; A100s also need the layers file).
* gsplat: use the **1.5.3 release**. The PyPI wheel compiles CUDA kernels on first use (needs `nvcc`, i.e. a
  CUDA "devel" image). The GitHub main branch has a different API; don't read its docs.

**Simulation and data**
* The motion planner runs headless on one CPU core at roughly **0.5 s per trajectory**. Data generation belongs
  on CPU jobs, not GPU.
* Randomized demos are longer than ManiSkill's official ones (≈ 7–13 s vs ≈ 5 s at 20 Hz). Policies need a
  longer episode limit (ManiSkill suggests ~2× the mean demo length).
* StackCube's default robot has a **wrist camera**, and the policy uses both cameras. PickCube's default robot
  has none. The wrist camera's pose can be computed exactly from the joint angles (we verified the formula).
* Re-sampling trajectories to a fixed 128 steps and back loses nothing important: 21/21 planner demos still
  succeeded when replayed in the simulator.
* ManiSkill's own "is grasped" check uses contact forces, which are stale when you set states without stepping
  physics. A geometric rule works reliably instead: gripper commanded closed, fingers not fully open, and the
  object within ~2 cm of the gripper centre.
* ManiSkill trajectory files store full simulator states per step (object poses, robot joint states), so
  extracting exact object and robot motion is straightforward. `--use-env-states` replay renders any stored
  trajectory with ManiSkill's renderer (useful as ground truth for our renders).

**Policy baseline**
* ManiSkill's Diffusion Policy expects its demo files to contain the full observation structure of the
  environment (robot state, task-specific extras, camera images *and* camera parameters for every camera),
  even keys it doesn't use.
* It loads the whole dataset **into GPU memory**. With 128×128 images from two cameras and ~200-step demos,
  budget roughly 20 MB per demo.

**Gaussian splatting**
* When parts rotate, view-dependent colour (spherical harmonics) must be evaluated with the viewing direction
  expressed in the part's own frame. gsplat will not rotate SH coefficients for you.
* gsplat's densification keeps any extra per-Gaussian tensor that is *not trainable* aligned through splits and
  prunes, which is a clean way to carry "which part does this Gaussian belong to".

---

## 5. Shared conventions (needed wherever workstreams meet)

* Units: metres, radians, seconds. World frame = ManiSkill's (z up, table top at z = 0, Panda base at
  x = −0.615).
* Quaternions in (w, x, y, z) order (ManiSkill and gsplat agree).
* Cameras in OpenCV convention with world-to-camera matrices (what gsplat expects). Note: ManiSkill returns 3×4
  extrinsics and a *vertical* field of view.
* Robot joint order = ManiSkill's (7 arm joints, then 2 finger joints). Gripper command +1 = open, −1 = close.
* Control rate 20 Hz.
* Data files that cross workstreams (motion episodes, Gaussian assets, generated demos) get a short README
  describing their contents. Freeze those interfaces early and change them only by agreement.

---

## 6. What success looks like (evaluation)

**Motion quality** (held-out start/end pairs, compared with several planner references per pair):
does every sample reach the end state; are motions collision-free and smooth; are they realistic (close to
some reference) and diverse (cover the different reference modes rather than random noise); do the actions
actually work when executed in the simulator; how fast is generation.

**Render quality:** how close our Gaussian renders are to ManiSkill's own renders of the same motion, for asset
mode and for scene mode (start + end snapshots only).

**Downstream (headline):** Diffusion Policy success rate with: real demos only; real + ours; real + planner
demos rendered by our Gaussians (oracle motion, same appearance); real + our motions rendered by ManiSkill
(isolates the appearance gap); real + straight-line motions (sanity check). Several seeds, disjoint evaluation
seeds.

**Ablations** should show which design choices matter (e.g. collision guidance, grasp consistency, joint vs
step-by-step generation, amount of training data).

**Minimum bar for a good project:** our samples hit the goal, beat all baselines on validity/realism, and
adding them improves policy success on StackCube.

---

## 7. Team and milestones

Workstreams (owners can swap):
* **A — Simulation & data (Aditya):** example motions, datasets, simulator replay checks, policy baseline,
  exporting generated data, downstream experiments.
* **B — Gaussians & rendering (Amogh):** captures, part-structured reconstruction, composition, rendering and
  labels, render-quality evaluation.
* **C — Motion model & evaluation (Shivam):** motion representation, model, training, sampling with physical
  sanity, baselines, metrics and results tables.

| Week | Gate (Friday) |
|---|---|
| 1 (Nov 9) | Environment works everywhere; first data shard; first reconstruction renders; model overfits a few examples |
| 2 (Nov 16) | Full datasets; Gaussian assets look right in held-out views; model beats straight-line baseline |
| 3 (Nov 23) | Renderer for trajectories; sampling with physical checks; first full motion-quality table |
| 4 (Nov 30) | Downstream policy results on StackCube; go/no-go on real-world stretch |
| 5 (Dec 7) | Ablations, PickCube, figures; all numbers frozen |
| 6 (Dec 14) | Report and presentation |

The workstreams run in parallel for the first two weeks. They converge when rendering (B) and sampling (C)
feed export and policy training (A). The trajectory renderer is the critical path. If behind, cut in this
order: real-world stretch → PickCube downstream → step-by-step baseline → scene-mode reconstruction.

---

## 8. How to work on this project

* Pick a task in `TASKS.md`, put your name on it, and own its design. The task says *what* and *done when*,
  not *how*.
* Before writing code, skim what exists and agree interfaces with the owner on the other side of any hand-off.
* Verify against the installed library source rather than memory or online docs (versions above are pinned).
* Prefer small, tested steps. Frame/convention mistakes are the most likely silent bugs: test geometry against
  the simulator.
* Record non-obvious choices (and numbers you measured) in the Decision log below; update `TASKS.md` status.
* Use CPU wherever possible; GPUs only for Gaussian optimization/rendering and model/policy training.

---

## 9. Compute and environment

* **CHTC (UW–Madison) HTC system.** GPU Lab has H200, H100, A100, L40/L40S. Job lengths short (12 h), medium (24 h,
  default), long (7 days, ≤ 4 GPUs per user). Short jobs that checkpoint can use backfill ("resumable").
  Interactive jobs for debugging; SSH into running GPU jobs is disabled, use `condor_tail`.
* Run jobs in our own Docker image (CUDA devel base, Vulkan for ManiSkill, gsplat compiled for A100/H100/L40),
  tagged per version; never reuse a tag (CHTC caches images).
* Large data in `/staging` (request a shared group directory). Store few large archives, not many small files.
* CPU jobs start much faster than GPU jobs: use them for data generation.
* Colab Pro: notebooks and quick visualization only.

---

## 10. Literature and positioning

* **3D Gaussian Splatting** (Kerbl et al., 2023); **gsplat** library (Ye et al., 2024).
* **Gaussian world models (forward only):** ManiGaussian (ECCV 2024), GWM (ICCV 2025), 4DGS-WAM (arXiv 2608.25956, 2026).
* **Goal-conditioned generation:** Act2Goal (arXiv 2512.23541, 2D video); Diffuser (ICML 2022; start/goal
  conditioning by inpainting a whole trajectory); In-2-4D (arXiv 2504.08366; single object, single output).
* **Generative modelling tools:** flow matching (Lipman et al., 2023); diffusion transformers (Peebles & Xie,
  2023); continuous 6D rotations (Zhou et al., 2019).
* **Robots in Gaussian scenes / demo synthesis:** SplatSim (2024; robot Gaussians moved per link with forward
  kinematics); RoboGSim (2024); RoboSplat (2025; edits Gaussians and transforms one demo); single-scan demo
  synthesis (arXiv 2609.21112, Sep 2026; planner on a reconstruction, 95% sim / 84% real success); robot-aligned
  3DGS for wrist-camera VLA reaching (arXiv 2610.35626, Oct 2026); MimicGen (2023), DemoGen (2025).
* **Policy:** Diffusion Policy (Chi et al., RSS 2023), via ManiSkill's implementation.

**Positioning:** prior Gaussian demo synthesis either transforms existing demos or calls a planner with
privileged geometry. We learn a stochastic, goal-conditioned motion prior over an explicit part-structured
Gaussian state, producing many physically checked trajectories between two observed snapshots. Expect the
question "why not just use the planner?": the planner needs full simulator state and object models; our method
needs only the captures. The planner is reported as an upper bound.

Check arXiv weekly for: goal-conditioned Gaussian world models, 4D Gaussian inbetweening, Gaussian splatting
demonstration synthesis.

---

## 11. Open questions

* Course deadline and report format (CVPR template like the proposal?).
* Shared CHTC staging directory and W&B project.
* Real-world stretch: objects-only tabletop captures with a phone (no robot needed) — decide at week 4.

---

## 12. Decision log

Record decisions that others depend on: date, owner, decision, one-line reason, measured numbers if any.

| Date | Owner | Decision | Reason / evidence |
|---|---|---|---|
| 2026-11-05 | all | v1 = ManiSkill StackCube + PickCube, Panda, Diffusion Policy downstream | see §3 |
