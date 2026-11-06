# GaussBridge task board

Each task states the **goal** and **done when**. How to get there is up to the owner: design it, note important
choices in the Decision log in `CLAUDE.md`, and agree interfaces with whoever is on the other side of a hand-off.
Pick a task whose dependencies are done, add your name, keep the status current, and paste the evidence
(numbers, plots, links) when you finish.

Status: todo · in progress · blocked · done

---

## Setup (everyone, first days)

**S1. Shared repo and working environment** — owner: Shivam — status: todo
Goal: a repo everyone can install, with tests and a quick-start README; ManiSkill runs on every laptop.
Done when: each teammate can run a ManiSkill motion-planning demo for StackCube locally.

**S2. Cluster ready** — owner: Aditya — status: todo
Goal: a versioned Docker image and job templates for CHTC CPU and GPU jobs; shared staging space.
Done when: a CPU job generates planner demos and a GPU job renders with both ManiSkill and gsplat.

**S3. Interface agreement** — owner: all — status: todo
Goal: agree on what crosses workstreams (motion episodes, Gaussian assets, renders, generated demos), what
each contains, and the conventions in `CLAUDE.md` §5.
Done when: a short written spec for each shared format is in the repo.

---

## A. Simulation & data (Aditya)

**A1. Diverse example motions** — deps: S1
Goal: a generator that solves StackCube and PickCube many times from the same start with genuinely different
valid motions.
Done when: several starts × several references each, high success rate, and a plot showing references from the
same start clearly differ.

**A2. Motion episodes from simulator files** — deps: A1, S3
Goal: convert simulator trajectories into our shared episode format, including when each object is held.
Done when: episodes round-trip (our format → actions → simulator replay) and still succeed.

**A3. Full datasets** — deps: A2, S2
Goal: train / validation / test splits (test with many references per start for coverage metrics), plus a
separate set of "real" demos for the policy; disjoint seed ranges; basic statistics.
Done when: datasets are on staging with a README and stats.

**A4. Policy baseline reproduced** — deps: S2
Goal: ManiSkill's Diffusion Policy trained on our real demos with a few dataset sizes.
Done when: a success-rate-vs-number-of-demos curve exists (the reference line for the main result), with GPU
memory use noted.

**A5. Export generated data for the policy** — deps: B4, C5
Goal: turn generated trajectories plus their renders into demos the policy baseline can train on, mixed with
real demos.
Done when: the baseline trains on a mixed dataset without changes to its code.

**A6. Downstream experiments** — deps: A4, A5, C6
Goal: the comparison arms in `CLAUDE.md` §6 on StackCube (then PickCube), several seeds.
Done when: a results table and plot of success rate per arm.

---

## B. Gaussians & rendering (Amogh)

**B1. Multi-view captures from simulation** — deps: S1
Goal: render the scene from many cameras with depth and part segmentation, for random robot/object
configurations (asset mode) and for a scene's start and end (scene mode).
Done when: captures are geometrically consistent (back-projected parts land on their masks).

**B2. Part-structured Gaussian reconstruction** — deps: B1
Goal: one reconstruction in which every Gaussian belongs to a rigid part in its own frame, so robot links and
objects can be posed arbitrarily.
Done when: renders of held-out configurations look right and score well against ManiSkill's renders
(target ≈ 28 dB PSNR at policy-camera resolution).

**B3. Pose-and-render** — deps: B2, A2
Goal: given an episode, render every step from the policy's cameras (including the moving wrist camera) with
correct view-dependent colour.
Done when: renders of planner episodes match ManiSkill's renders of the same episodes (side-by-side video plus
metrics).

**B4. Rendering at scale + labels** — deps: B3
Goal: fast batch rendering of many generated episodes; depth, masks, flow and point tracks as extras.
Done when: hundreds of episodes render within an hour; labels pass simple consistency checks.

**B5. Scene-mode evaluation** — deps: B2
Goal: measure how good reconstructions from only start + end snapshots are, with and without a pre-built robot
model.
Done when: render-quality numbers and documented failure cases.

**B6. (Stretch) Real tabletop capture** — deps: B5, week-4 go decision
Goal: before/after phone captures of a real tabletop, objects only, with generated motions rendered in it.
Done when: a qualitative video and measured endpoint accuracy.

---

## C. Motion model & evaluation (Shivam)

**C1. Motion representation** — deps: S3
Goal: decide how trajectories are represented for learning (time normalization, rotations, robot and object
states, what is "known" at start/end) and how to convert back to executable actions.
Done when: conversion round-trips without breaking task success (with A2).

**C2. Goal-conditioned motion model** — deps: C1, a small shard from A2
Goal: a generative model that fills in all intermediate steps given start and end, in both goal modes.
Done when: it overfits a handful of episodes and its samples match the given endpoints exactly.

**C3. Full training** — deps: C2, A3, S2
Goal: a trained model per task, resumable on the cluster, with validation monitoring.
Done when: validation samples look plausible and beat the straight-line baseline.

**C4. Metrics and evaluation harness** — deps: C1
Goal: the motion-quality metrics in `CLAUDE.md` §6 on the test split, producing tables automatically.
Done when: sanity checks pass (planner references vs themselves; straight-line baseline).

**C5. Physically sane sampling** — deps: C3, B2 (part shapes; simple box shapes are fine before then)
Goal: collision avoidance, grasp consistency, filtering of invalid samples, diversity selection.
Done when: these measurably reduce collisions without hurting goal accuracy, and simulator replay success is
reported.

**C6. Baselines and generated datasets** — deps: C3, C4, C5
Goal: forward-only and step-by-step baselines in the same harness, plus generated trajectory sets for the
downstream arms (from new starts with object-only goals).
Done when: the full motion-quality table exists and generated sets are handed to A5.

**C7. Ablations** — deps: C6
Goal: show which design choices matter.
Done when: an ablation table is frozen with configs recorded.

---

## Shared (weeks 5–6)

**W1. Figures**: pipeline diagram, many diverse trajectories from one start/goal, render comparison, success curves.
**W2. Report and presentation**: CVPR template; every number traceable to a results file.
