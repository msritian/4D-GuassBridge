# 4D-GuassBridge
Given a few multi-view frames of a scene at a start time and a few frames of the same scene at a later end time, we want to generate what happened in between: many different trajectories, each as realistic as what could actually have occurred in the real world between those two states, each as a complete 3D world at every time step.

Read `CLAUDE.md` (project brief), `TASKS.md` (task board) and `progress.md` (status) first.

## Quick start
```bash
conda create -n gaussbridge python=3.11 -y && conda activate gaussbridge
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
pip install -e ".[dev]"            # + ".[gs]" for gsplat (Windows: build it from source, see progress.md)
pytest -q                          # planner tests run on Linux only (mplib); CI runs everything on Ubuntu
```
Simulator needs a Vulkan device (CLAUDE.md §3.3); CPU-only / Mac: use `docker/Dockerfile.cpu`.
