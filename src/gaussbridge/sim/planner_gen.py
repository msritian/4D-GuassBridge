"""A1: randomised planner demos (CLAUDE.md §6.1).

Wraps ManiSkill's official StackCube / PickCube motion-planning solutions with random "knobs", so one start
state (seed) yields many different valid motions. Output per shard: ManiSkill's own <out>.h5 / <out>.json
(successful episodes only) plus <out>.knobs.json = {episode_id: {seed, ref_idx, attempt, knobs, num_steps}}.
"""
import dataclasses
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

import mani_skill.envs  # noqa: F401  (registers the envs)
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils.wrappers.record import RecordEpisode

FINGER_LENGTH = 0.025  # same as the official solutions


@dataclasses.dataclass
class Knobs:
    grasp_yaw_k: int        # extra yaw of k * 90 deg on the top grasp (cube is 4-fold symmetric)
    reach_h: float          # pre-grasp height above the grasp (m)
    lift_h: float           # lift height after grasping (m)
    place_h: float          # extra pre-place height above the target, 0 = no waypoint (StackCube only)
    via: list | None        # offset of a mid-transport waypoint from the lift/target midpoint (m)
    speed: float            # scales the planner's joint velocity AND acceleration limits
    rrt_transport: bool     # RRTConnect instead of screw motion for transport moves (falls back to screw)


def sample_knobs(rng: np.random.Generator) -> Knobs:
    return Knobs(
        grasp_yaw_k=int(rng.integers(4)),
        reach_h=float(rng.uniform(0.03, 0.10)),
        lift_h=float(rng.uniform(0.06, 0.20)),
        place_h=0.0 if rng.random() < 0.5 else float(rng.uniform(0.0, 0.08)),
        via=None if rng.random() < 0.5 else [float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.08, 0.08)),
                                             float(rng.uniform(0.0, 0.08))],
        speed=float(rng.uniform(0.4, 0.9)),
        rrt_transport=bool(rng.random() < 0.15),
    )


def _pick_and_carry(env, seed, knobs: Knobs, obj, target_fn, release: bool):
    """reset -> reach -> grasp -> close -> lift -> [via] -> [pre-place] -> target -> [open].
    Returns (info of the last env step, or None if planning failed; knobs actually used)."""
    env.reset(seed=seed)
    u = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=u.agent.robot.pose, visualize_target_grasp_pose=False,
        print_env_info=False, joint_vel_limits=knobs.speed, joint_acc_limits=knobs.speed)

    approaching = np.array([0, 0, -1])
    closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    g = compute_grasp_info_by_obb(get_actor_obb(obj), approaching=approaching, target_closing=closing,
                                  depth=FINGER_LENGTH)
    grasp0 = u.agent.build_grasp_pose(approaching, g["closing"], g["center"])
    for i in range(4):  # sampled yaw first, then the next reachable one
        k = (knobs.grasp_yaw_k + i) % 4
        grasp = grasp0 * sapien.Pose(q=euler2quat(0, 0, k * np.pi / 2))
        if planner.move_to_pose_with_screw(grasp, dry_run=True) != -1:
            break
    else:
        return None, knobs
    knobs = dataclasses.replace(knobs, grasp_yaw_k=k)

    def move(pose, rrt=False):
        if rrt:
            res = planner.move_to_pose_with_RRTConnect(pose)
            if res != -1:
                return res
        return planner.move_to_pose_with_screw(pose)

    lift = sapien.Pose([0, 0, knobs.lift_h]) * grasp
    if move(grasp * sapien.Pose([0, 0, -knobs.reach_h])) == -1 or move(grasp) == -1:
        return None, knobs
    planner.close_gripper()
    if move(lift) == -1:
        return None, knobs

    target = target_fn(u, lift)
    waypoints = []
    if knobs.via is not None:
        waypoints.append(sapien.Pose((lift.p + target.p) / 2 + np.array(knobs.via), lift.q))
    if knobs.place_h > 0:
        waypoints.append(sapien.Pose(target.p + np.array([0, 0, knobs.place_h]), lift.q))
    for pose in waypoints + [target]:
        res = move(pose, knobs.rrt_transport)
        if res == -1:
            return None, knobs
    if release:
        res = planner.open_gripper()
    planner.close()
    return res[-1], knobs


def _stack_target(u, lift):
    """TCP pose that puts cubeA on cubeB: shift the TCP by the cube's offset to its goal (official solution)."""
    goal = u.cubeB.pose * sapien.Pose([0, 0, (u.cube_half_size[2] * 2).item()])
    offset = (goal.p - u.cubeA.pose.p).cpu().numpy()[0]
    return sapien.Pose(lift.p + offset, lift.q)


def solve_stack_cube(env, seed, knobs):
    return _pick_and_carry(env, seed, knobs, env.unwrapped.cubeA, _stack_target, release=True)


def solve_pick_cube(env, seed, knobs):
    knobs = dataclasses.replace(knobs, place_h=0.0)  # no pre-place waypoint for PickCube (§6.1)
    target = lambda u, lift: sapien.Pose(u.goal_site.pose.sp.p, lift.q)  # noqa: E731
    return _pick_and_carry(env, seed, knobs, env.unwrapped.cube, target, release=False)


SOLVERS = {"StackCube-v1": solve_stack_cube, "PickCube-v1": solve_pick_cube}


def generate(env_id, seed_start, num_seeds, refs_per_seed, out, max_attempts=3):
    """Plan refs_per_seed successful demos per seed (up to max_attempts tries each; failures are discarded).
    Writes <out>.h5, <out>.json, <out>.knobs.json. Returns summary stats."""
    out = Path(out)
    env = gym.make(env_id, obs_mode="none", control_mode="pd_joint_pos", sim_backend="physx_cpu",
                   render_backend="sapien_cpu", max_episode_steps=1000)  # both tasks default to 50 steps
    env = RecordEpisode(env, output_dir=str(out.parent), trajectory_name=out.name, save_video=False,
                        record_reward=False, save_on_reset=False, source_type="motionplanning",
                        source_desc="GaussBridge randomised planner (A1)")
    sidecar, attempts = {}, 0
    for seed in range(seed_start, seed_start + num_seeds):
        for ref in range(refs_per_seed):
            for attempt in range(max_attempts):
                knobs = sample_knobs(np.random.default_rng([seed, ref, attempt]))
                try:
                    info, knobs = SOLVERS[env_id](env, seed, knobs)
                except Exception as e:  # mplib can raise on bad configurations; treat as a failed attempt
                    print(f"seed {seed} ref {ref} attempt {attempt}: {e!r}")
                    info = None
                ok = info is not None and bool(info["success"].item())
                env.flush_trajectory(save=ok)
                attempts += 1
                if ok:  # episode ids count saved episodes only, starting at 0
                    n = int(info["elapsed_steps"].item())
                    sidecar[str(len(sidecar))] = dict(seed=seed, ref_idx=ref, attempt=attempt,
                                                      knobs=dataclasses.asdict(knobs), num_steps=n)
                    break
    env.close()
    Path(f"{out}.knobs.json").write_text(json.dumps(sidecar, indent=1))
    lengths = [v["num_steps"] for v in sidecar.values()]
    return dict(attempts=attempts, saved=len(sidecar), success_per_attempt=len(sidecar) / max(attempts, 1),
                len_min=min(lengths, default=0), len_mean=float(np.mean(lengths)) if lengths else 0.0,
                len_max=max(lengths, default=0))
