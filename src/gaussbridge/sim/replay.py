"""A3: replay an Episode's pd_joint_pos actions in ManiSkill from its start state (CLAUDE.md §8.3, §10.1).
Checks that a motion is physically executable: planner demos, round-tripped motions, generated samples."""
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

HOLD_STEPS = 10  # repeat the last action so the scene settles before success is read


def make_env(env_id):
    return gym.make(env_id, obs_mode="none", control_mode="pd_joint_pos", sim_backend="physx_cpu",
                    render_backend="sapien_cpu", max_episode_steps=100_000)


def replay(ep, env=None, hold_steps=HOLD_STEPS):
    """Reset to the episode's start (env.reset(seed=meta["seed"])), run ep.actions() + hold steps.
    -> {"success": task success at the last step, "obj_end_err": [K] m, object centres vs ep.obj_pose[-1]}.
    Pass an env from make_env to reuse one simulator across many episodes."""
    own = env is None
    env = env or make_env(ep.meta["env_id"])
    env.reset(seed=ep.meta["seed"])
    u = env.unwrapped

    def obj_pos():
        return np.stack([getattr(u, n).pose.p[0].numpy() for n in ep.obj_names])

    if np.abs(obj_pos() - ep.obj_pose[0, :, :3]).max() > 1e-3:  # otherwise the result means nothing
        raise ValueError(f"seed {ep.meta['seed']} does not reproduce the episode's start state")
    actions = ep.actions()
    for a in np.concatenate([actions, np.repeat(actions[-1:], hold_steps, 0)]):
        _, _, _, _, info = env.step(a)
    out = {"success": bool(info["success"].item()),
           "obj_end_err": np.linalg.norm(obj_pos() - ep.obj_pose[-1, :, :3], axis=-1)}
    if own:
        env.close()
    return out
