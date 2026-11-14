"""C2: Panda forward kinematics (CLAUDE.md §5 "Kinematics"). Torch, differentiable, any leading batch dims.

T_world_link = T_world_base @ FK(qpos)[link], with pytorch_kinematics on the same URDF ManiSkill loads.
Does not import ManiSkill (only reads its URDF files), so it is cheap to use in model training and guidance.
"""
import contextlib
import importlib.util
import os
from pathlib import Path

import pytorch_kinematics as pk
import torch
from pytorch_kinematics.urdf_parser_py.xml_reflection import core as _urdf_xml

from gaussbridge import geometry as G

URDFS = {"panda": "robots/panda/panda_v2.urdf", "panda_wristcam": "robots/panda/panda_v3.urdf"}
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
BASE_POSE = (-0.615, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # pose7 of the robot root in both v1 tasks (§4)
TCP_LINK = "panda_hand_tcp"
HAND_CAMERA_LINK = "camera_link"


def urdf_path(robot_uid):
    """Path of the URDF ManiSkill uses for robot_uid, found without importing mani_skill (which loads SAPIEN).
    If the environment variable GB_URDF_DIR is set, the file is read from <GB_URDF_DIR>/<same relative path>
    instead (for jobs without ManiSkill, e.g. training in the plain PyTorch image; chtc/train.sh ships the files).
    FK only needs the kinematic tree, so the meshes the URDF refers to need not exist."""
    override = os.environ.get("GB_URDF_DIR")
    if override:
        return Path(override) / URDFS[robot_uid]
    spec = importlib.util.find_spec("mani_skill")
    if spec is None:
        raise ImportError("mani_skill is not installed; it provides the Panda URDFs")
    return Path(spec.submodule_search_locations[0]) / "assets" / URDFS[robot_uid]


@contextlib.contextmanager
def _quiet_urdf_parser():
    """The parser warns about every unknown URDF attribute (ManiSkill's joint dynamics); they are harmless."""
    old, _urdf_xml.on_error = _urdf_xml.on_error, lambda message: None
    try:
        yield
    finally:
        _urdf_xml.on_error = old


class PandaFK:
    """Forward kinematics of ManiSkill's Panda. qpos [..., 9] in ManiSkill joint order (§4)."""

    def __init__(self, robot_uid="panda_wristcam", device="cpu", dtype=torch.float32):
        if robot_uid not in URDFS:
            raise ValueError(f"unknown robot_uid {robot_uid!r}; expected one of {sorted(URDFS)}")
        self.robot_uid = robot_uid
        with _quiet_urdf_parser():
            chain = pk.build_chain_from_urdf(urdf_path(robot_uid).read_bytes())
        if chain.get_joint_parameter_names() != JOINT_NAMES:  # FK input order must equal ManiSkill's qpos order
            raise RuntimeError(f"unexpected URDF joint order: {chain.get_joint_parameter_names()}")
        self.chain = chain.to(dtype=dtype, device=device)
        self.link_names = self.chain.get_frame_names(exclude_fixed=False)
        self.dtype, self.device = dtype, device

    def _base(self, base_pose, batch_shape):
        """base_pose: None (default root pose), pose7 [..., 7] or matrix [..., 4, 4] -> [..., 4, 4]."""
        if base_pose is None:
            base_pose = torch.tensor(BASE_POSE)
        base_pose = torch.as_tensor(base_pose).to(dtype=self.dtype, device=self.device)
        T = base_pose if base_pose.shape[-2:] == (4, 4) else G.pose7_to_matrix(base_pose)
        return T.expand(*batch_shape, 4, 4)

    def link_poses(self, qpos, base_pose=None, links=None):
        """{link name: T_world_link [..., 4, 4]} for the requested links (default: every URDF frame)."""
        qpos = torch.as_tensor(qpos).to(dtype=self.dtype, device=self.device)
        batch = qpos.shape[:-1]
        fk = self.chain.forward_kinematics(qpos.reshape(-1, qpos.shape[-1]))
        T_base = self._base(base_pose, batch)
        out = {}
        for name in links or self.link_names:
            if name not in fk:
                raise KeyError(f"{name!r} is not a frame of {self.robot_uid} (frames: {self.link_names})")
            out[name] = T_base @ fk[name].get_matrix().reshape(*batch, 4, 4)
        return out

    def tcp_pose(self, qpos, base_pose=None):
        """T_world_tcp [..., 4, 4] (the point between the fingertips that grasps use)."""
        return self.link_poses(qpos, base_pose, [TCP_LINK])[TCP_LINK]

    def hand_camera_viewmat(self, qpos, base_pose=None):
        """OpenCV world->camera matrix [..., 4, 4] of StackCube's wrist camera (panda_wristcam only)."""
        if self.robot_uid != "panda_wristcam":
            raise ValueError("only panda_wristcam has a hand camera")
        return G.sapien_pose_to_viewmat(self.link_poses(qpos, base_pose, [HAND_CAMERA_LINK])[HAND_CAMERA_LINK])
