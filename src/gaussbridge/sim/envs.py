"""B1: task tables and capture environments (CLAUDE.md §5 "Custom cameras", §7.2).

`StackCubeCapture-v1` / `PickCubeCapture-v1` = the task env (same scene, lighting and policy cameras, which stay
first) plus rings of capture cameras for multi-view Gaussian reconstruction. Importing this module registers them.
"""
import dataclasses
import math

import numpy as np
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

OBJECTS = {"StackCube-v1": ["cubeA", "cubeB"], "PickCube-v1": ["cube"]}
POLICY_CAMERAS = {"StackCube-v1": ["base_camera", "hand_camera"], "PickCube-v1": ["base_camera"]}
DEFAULT_ROBOT = {"StackCube-v1": "panda_wristcam", "PickCube-v1": "panda"}
CAPTURE_ENV = {"StackCube-v1": "StackCubeCapture-v1", "PickCube-v1": "PickCubeCapture-v1"}


@dataclasses.dataclass(frozen=True)
class Ring:
    """n_azimuth cameras per elevation on a sphere of `radius` around `target`, all looking at `target`."""
    name: str
    target: tuple
    radius: float
    elevations_deg: tuple
    n_azimuth: int
    size: int = 256
    fovy_deg: float = 60.0

    def cameras(self):
        """[(uid, eye xyz)] in a fixed order."""
        out = []
        for el in self.elevations_deg:
            for i in range(self.n_azimuth):
                az = 2 * math.pi * i / self.n_azimuth
                d = (math.cos(math.radians(el)) * math.cos(az), math.cos(math.radians(el)) * math.sin(az),
                     math.sin(math.radians(el)))
                eye = tuple(t + self.radius * di for t, di in zip(self.target, d))
                out.append((f"{self.name}_e{el}_a{i:02d}", eye))
        return out

    def camera_configs(self):
        return [CameraConfig(uid, sapien_utils.look_at(eye, self.target), self.size, self.size,
                             math.radians(self.fovy_deg), 0.01, 100) for uid, eye in self.cameras()]


# Wide ring = CLAUDE.md §7.2 (whole robot; about 5 mm per pixel at the target). Close ring: about 2.3 mm per pixel
# around the cube spawn area, because the hand camera sees cubes and fingers from a few cm (finer than the wide ring).
RINGS = (
    Ring("wide", target=(-0.15, 0.0, 0.15), radius=1.1, elevations_deg=(20, 45, 70), n_azimuth=12),
    Ring("close", target=(-0.05, 0.0, 0.08), radius=0.5, elevations_deg=(30, 60), n_azimuth=8),
)


def capture_camera_names(rings=RINGS):
    return [uid for ring in rings for uid, _ in ring.cameras()]


class _CaptureMixin:
    def __init__(self, *args, capture_rings=RINGS, **kwargs):
        self.capture_rings = tuple(capture_rings)  # must exist before the parent builds sensors
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        return list(super()._default_sensor_configs) + [c for r in self.capture_rings for c in r.camera_configs()]


@register_env("StackCubeCapture-v1", max_episode_steps=400)
class StackCubeCaptureEnv(_CaptureMixin, StackCubeEnv):
    task_id = "StackCube-v1"


@register_env("PickCubeCapture-v1", max_episode_steps=400)
class PickCubeCaptureEnv(_CaptureMixin, PickCubeEnv):
    task_id = "PickCube-v1"


def ring_coverage(rings=RINGS):
    """Approximate ground footprint of one pixel at each ring's target (metres), for documentation and tests."""
    return {r.name: float(2 * r.radius * np.tan(math.radians(r.fovy_deg) / 2) / r.size) for r in rings}
