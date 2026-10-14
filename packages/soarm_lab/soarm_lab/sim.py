"""The compiled station scene the lab trains and evaluates in: model, state, the arm's kinematics, renders.

It holds what the environments, the GPU batch and the evaluation use. The scene comes from an exported
station (`station.Station`) plus the task's objects; `SceneConfig` says which.

    cube = Box("cube", (0.015, 0.015, 0.015), (0.2, 0.0))
    cfg = SceneConfig(Station.load("sim"), objects=[cube], grasp_sensors=["cube"])
    scene = build(cfg)
    scene.set_joints({"shoulder_pan": 10.0})
    frame = scene.render("sym-lewa")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .kinematics import RobotKinematics, inverse, pose
from .robot import RobotSpec
from .station import CV_TO_MJ, PREFIX, Box, CameraView, Station

__all__ = ["Box", "CameraView", "Scene", "SceneConfig", "build"]


@dataclass
class SceneConfig:
    """A station and what a task adds to it."""

    station: Station
    objects: list[Box] = field(default_factory=list)
    #: Objects that get a contact sensor with each jaw: `<object>_jaw0` (fixed), `<object>_jaw1` (moving).
    grasp_sensors: list[str] = field(default_factory=list)
    #: Every camera of the scene: the station's (None = take them) and any added here.
    cameras: list[CameraView] | None = None

    def __post_init__(self) -> None:
        if self.cameras is None:
            self.cameras = list(self.station.cameras)

    @property
    def robot(self) -> RobotSpec:
        return self.station.spec()


class Scene:
    """A compiled scene: model, state, the arm's kinematics and renders from its cameras."""

    def __init__(self, cfg: SceneConfig, model: mujoco.MjModel):
        self.cfg = cfg
        self.model = model
        self.data = mujoco.MjData(model)
        spec = cfg.robot
        self.kin = RobotKinematics(spec, model, PREFIX)
        self.act_ids = np.array([model.actuator(PREFIX + j).id for j in spec.joints])
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        mujoco.mj_forward(model, self.data)
        base = model.body(PREFIX + spec.base_body).id
        self.T_base2world = pose(self.data.xmat[base].reshape(3, 3), self.data.xpos[base])

    # ------------------------------------------------------------------ the arm

    def set_joints(self, joints: dict[str, float], *, hold: bool = True) -> None:
        """Puts the arm in a pose at once (no dynamics) and, by default, holds it there with the servos."""
        q = self.kin.to_q(joints)
        self.kin.write_qpos(self.data.qpos, q)
        self.kin.stop_qvel(self.data.qvel)
        if hold:
            self.data.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.model, self.data)

    def command(self, joints: dict[str, float]) -> None:
        """Servo targets: the arm drives there over the next physics steps."""
        self.data.ctrl[self.act_ids] = self.kin.to_q(joints)

    def joints(self) -> dict[str, float]:
        """Joint positions in app units."""
        return self.kin.from_q(self.data.qpos[self.kin.qadr])

    def step(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        mujoco.mj_step(self.model, self.data, nstep=n)
        # After `mj_step` body and site poses are one substep older than qpos: recompute them from
        # the same qpos (as the environments do).
        mujoco.mj_kinematics(self.model, self.data)

    # ------------------------------------------------------------------ cameras

    def render(self, camera: str) -> np.ndarray:
        """An RGB frame from a scene camera at its own resolution."""
        view = self.camera(camera)
        key = (view.width, view.height)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height=view.height, width=view.width)
            self._renderers[key] = r
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def camera(self, name: str) -> CameraView:
        for view in self.cfg.cameras or []:
            if view.name == name:
                return view
        raise KeyError(f"brak kamery {name!r} w scenie")

    def camera_pose(self, name: str) -> np.ndarray:
        """A camera's pose relative to the base in the OpenCV convention, read from the MODEL."""
        c = self.model.camera(name).id
        T_world = pose(self.data.cam_xmat[c].reshape(3, 3) @ CV_TO_MJ, self.data.cam_xpos[c])
        return inverse(self.T_base2world) @ T_world

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build(cfg: SceneConfig) -> Scene:
    """Compiles the station's scene with the task's objects, sensors and the cameras it does not have yet."""
    own = {view.name for view in cfg.station.cameras}
    extra = [view for view in cfg.cameras or [] if view.name not in own]
    return Scene(cfg, cfg.station.build(objects=cfg.objects, grasp_sensors=cfg.grasp_sensors, cameras=extra))
