"""The SO-101 description as the lab reads it: `robot.yaml` of `soarm.ROBOT_DIR` as a dataclass.

`robot.yaml` follows the twin's robot schema. The lab reads only the fields its kinematics, tasks and
strategy use, with the schema's names, types and defaults. Keys it does not use (panel labels,
connections, staged home) are skipped, so a new field in `robot.yaml` does not break the lab. The fields
it reads are built strictly by `soarm.settings.from_mapping`, the cameras on the arm included.

App units are LeRobot's: joints in degrees, the gripper 0..100. `kinematics.RobotKinematics` maps
the gripper linearly onto `RobotSpec.gripper_range_q` (closed, open) in MJCF units.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from soarm import ROBOT_DIR
from soarm.settings import from_mapping

ROBOT_FILE = "robot.yaml"


@dataclass(frozen=True)
class JawThresholds:
    """Jaw thresholds in the MJCF units of the gripper joint (rad for a hinge, m for a slide)."""

    #: The jaw "stands still": it moved less than this between ticks.
    still: float = 0.02
    #: The jaw is wider than commanded / than closed by this much = something holds it apart.
    margin: float = 0.05
    #: The most the jaw may creep closed during a success streak.
    creep: float = 0.03
    #: Wider than this = surely empty.
    open: float = 0.40


@dataclass(frozen=True)
class Desk:
    """Where the arm works on the table [m, rad], in the base frame.

    The lab uses the goal rings and cube placement of the RL tasks and `map_side` (the scale of the
    preview camera). The other fields belong to the twin's calibration and map; they are here so the
    `desk:` section reads strictly.
    """

    goal_radius: tuple[float, float] = (0.13, 0.32)
    goal_height: tuple[float, float] = (0.03, 0.22)
    cube_radius: tuple[float, float] = (0.15, 0.25)
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    wave_radius: tuple[float, float] = (0.12, 0.36)
    search_centre: tuple[float, float, float] = (0.2, 0.0, 0.05)
    search_dist: tuple[float, float] = (0.4, 0.9)
    camera_dist: tuple[float, float] = (0.45, 0.75)
    camera_centre: tuple[float, float, float] = (0.20, 0.0, 0.06)
    map_centre: tuple[float, float] = (0.22, 0.0)
    map_side: float = 0.56


Row4 = tuple[float, float, float, float]
Matrix4 = tuple[Row4, Row4, Row4, Row4]


@dataclass(frozen=True)
class RobotCamera:
    """A camera riding on a body of the arm, e.g. a wrist camera, as the twin's schema has it.

    An ideal pinhole: square pixels and the principal point at the image centre, so `fovy_deg` (the vertical
    field of view) and the size say everything about the image. The twin renders it from the arm's pose;
    with a real arm it serves the workspace camera of the same name.
    """

    #: MJCF body the camera is fixed to.
    body: str
    #: T_cam2body, 4x4, in OpenCV camera axes: z forward, y down, x right.
    pose: Matrix4
    #: The vertical field of view [deg].
    fovy_deg: float
    #: The rendered image's size [px].
    width: int
    height: int


@dataclass(frozen=True)
class RobotSpec:
    """The part of the robot description the lab uses."""

    name: str
    title: str
    #: MJCF of the arm alone (no table, no light), relative to `root`.
    mjcf: str
    #: Joints in app order. Names as in the MJCF.
    joints: tuple[str, ...]
    #: Which joint is the gripper (app units 0..100); None = no gripper.
    gripper: str | None
    #: Tool point site: the TCP of the tasks and of IK.
    tcp_site: str
    #: Body that carries the calibration card held in the jaws.
    hand_body: str
    #: Base body: the frame of goals, cubes and camera poses.
    base_body: str
    #: The robot's folder; `robot.yaml` never sets it.
    root: Path = Path()
    #: Rest pose in app units.
    home: dict[str, float] = field(default_factory=dict)
    #: Approach direction and jaw closing axis, both in the `tcp_site` frame.
    tcp_approach: tuple[float, float, float] = (1.0, 0.0, 0.0)
    tcp_closing: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Geoms of both jaw tips.
    fingertips: tuple[str, str] = ("", "")
    #: Joint that rolls the tool about the approach direction (`roll_penalty` of reach).
    roll_joint: str | None = None
    #: Ranges [app units] reach goals are sampled from (joint space, then FK).
    wave_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Bodies of both jaws (fixed, moving): grip contact sensors of the RL tasks.
    jaw_bodies: tuple[str, str] = ("", "")
    #: Cameras riding on bodies of the arm, by name (`robot.yaml` `cameras:`); what a strategy sees from the
    #: arm.
    cameras: dict[str, RobotCamera] = field(default_factory=dict)
    #: Joint limits [app units]: tighter than the mechanical ones in the MJCF.
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Highest commanded speed [app units / s].
    max_vel: dict[str, float] = field(default_factory=dict)
    #: Gripper 0..100 maps linearly onto (closed, open) in the gripper joint's MJCF units.
    gripper_range_q: tuple[float, float] | None = None
    #: MJCF joints coupled to another joint: joint -> (app joint, ratio).
    mirror_joints: dict[str, tuple[str, float]] = field(default_factory=dict)
    jaw: JawThresholds = JawThresholds()
    desk: Desk = Desk()

    @property
    def mjcf_path(self) -> Path:
        return self.root / self.mjcf

    @property
    def arm_joints(self) -> tuple[str, ...]:
        return tuple(j for j in self.joints if j != self.gripper)


def load_robot(folder: str | Path = ROBOT_DIR) -> RobotSpec:
    """The description in `folder/robot.yaml` (default: the SO-101 of `soarm`).

    Keys the lab does not use are skipped; the ones it uses are built strictly. A joint named in
    `limits`, `home` or `roll_joint` but missing from `joints` is an error.
    """
    folder = Path(folder).resolve()
    path = folder / ROBOT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{folder}: brak pliku {ROBOT_FILE}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: oczekiwano mapowania YAML")
    known = {f.name for f in fields(RobotSpec)} - {"root"}
    used = {k: v for k, v in data.items() if k in known}
    spec = from_mapping(RobotSpec, {**used, "root": folder}, str(path))
    _check(spec, path)
    return spec


def _check(spec: RobotSpec, path: Path) -> None:
    joints = set(spec.joints)
    problems = []
    if not spec.joints or len(joints) != len(spec.joints):
        problems.append("`joints` musi byc niepusta lista bez powtorzen")
    if spec.gripper is not None and spec.gripper not in joints:
        problems.append(f"chwytak {spec.gripper!r} nie jest na liscie `joints`")
    if spec.gripper is not None and spec.gripper_range_q is None:
        problems.append("ramie z chwytakiem potrzebuje `gripper_range_q` (zamkniety, otwarty) "
                        "w jednostkach MJCF")
    missing = joints - set(spec.limits)
    if missing:
        problems.append(f"`limits` nie ma stawow {sorted(missing)}")
    named = {
        "home": set(spec.home),
        "wave_ranges": set(spec.wave_ranges),
        "roll_joint": {spec.roll_joint} - {None},
        "mirror_joints": {src for src, _ in spec.mirror_joints.values()},
    }
    for table, names in named.items():
        if names - joints:
            problems.append(f"`{table}` wymienia stawy spoza `joints`: {sorted(names - joints)}")
    if not spec.mjcf_path.is_file():
        problems.append(f"nie ma pliku modelu {spec.mjcf_path}")
    problems.extend(_camera_problems(spec))
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))


def _camera_problems(spec: RobotSpec) -> list[str]:
    """What is wrong with the cameras on the arm, with the twin's texts. Whether each camera's body exists
    is checked by the kinematics when the camera is used."""
    problems = []
    for name, camera in spec.cameras.items():
        where = f"`cameras.{name}`"
        if not _is_rigid(camera.pose):
            problems.append(f"{where}: `pose` musi byc przeksztalceniem sztywnym "
                            "(obrot, przesuniecie, ostatni wiersz 0 0 0 1)")
        if not 0.0 < camera.fovy_deg < 180.0:
            problems.append(f"{where}: `fovy_deg` musi lezec miedzy 0 a 180")
        if not _is_size(camera.width) or not _is_size(camera.height):
            problems.append(f"{where}: `width` i `height` musza byc dodatnimi liczbami calkowitymi")
    return problems


def _is_rigid(pose: object) -> bool:
    try:
        T = np.asarray(pose, float)
    except (TypeError, ValueError):
        return False
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    orthonormal = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    right_handed = bool(np.isclose(np.linalg.det(R), 1.0, atol=1e-6))
    return bool(orthonormal and right_handed and np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0]))


def _is_size(value: object) -> bool:
    # YAML `true` is a bool, and a bool is an int in Python.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
