"""Robot description: the `RobotSpec` dataclass and the `robot.yaml` loader.

`RobotSpec` mirrors the twin's `lerobot_mp.twin.robots.RobotSpec` field for field and adds `backend`,
`hardware`, `status` and `root`. Joints are in degrees and the gripper in 0..100, as in LeRobot.
The schema is described in docs/conventions.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

SPEC_FILE = "robot.yaml"
STATUSES = ("ready", "draft")


class SpecError(ValueError):
    """A robot.yaml that breaks the schema."""


@dataclass(frozen=True)
class JawThresholds:
    """Jaw thresholds in MJCF units of the gripper joint (rad for a hinge, m for a slide)."""

    still: float
    margin: float
    creep: float
    open: float


@dataclass(frozen=True)
class Desk:
    """Where the arm works on the table, in the base frame [m, rad]."""

    goal_radius: tuple[float, float]
    goal_height: tuple[float, float]
    cube_radius: tuple[float, float]
    cube_bearing: tuple[float, float]
    wave_radius: tuple[float, float]
    search_centre: tuple[float, float, float]
    search_dist: tuple[float, float]
    camera_dist: tuple[float, float]
    camera_centre: tuple[float, float, float]
    map_centre: tuple[float, float]
    map_side: float


@dataclass(frozen=True)
class Hardware:
    """Servo bus settings shared by every arm of one model."""

    kind: str
    servo_ids: dict[str, int]
    baudrate: int
    center_ticks: int
    gripper_closed_ticks: int | None
    gripper_open_ticks: int | None
    use_degrees: bool
    max_relative_target: float | None
    read_state_hz: float
    torque_off_on_exit: bool


@dataclass(frozen=True, kw_only=True)
class RobotSpec:
    """Everything the twin needs to know about an arm besides the MJCF itself."""

    name: str
    title: str
    mjcf: str
    joints: tuple[str, ...]
    gripper: str | None
    tcp_site: str
    hand_body: str
    base_body: str
    home: dict[str, float]
    tcp_approach: tuple[float, float, float] = (1.0, 0.0, 0.0)
    tcp_closing: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fingertips: tuple[str, str] = ("", "")
    roll_joint: str | None = None
    wave_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    onboard_cameras: tuple[str, ...] = ()
    jaw_bodies: tuple[str, str] = ("", "")
    limits: dict[str, tuple[float, float]]
    max_vel: dict[str, float]
    gripper_range_q: tuple[float, float] | None = None
    mirror_joints: dict[str, tuple[str, float]] = field(default_factory=dict)
    jaw: JawThresholds
    desk: Desk
    home_stages: tuple[tuple[str, ...], ...] = ()
    backend: str
    hardware: Hardware
    status: str = "ready"
    root: Path = field(default=Path("."), compare=False)

    @property
    def mjcf_path(self) -> Path:
        return self.root / self.mjcf

    @property
    def arm_joints(self) -> tuple[str, ...]:
        return tuple(j for j in self.joints if j != self.gripper)

    @property
    def is_draft(self) -> bool:
        return self.status == "draft"


def data_dir(name: str) -> Path:
    """A data folder of the package (`robot`, `end_effectors`, `mounts`, `embodiment`).

    A wheel ships it inside `soarm/`. A source checkout keeps it next to the `soarm/` package directory.
    """
    packaged = resources.files("soarm").joinpath(name)
    if packaged.is_dir():
        return Path(str(packaged))
    return Path(__file__).resolve().parent.parent / name


def robot_dir(name: str) -> Path:
    return data_dir("robot") / name


def so101() -> RobotSpec:
    return load(robot_dir("so101"))


def load(path: str | Path) -> RobotSpec:
    """Read and validate a robot.yaml, or the robot.yaml inside the folder `path`."""
    path = Path(path)
    if path.is_dir():
        path = path / SPEC_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        spec = _build(data, root=path.parent)
    except SpecError as exc:
        raise SpecError(f"{path}: {exc}") from None
    return spec


# --------------------------------------------------------------------------- value parsers

def _fail(where: str, expected: str, value: Any) -> SpecError:
    return SpecError(f"{where}: expected {expected}, got {value!r}")


def _str(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(where, "a non-empty string", value)
    return value


def _opt_str(where: str, value: Any) -> str | None:
    return None if value is None else _str(where, value)


def _float(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(where, "a number", value)
    return float(value)


def _opt_float(where: str, value: Any) -> float | None:
    return None if value is None else _float(where, value)


def _int(where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(where, "an integer", value)
    return value


def _opt_int(where: str, value: Any) -> int | None:
    return None if value is None else _int(where, value)


def _bool(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _fail(where, "true or false", value)
    return value


def _list(where: str, value: Any, size: int | None = None) -> list[Any]:
    if not isinstance(value, list) or (size is not None and len(value) != size):
        raise _fail(where, f"a list of {size}" if size else "a list", value)
    return value


def _mapping(where: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise _fail(where, "a mapping with string keys", value)
    return value


def _names(where: str, value: Any, size: int | None = None) -> tuple[str, ...]:
    return tuple(_str(f"{where}[{i}]", v) for i, v in enumerate(_list(where, value, size)))


def _floats(where: str, value: Any, size: int) -> tuple[float, ...]:
    return tuple(_float(f"{where}[{i}]", v) for i, v in enumerate(_list(where, value, size)))


def _range(where: str, value: Any) -> tuple[float, float]:
    lo, hi = _floats(where, value, 2)
    if lo >= hi:
        raise _fail(where, "[low, high] with low < high", value)
    return lo, hi


def _opt_range(where: str, value: Any) -> tuple[float, float] | None:
    return None if value is None else _range(where, value)


def _float_map(where: str, value: Any) -> dict[str, float]:
    return {k: _float(f"{where}.{k}", v) for k, v in _mapping(where, value).items()}


def _range_map(where: str, value: Any) -> dict[str, tuple[float, float]]:
    return {k: _range(f"{where}.{k}", v) for k, v in _mapping(where, value).items()}


def _int_map(where: str, value: Any) -> dict[str, int]:
    return {k: _int(f"{where}.{k}", v) for k, v in _mapping(where, value).items()}


def _mirror_map(where: str, value: Any) -> dict[str, tuple[str, float]]:
    out = {}
    for k, v in _mapping(where, value).items():
        source, ratio = _list(f"{where}.{k}", v, 2)
        out[k] = (_str(f"{where}.{k}[0]", source), _float(f"{where}.{k}[1]", ratio))
    return out


def _stages(where: str, value: Any) -> tuple[tuple[str, ...], ...]:
    return tuple(_names(f"{where}[{i}]", v) for i, v in enumerate(_list(where, value)))


def _status(where: str, value: Any) -> str:
    if value not in STATUSES:
        raise _fail(where, " or ".join(STATUSES), value)
    return value


def _backend(where: str, value: Any) -> str:
    module, sep, attr = _str(where, value).partition(":")
    if not (module and sep and attr):
        raise _fail(where, "'package.module:Class'", value)
    return value


def _section(cls: type, parsers: Mapping[str, Callable[[str, Any], Any]]) -> Callable[[str, Any], Any]:
    def parse(where: str, value: Any) -> Any:
        return cls(**_parse_keys(where, _mapping(where, value), parsers, required=set(parsers)))

    return parse


def _parse_keys(where: str, data: dict[str, Any], parsers: Mapping[str, Callable[[str, Any], Any]],
                required: set[str]) -> dict[str, Any]:
    unknown = sorted(set(data) - set(parsers))
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {', '.join(unknown)}")
    missing = sorted(required - set(data))
    if missing:
        raise SpecError(f"{where}: missing key(s) {', '.join(missing)}")
    return {k: parsers[k](f"{where}.{k}" if where else k, v) for k, v in data.items()}


_JAW = _section(JawThresholds, {f.name: _float for f in fields(JawThresholds)})

_DESK = _section(Desk, {
    "goal_radius": _range,
    "goal_height": _range,
    "cube_radius": _range,
    "cube_bearing": _range,
    "wave_radius": _range,
    "search_centre": lambda w, v: _floats(w, v, 3),
    "search_dist": _range,
    "camera_dist": _range,
    "camera_centre": lambda w, v: _floats(w, v, 3),
    "map_centre": lambda w, v: _floats(w, v, 2),
    "map_side": _float,
})

_HARDWARE = _section(Hardware, {
    "kind": _str,
    "servo_ids": _int_map,
    "baudrate": _int,
    "center_ticks": _int,
    "gripper_closed_ticks": _opt_int,
    "gripper_open_ticks": _opt_int,
    "use_degrees": _bool,
    "max_relative_target": _opt_float,
    "read_state_hz": _float,
    "torque_off_on_exit": _bool,
})

_SPEC: dict[str, Callable[[str, Any], Any]] = {
    "status": _status,
    "name": _str,
    "title": _str,
    "mjcf": _str,
    "joints": _names,
    "gripper": _opt_str,
    "tcp_site": _str,
    "hand_body": _str,
    "base_body": _str,
    "home": _float_map,
    "tcp_approach": lambda w, v: _floats(w, v, 3),
    "tcp_closing": lambda w, v: _floats(w, v, 3),
    "fingertips": lambda w, v: _names(w, v, 2),
    "roll_joint": _opt_str,
    "wave_ranges": _range_map,
    "onboard_cameras": _names,
    "jaw_bodies": lambda w, v: _names(w, v, 2),
    "limits": _range_map,
    "max_vel": _float_map,
    "gripper_range_q": _opt_range,
    "mirror_joints": _mirror_map,
    "jaw": _JAW,
    "desk": _DESK,
    "home_stages": _stages,
    "backend": _backend,
    "hardware": _HARDWARE,
}

_REQUIRED = {
    "name", "title", "mjcf", "joints", "gripper", "tcp_site", "hand_body", "base_body",
    "home", "limits", "max_vel", "jaw", "desk", "backend", "hardware",
}


def _build(data: Any, root: Path) -> RobotSpec:
    if not isinstance(data, dict):
        raise _fail("robot.yaml", "a mapping at the top level", data)
    spec = RobotSpec(**_parse_keys("", data, _SPEC, _REQUIRED), root=root)
    _check_joint_references(spec)
    return spec


def _check_joint_references(spec: RobotSpec) -> None:
    joints = set(spec.joints)
    if not spec.joints or len(joints) != len(spec.joints):
        raise SpecError(f"joints: expected unique names, got {list(spec.joints)}")

    def within(where: str, names: Any) -> None:
        stray = sorted(set(names) - joints)
        if stray:
            raise SpecError(f"{where}: {', '.join(stray)} not in joints")

    def complete(where: str, names: Any) -> None:
        missing = [j for j in spec.joints if j not in set(names)]
        if missing:
            raise SpecError(f"{where}: missing joint(s) {', '.join(missing)}")

    within("gripper", [spec.gripper] if spec.gripper else [])
    within("roll_joint", [spec.roll_joint] if spec.roll_joint else [])
    within("home", spec.home)
    within("limits", spec.limits)
    within("max_vel", spec.max_vel)
    within("wave_ranges", spec.wave_ranges)
    within("mirror_joints", [source for source, _ in spec.mirror_joints.values()])
    within("home_stages", [j for stage in spec.home_stages for j in stage])
    within("hardware.servo_ids", spec.hardware.servo_ids)
    complete("limits", spec.limits)
    complete("max_vel", spec.max_vel)
    complete("hardware.servo_ids", spec.hardware.servo_ids)

    ids = list(spec.hardware.servo_ids.values())
    if len(set(ids)) != len(ids):
        raise SpecError(f"hardware.servo_ids: expected unique IDs, got {ids}")
