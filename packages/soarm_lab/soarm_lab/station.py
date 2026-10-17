"""A measured station, handed over by the twin as data: `scene.xml`, `assets/` and `station.json`.

The twin writes such a folder with `twin export <folder>`:

* `scene.xml`: the station's MuJoCo scene (floor, table, lights, the arm under the prefix `robot/`,
  every enabled camera with a pose and its full K, the arm's physics options), `meshdir="assets/"`;
* `assets/`: the arm's meshes;
* `station.json`: the station record (robot, table, cameras, identified dynamics, home).

    station = Station.load("sim")            # the station bundled with soarm-lab
    station = Station.load("~/stations/desk1")
    model = station.build(objects=[Box("cube", (0.015,) * 3, (0.2, 0.0))], grasp_sensors=["cube"])

The lab reads the folder with plain `mujoco` and never imports the twin. `build` loads `scene.xml`
through `MjSpec`, adds the task's objects and a contact sensor between each grasped object and each jaw,
and compiles. A folder without `assets/` takes the meshes from the robot folder of `soarm`; that is how
the bundled `sim` station is shipped.

Poses are in the robot's BASE frame, cameras in the OpenCV convention (z forward, y down).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .kinematics import pose
from .robot import RobotSpec, load_robot

#: Stations bundled with the package, by name (`Station.load("sim")`).
STATIONS_DIR = Path(__file__).parent / "stations"
SCENE_FILE = "scene.xml"
STATION_FILE = "station.json"
ASSETS_DIR = "assets"
#: The arm's names in the scene: joints, actuators, sites and bodies are `robot/<name>`.
PREFIX = "robot/"
#: A MuJoCo camera looks along -z with y up; OpenCV along +z with y down.
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
#: Horizontal field of view of a camera without intrinsics in `station.json`. Its nominal K has the
#: focal length of this field of view and the principal point at the image centre.
NOMINAL_HFOV_DEG = 65.0


@dataclass
class CameraView:
    """A camera of the scene: intrinsics and its pose relative to the arm's base (OpenCV)."""

    name: str
    K: np.ndarray
    width: int
    height: int
    T_cam2base: np.ndarray

    @staticmethod
    def from_fov(name: str, width: int, height: int, fovy_deg: float, T_cam2base: np.ndarray) -> CameraView:
        f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        K = np.array([[f, 0.0, width / 2 - 0.5], [0.0, f, height / 2 - 0.5], [0.0, 0.0, 1.0]])
        return CameraView(name, K, width, height, np.asarray(T_cam2base, float))


@dataclass
class Table:
    """Table top size [m], its height above the floor, and where the arm's base stands on it."""

    size: tuple[float, float] = (0.9, 0.6)
    height: float = 0.75
    base_xy: tuple[float, float] = (-0.35, 0.0)
    base_yaw: float = 0.0
    rgba: tuple[float, float, float, float] = (0.55, 0.50, 0.44, 1.0)

    @property
    def T_base2world(self) -> np.ndarray:
        """The arm's base in the scene's world frame: on the table top at `base_xy`, turned by `base_yaw`."""
        c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return pose(R, np.array([self.base_xy[0], self.base_xy[1], self.height]))


@dataclass
class Box:
    """A box on the table: a task object (free body) or a marker (mocap body without collisions)."""

    name: str
    #: Half sides [m].
    size: tuple[float, float, float]
    #: Where it lies on the table, in the base frame [m].
    xy: tuple[float, float]
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)
    mass: float = 0.05
    free: bool = True
    #: A mocap body without collisions, for looking only (e.g. the pickplace target pad). Its pose is
    #: set through `data.mocap_pos` / `mocap_quat`; objects pass through it.
    mocap: bool = False


class Station:
    """One exported station: its folder, its `station.json` and the robot description it names."""

    def __init__(self, folder: str | Path, data: dict[str, Any]):
        self.folder = Path(folder)
        self.data = data
        #: The robot's name as the twin registers it (`so101`).
        self.robot: str = str(data.get("robot", "so101"))
        self._spec: RobotSpec | None = None

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, where: str | Path = "sim") -> Station:
        """A station folder (`twin export <folder>`), or the name of a bundled station."""
        folder = resolve(where)
        data = json.loads((folder / STATION_FILE).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{folder / STATION_FILE}: expected a JSON object")
        return cls(folder, data)

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def scene_path(self) -> Path:
        return self.folder / SCENE_FILE

    def spec(self) -> RobotSpec:
        """The robot's description, from `soarm`'s robot folder; the station must name that robot."""
        if self._spec is None:
            spec = load_robot()
            if spec.name.lower() != self.robot.lower():
                raise ValueError(f"{self.folder}: the station's robot is {self.robot!r}; soarm-lab knows "
                                 f"only {spec.name!r}")
            self._spec = spec
        return self._spec

    # --------------------------------------------------------------- the record

    @property
    def table(self) -> Table:
        data = dict(self.data.get("table") or {})
        for key in ("size", "base_xy", "rgba"):
            if key in data:
                data[key] = tuple(data[key])
        return Table(**data)

    @property
    def T_base2world(self) -> np.ndarray:
        """The arm's base in the scene's world frame (4x4)."""
        return self.table.T_base2world

    @property
    def cameras(self) -> list[CameraView]:
        """Every enabled camera with a pose: the cameras `scene.xml` holds.

        A simulated camera renders from its true pose (`sim_pose`), a real one from its calibrated
        pose (`T_cam2base`). A camera without K gets the nominal one (`NOMINAL_HFOV_DEG`).
        """
        views = []
        for cam in self.data.get("cameras") or []:
            if not cam.get("enabled", True):
                continue
            simulated = cam.get("source") == "sim"
            T = cam.get("T_cam2base")
            if simulated and cam.get("sim_pose") is not None:
                T = cam["sim_pose"]
            if T is None:
                continue
            width, height = int(cam.get("width", 640)), int(cam.get("height", 480))
            K = cam.get("K")
            K = np.asarray(K, float) if K is not None else _nominal_K(width, height)
            views.append(CameraView(str(cam["name"]), K, width, height, np.asarray(T, float)))
        return views

    @property
    def dynamics(self) -> dict[str, Any]:
        """Servo dynamics the twin identified on this arm; {} = the Menagerie model."""
        return dict(self.data.get("dynamics") or {})

    @property
    def home(self) -> dict[str, float] | None:
        """This station's home pose [app units]; None = the robot description's."""
        home = self.data.get("home")
        return None if home is None else {str(k): float(v) for k, v in home.items()}

    # ---------------------------------------------------------------- the model

    def compose(self, objects: Sequence[Box] = (), grasp_sensors: Sequence[str] = (),
                cameras: Sequence[CameraView] = ()) -> mujoco.MjSpec:
        """`scene.xml` as an `MjSpec`, plus task objects, jaw contact sensors and extra cameras.

        Each of `objects` is a free body with a box geom (condim 4, friction 1 / 0.02 / 0.002), or a
        mocap body without collisions. `grasp_sensors` names objects that get a contact sensor with
        each jaw, `<object>_jaw0` (fixed) and `<object>_jaw1` (moving), 1 when they touch. `cameras`
        are added with their full K; a name the scene already has fails.
        """
        spec = mujoco.MjSpec.from_file(str(self.scene_path))
        if not (self.folder / ASSETS_DIR).is_dir():
            spec.meshdir = str(_robot_meshdir(self.spec()))
        T_base2world = self.T_base2world

        world = spec.worldbody
        for obj in objects:
            p = T_base2world @ np.array([obj.xy[0], obj.xy[1], obj.size[2], 1.0])
            body = world.add_body()
            body.name = obj.name
            body.pos = p[:3]
            if obj.mocap:
                body.mocap = True
            elif obj.free:
                body.add_freejoint()
            g = body.add_geom()
            g.name = obj.name
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(obj.size)
            g.rgba = list(obj.rgba)
            g.mass = obj.mass
            if obj.mocap:
                g.contype = g.conaffinity = 0
                g.group = 1
                continue
            g.condim = 4
            g.friction = [1.0, 0.02, 0.002]

        for obj in grasp_sensors:
            for k, jaw in enumerate(self.spec().jaw_bodies):
                s = spec.add_sensor()
                s.name = f"{obj}_jaw{k}"
                s.type = mujoco.mjtSensor.mjSENS_CONTACT
                s.objtype, s.objname = mujoco.mjtObj.mjOBJ_BODY, PREFIX + jaw
                s.reftype, s.refname = mujoco.mjtObj.mjOBJ_BODY, obj
                s.intprm = [1, 0, 1]                 # data: "found"; no reduction; one contact

        present = {c.name for c in spec.cameras}
        for view in cameras:
            if view.name in present:
                raise ValueError(f"{self.scene_path}: the scene already has a camera {view.name!r}")
            _add_camera(spec, view, T_base2world)

        g = spec.visual.global_
        g.offwidth = max([g.offwidth] + [v.width for v in cameras])
        g.offheight = max([g.offheight] + [v.height for v in cameras])
        return spec

    def build(self, objects: Sequence[Box] = (), grasp_sensors: Sequence[str] = (),
              cameras: Sequence[CameraView] = ()) -> mujoco.MjModel:
        """The compiled scene with the task's additions (see `compose`)."""
        return self.compose(objects, grasp_sensors, cameras).compile()


def stations() -> list[str]:
    """Names of the stations bundled with soarm-lab."""
    return sorted(p.name for p in STATIONS_DIR.iterdir() if (p / STATION_FILE).is_file())


def resolve(where: str | Path) -> Path:
    """The station folder: an existing folder with `station.json`, else a bundled station's name."""
    folder = Path(where).expanduser()
    if (folder / STATION_FILE).is_file():
        if not (folder / SCENE_FILE).is_file():
            raise FileNotFoundError(f"{folder}: no {SCENE_FILE}; export the station again "
                                    "(`twin export`)")
        return folder
    bundled = STATIONS_DIR / str(where)
    if folder.name == str(where) and (bundled / STATION_FILE).is_file():
        return bundled
    raise FileNotFoundError(f"No station {str(where)!r}: neither a folder with {STATION_FILE} nor a "
                            f"bundled station ({', '.join(stations())}). "
                            "Export one with `twin export <folder>`.")


def _nominal_K(width: int, height: int, hfov_deg: float = NOMINAL_HFOV_DEG) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _robot_meshdir(spec: RobotSpec) -> Path:
    """Where the robot's own MJCF finds its meshes."""
    robot = mujoco.MjSpec.from_file(str(spec.mjcf_path))
    return (Path(robot.modelfiledir) / robot.meshdir).resolve()


def _add_camera(spec: mujoco.MjSpec, view: CameraView, T_base2world: np.ndarray) -> None:
    """A camera with its full K at `view.T_cam2base`, in the scene's world frame."""
    T_world = T_base2world @ np.asarray(view.T_cam2base, float)
    cam = spec.worldbody.add_camera()
    cam.name = view.name
    cam.pos = T_world[:3, 3]
    cam.quat = _quat(T_world[:3, :3] @ CV_TO_MJ)
    K = np.asarray(view.K, float)
    cam.resolution = [view.width, view.height]
    # The sensor size only switches MuJoCo from `fovy` to the focal model; with the focal length in
    # pixels it cancels out.
    cam.sensor_size = [view.width * 1e-6, view.height * 1e-6]
    cam.focal_pixel = [K[0, 0], K[1, 1]]
    # The principal point as the offset of the image centre from it: pixel (0, 0) is centred at 0
    # in OpenCV, so the image centre is (W - 1) / 2, (H - 1) / 2. The sign was measured on renders.
    cam.principal_pixel = [(view.width - 1) / 2 - K[0, 2], (view.height - 1) / 2 - K[1, 2]]
