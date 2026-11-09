"""A simulated stand for the approach calibration's twin tests: a fixed camera, a ChArUco board and the arm.

`Stand(seed, ...)` draws, from its seed, where the board lies, where the fixed camera `kamera1` stands, how
wrong its calibrated pose is, how far the wrist camera sits from where robot.yaml puts it and how much play
the arm's joints have:

- `kamera1` is C920-like: 1280x720, fx 920, about 0.6 m in front of the arm and to its right, looking at
  (0.2, 0, 0). The twin renders it from its true pose (`sim_pose`); its perception places detections from
  the believed one (`T_cam2base`). `error` gives the same error field for anything else that camera sees,
  such as the fake block cameras of the pick_place tests;
- the board is a textured panel on the table, without collision, its long side across the arm, centred
  near (0.2, 0) and turned by up to 10 degrees, as `calib.intrinsics.simulate` lays out its board;
- `shift_wrist_camera` moves the rendered wrist camera by `mount_m` in a seeded direction, so the strategy's
  camera model is off by that much;
- `play()` is `play_for(play_m)`, for the twin's `stand_errors.Backlash`;
- `block_error(half_height)` is the same error for the blocks of the pick_place and build tests, whose fake
  cameras see each block's top.

`perception()` runs the real `charuco` detector through the twin's `Perception`, stepped by `drive`.
`calibrate()` runs `calibrate_approach` on a stand's twin, the strategy wrapped in `Backlash` when the stand
has play, and keeps the simulation's truth for every approach to a corner. `Stand.table()` gives the stand's
approach table, calibrated once in a twin of its own and kept as a file. Only tests marked `twin` import this
module.
"""

from __future__ import annotations

import functools
import math
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from machinekind_twin import scene as twin_scene
from machinekind_twin.clock import VirtualClock
from machinekind_twin.detection import DetectorRunner, Perception
from machinekind_twin.detectors import load_detector
from machinekind_twin.kinematics import inverse
from machinekind_twin.runtime import Twin
from machinekind_twin.stand_errors import Backlash, CameraPoseError, tilted, uniform
from machinekind_twin.stand_errors import play_for as twin_play_for
from machinekind_twin.strategy import StrategyRunner
from machinekind_twin.workspace import CameraRecord, Workspace
from machinekind_twin_api.boards import BoardSpec

from soarm_lab.approach_table import ApproachTable
from soarm_lab.kinematics import RobotKinematics, pose
from soarm_lab.robot import load_robot
from soarm_lab.strategies.calibrate_approach import (
    CORNERS,
    CalibrateApproachSettings,
    CalibrateApproachStrategy,
    Planned,
)
from soarm_lab.strategies.grasp import top_down

CAMERA = "kamera1"
SIZE = (1280, 720)
FOCAL_PX = 920.0
#: Where the fixed camera stands and looks [m, base frame], and the seeded jitter of both.
EYE = (0.62, -0.30, 0.38)
LOOK_AT = (0.2, 0.0, 0.0)
EYE_JITTER_M = 0.03
LOOK_JITTER_M = 0.02
#: The board: its top above the table [m], its texture's resolution [px/m], where its centre lies and how
#: far it may be moved and turned [m, deg].
THICKNESS_M = 0.0005
PX_PER_M = 4000.0
BOARD_CENTRE = (0.20, 0.0)
BOARD_SHIFT_M = 0.01
BOARD_TURN_DEG = 10.0
#: The arm's start pose: turned aside, away from the fixed camera, as the stand procedure asks.
PARK_PAN_DEG = -80.0
#: Simulated time step, twin loop period [s] and how often the detector runs [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
DETECT_EVERY_S = 0.5
#: Tables the campaign keeps, under `SOARM_CAMPAIGN_DIR`.
TABLES = "tables"
#: A calibration's simulated time limit [s].
CALIBRATION_S = 3600.0
#: The arm joints that place the hand, and so carry play.
PLAY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


@dataclass
class Clock:
    """Simulated time shared by the runner, the detector and the trackers [s]."""

    t: float = 0.0


@functools.lru_cache(maxsize=1)
def _kinematics() -> RobotKinematics:
    return RobotKinematics(load_robot())


def play_for(metres: float, xy: tuple[float, float] = (0.2, 0.0)) -> dict[str, float]:
    """Play per joint [deg] that moves the hand `metres` across its whole width, at a grasp over `xy`."""
    kin = _kinematics()
    home = dict(kin.spec.home, gripper=0.0)
    sol = kin.ik(np.array([*xy, 0.02]), top_down(0.0), seed=home)
    assert sol.ok
    return twin_play_for(metres, kin.tcp, sol.joints, PLAY_JOINTS)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """An OpenCV camera pose (z forward, y down) at `eye` looking at `target`."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Stand:
    """One seeded stand: the board, the fixed camera and its error, the wrist camera's offset and the play."""

    def __init__(self, seed: int, *, camera: str = "none", camera_m: float = 0.0, play_m: float = 0.0,
                 mount_m: float = 0.0, board: BoardSpec | None = None):
        self.seed = seed
        rng = np.random.default_rng(1000 + seed)
        self.board = board or BoardSpec(thickness_m=THICKNESS_M)
        self.board_yaw = math.radians(float(rng.uniform(-BOARD_TURN_DEG, BOARD_TURN_DEG)))
        centre = np.array(BOARD_CENTRE) + rng.uniform(-BOARD_SHIFT_M, BOARD_SHIFT_M, 2)
        self.T_panel = pose(rot_z(self.board_yaw), np.array([*centre, self.board.thickness_m]))
        bw, bh = self.board.size_m
        self.T_board = self.T_panel @ pose(np.diag([1.0, -1.0, -1.0]), np.array([-bw / 2, bh / 2, 0.0]))
        eye = np.array(EYE) + rng.uniform(-EYE_JITTER_M, EYE_JITTER_M, 3)
        target = np.array(LOOK_AT) + rng.uniform(-LOOK_JITTER_M, LOOK_JITTER_M, 3)
        self.K = np.array([[FOCAL_PX, 0.0, (SIZE[0] - 1) / 2], [0.0, FOCAL_PX, (SIZE[1] - 1) / 2],
                           [0.0, 0.0, 1.0]])
        T_true = look_at(eye, target)
        self.camera_kind, self.camera_m = camera, float(camera_m)
        if camera == "uniform" and camera_m:
            a = float(rng.uniform(0.0, 2 * math.pi))
            self.error = uniform(self.K, SIZE, T_true, camera_m * np.array([math.cos(a), math.sin(a)]))
        elif camera == "tilted" and camera_m:
            self.error = tilted(self.K, SIZE, T_true, camera_m, rng, LOOK_AT)
        elif camera in ("none", "uniform", "tilted"):
            self.error = CameraPoseError(self.K, SIZE, T_true, T_true.copy())
        else:
            raise ValueError(f"unknown camera error {camera!r}")
        self.play_m, self.mount_m = float(play_m), float(mount_m)
        d = rng.normal(size=3)
        self.mount_offset = (mount_m * d / np.linalg.norm(d)) if mount_m else np.zeros(3)

    # ------------------------------------------------------------------ truth
    def corners(self) -> np.ndarray:
        """The board's inner corners where they really are (n, 3) [m, base frame]."""
        obj = self.board.corner_obj(range(self.board.n_corners))
        return obj @ self.T_board[:3, :3].T + self.T_board[:3, 3]

    def board_point(self, board_xy: np.ndarray) -> np.ndarray:
        """Board-frame points (n, 2) where they really are (n, 3) [m, base frame]."""
        xy = np.atleast_2d(np.asarray(board_xy, float))
        obj = np.column_stack([xy, np.zeros(len(xy))])
        return obj @ self.T_board[:3, :3].T + self.T_board[:3, 3]

    def seen(self, P: np.ndarray) -> np.ndarray:
        """Where the stand's perception puts points `P` (n, 3)."""
        return self.error.field(P)

    def block_error(self, half_height: float) -> Callable[[np.ndarray], np.ndarray]:
        """Where the stand's perception puts the centres (n, 3) of blocks `half_height` high: the cameras see
        each block's top, so its error is the field at the top's height."""
        up = np.array([0.0, 0.0, float(half_height)])

        def perceived(P: np.ndarray) -> np.ndarray:
            P = np.atleast_2d(np.asarray(P, float))
            return self.seen(P + up) - up

        return perceived

    def play(self) -> dict[str, float]:
        return play_for(self.play_m) if self.play_m else {}

    # ------------------------------------------------------------------ the twin
    def workspace(self) -> Workspace:
        ws = Workspace(robot="so101")
        ws.cameras.append(CameraRecord(CAMERA, source="sim", width=SIZE[0], height=SIZE[1], K=self.K.tolist(),
                                       T_cam2base=self.error.T_believed.tolist(),
                                       sim_pose=self.error.T_true.tolist(), intrinsics_from="symulacja"))
        return ws

    def board_panel(self) -> twin_scene.Panel:
        import cv2

        return twin_scene.Panel("charuco", self.board.image(cv2, PX_PER_M), self.board.size_m, self.T_panel)

    def twin(self, **extras: Any) -> Twin:
        """A simulated twin with the board on the table and the arm parked aside, connected for `step`."""
        twin = Twin(self.workspace(), clock=VirtualClock(t0=0.0))
        twin.configure(panels=[self.board_panel()], **extras)
        twin.connect("sim", threaded=False)
        self.park(twin)
        if self.mount_m:
            self.shift_wrist_camera(twin)
        return twin

    def park(self, twin: Twin, seconds: float = 3.0) -> None:
        """Drive the arm to its start pose through the twin's supervisor and let it settle."""
        spec = twin.workspace.spec()
        twin.set_engaged(True)
        twin.set_target(dict(spec.home, shoulder_pan=PARK_PAN_DEG))
        for _ in range(round(seconds / TWIN_PERIOD)):
            twin.step(TWIN_PERIOD)
        twin.set_engaged(False)

    def shift_wrist_camera(self, twin: Twin, camera: str = "wrist") -> None:
        """Move the rendered wrist camera by `mount_offset` [m, in its body's frame]."""
        with twin.lock:
            model = twin.scene.model
            model.cam_pos[model.camera(twin_scene.PREFIX + camera).id] += self.mount_offset
            mujoco.mj_forward(model, twin.scene.data)

    def perception(self, twin: Twin, clock: Clock, **settings: Any) -> tuple[Perception, DetectorRunner]:
        """The twin's perception with the `charuco` detector, which `drive` steps."""
        perception = Perception(twin, twin.workspace, frame=lambda name: (twin.render(name), clock.t),
                                hand=lambda: None, clock=lambda: clock.t)
        raw = {"thickness_m": self.board.thickness_m, "square_m": self.board.square_m,
               "marker_m": self.board.marker_m, **settings}
        runner = perception.start_detector(load_detector("charuco"), raw, threaded=False)
        return perception, runner

    def table_path(self, fallback: Path | None = None, name: str = "") -> Path:
        """Where this stand's table is kept: under the campaign folder, else `fallback`, else a temporary
        folder that lasts as long as the process."""
        root = os.environ.get("SOARM_CAMPAIGN_DIR", "").strip()
        base = Path(root) / TABLES if root else (fallback or _scratch())
        return base / f"stand-{self.seed:04d}{name}.json"

    @property
    def name(self) -> str:
        """The stand's errors in a file name: the same seed with other errors is another stand."""
        return (f"-{self.camera_kind}-{self.camera_m * 1000:.0f}mm-play-{self.play_m * 1000:.0f}mm"
                f"-mount-{self.mount_m * 1000:.1f}mm")

    def table(self, fallback: Path | None = None, **settings: Any) -> ApproachTable:
        """This stand's approach table: the kept file when there is one, else a calibration in a twin of its
        own, kept under `table_path`. Raises when the calibration does not finish."""
        path = self.table_path(fallback, self.name)
        if path.exists():
            table = ApproachTable.load(path)
            if table.data.get("complete"):
                return table
        twin = self.twin()
        try:
            cal = calibrate(self, twin, Clock(), path, **settings)
        finally:
            twin.close()
        if cal.table is None or not cal.strategy.status.success:
            raise RuntimeError(f"stand {self.seed}: the calibration did not finish: {cal.reason}")
        return ApproachTable.load(path)


@functools.lru_cache(maxsize=1)
def _scratch() -> Path:
    return Path(tempfile.mkdtemp(prefix="approach-tables-"))


def truth_tcp(twin: Twin) -> np.ndarray:
    """The TCP where the simulated arm really is [m, base frame]."""
    with twin.lock:
        s = twin.scene
        site = s.model.site(s.kin.prefix + s.kin.spec.tcp_site).id
        Ti = inverse(s.T_base2world)
        return Ti[:3, :3] @ s.data.site_xpos[site] + Ti[:3, 3]


class TableTouch:
    """The deepest contact of the arm with the table, at every physics step [m]."""

    def __init__(self, twin: Twin):
        scene = twin.scene
        model, data = scene.model, scene.data
        self.table = model.geom("table").id
        self.arm = {g for g in range(model.ngeom) if scene.kin.prefix
                    and model.body(int(model.geom_bodyid[g])).name.startswith(scene.kin.prefix)}
        self.depth = 0.0
        self.touches = 0
        step = scene.step

        def substeps(seconds: float) -> None:
            for _ in range(max(1, round(seconds / model.opt.timestep))):
                step(model.opt.timestep)
                self.look(data)

        scene.step = substeps

    def look(self, data: Any) -> None:
        for k in range(data.ncon):
            c = data.contact[k]
            pair = {int(c.geom1), int(c.geom2)}
            if self.table in pair and pair & self.arm:
                self.touches += 1
                self.depth = max(self.depth, -float(c.dist))

    def artefacts(self) -> list[str]:
        return [f"the arm touched the table ({self.touches} contacts, {self.depth * 1000:.2f} mm deep)"] \
            if self.touches else []


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          detector: DetectorRunner | None = None, detect_while: Callable[[], bool] | None = None,
          watch: Callable[[], None] | None = None) -> None:
    """Run twin, detector and strategy ticks on a shared simulated clock until the run ends.

    The detector runs every `DETECT_EVERY_S` while `detect_while()` holds (always without it)."""
    strategy_every = max(1, round(1.0 / runner.strategy.hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    detect_every = round(DETECT_EVERY_S / CLOCK_STEP)
    for _ in range(round(seconds / CLOCK_STEP)):
        k = round(clock.t / CLOCK_STEP) + 1
        clock.t = k * CLOCK_STEP
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
            if watch is not None:
                watch()
        if detector is not None and k % detect_every == 0 and (detect_while is None or detect_while()):
            detector.step_once()
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(clock.t)
        if not runner.status.running:
            break


# --------------------------------------------------------------------------- a calibration on the stand
@dataclass
class Calibration:
    """A calibration run and what the simulation's truth says about its samples."""

    strategy: CalibrateApproachStrategy
    reason: str
    sim_s: float
    wall_s: float
    artefacts: list[str]
    #: Per approach to a corner (kind, corner, level, direction, repeat): the true correction [m, x and y].
    truth: dict[tuple[str, int, int, str, int], np.ndarray] = field(default_factory=dict)

    @property
    def table(self) -> ApproachTable | None:
        return self.strategy.table

    def sample_errors(self) -> list[float]:
        """How far each sample's correction lies from the truth [m]."""
        out = []
        for s in self.strategy.samples:
            key = ("sample", int(s["corner"]), int(s["level"]), str(s["direction"]), int(s["repeat"]))
            if key in self.truth:
                out.append(float(np.linalg.norm(np.asarray(s["delta_m"], float)[:2] - self.truth[key])))
        return out


def calibrate(stand: Stand, twin: Twin, clock: Clock, table: Path, **settings: Any) -> Calibration:
    """Run `calibrate_approach` on the stand's twin to its end, its table going to `table`.

    The true correction of each approach to a corner is the true corner minus where the TCP really is, plus
    what the TCP lacked of its command."""
    perception, detector = stand.perception(twin, clock)
    truth: dict[tuple[str, int, int, str, int], np.ndarray] = {}
    corners = stand.corners()

    def on_still(strategy: CalibrateApproachStrategy, obs: Any, planned: Planned) -> None:
        if planned.corner < 0:
            return
        missing = corners[planned.corner][:2] - truth_tcp(twin)[:2] + (strategy.tcp_cmd[:2] - planned.xy)
        truth[(planned.kind, planned.corner, planned.level, planned.direction, planned.repeat)] = missing

    cfg = CalibrateApproachSettings(**{"table_file": str(table), "merge": False,
                                       "thickness_m": stand.board.thickness_m, **settings})
    strategy = CalibrateApproachStrategy(cfg, on_still=on_still)
    driven = Backlash(strategy, stand.play()) if stand.play_m else strategy
    runner = StrategyRunner(twin, driven, objects=perception.objects, cameras=("wrist",))
    touch = TableTouch(twin)
    started, t0 = time.monotonic(), clock.t
    runner.start(threaded=False)
    drive(twin, runner, CALIBRATION_S, clock, detector=detector,
          detect_while=lambda: not strategy.started or strategy.status.phase == CORNERS)
    if runner.status.running:
        runner.stop()
    perception.stop_all("koniec")
    reason = strategy.status.stopped_because or runner.status.stopped_because
    return Calibration(strategy, reason, clock.t - t0, time.monotonic() - started, touch.artefacts(), truth)
