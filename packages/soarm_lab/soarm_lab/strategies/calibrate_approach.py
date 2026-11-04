"""Calibrate how the arm lands on what the stand's cameras see: `calibrate_approach`, a twin strategy.

The arm approaches the corners of a ChArUco board lying flat on the table from several sides and at the
heights where a held block is released on a tower's first, second and third level. At each stop the wrist
camera, rigid to the jaws, measures where the corner lies against the TCP (`charuco`): no joint reading
enters a sample. The corrections go to an approach table (`approach_table`) that `pick_place` reads.

    # configs/local.yaml of the twin
    calibrate_approach:
      square_m: 0.045                         # as measured on the print, and the same in `charuco:`
      marker_m: 0.035
      table_file: ~/stands/desk/approach.json
      wrist_camera: wrist
      rotate_180: false                       # as in `grasp_wrist`

The board's corners come from the `charuco` detector through the twin's perception, the same camera
models and geometry that place the blocks. So a corner carries the camera error a block would have there.

Phases (`CalibrationStatus.phase`):

- `corners`: average at least `corner_samples` sightings of each corner and fit the board to them. A
  board whose corners do not match its set size stops the run before the arm moves;
- `intrinsics`: look at the board from above, the hand leaning away from the base by up to 30 degrees,
  and calibrate the wrist camera's K from those views (or take it from the table file with
  `reuse_intrinsics`);
- `mount`: per level, approach a corner, look, turn the wrist roll alone half a turn and look again. The
  arm's other joints and their play stay put and the TCP's move round the roll axis is in its command, so
  what the two looks disagree on beyond that move is a constant error of the TCP in the camera's axes,
  which turns with the hand (`charuco.flip_bias`). It is removed from every sample of that level;
- `samples`: per corner, level and repeat, one approach from each direction (`strategies.approach`):
  straight up, a joint move to the lead point at the carry height, a horizontal leg along the direction,
  down fast to `SLOW_ABOVE_M` over the release height and slowly the rest, then still, then `looks`
  frames;
- `verify`: per corner and level, one approach to the corner plus the table's correction from the
  preferred direction, and what is left of the error;
- `save`, then `park`: back to the start pose.

The heading of each approach turns the camera toward the board: of a quarter turn either way from across
the arm, along it and halfway, the one whose predicted view holds the most of the board clear of the jaws,
with the board a little off where the plan expects it. When the camera still sees too little, the wrist
roll alone turns and it looks again, up to `MAX_RELOOKS` times: the hand's error stays the same.

An approach that does not arrive within `move_timeout_s`, whose frames show no usable board within
`look_timeout_s`, or whose next move would bring the arm onto itself, is skipped; the arm backs out the way
it came. A joint move that would touch the arm's own body goes through a hub pose over the board, or is
skipped. More than `max_skip_share` of the samples skipped stops the run. A stopped or failed run writes
`<table_file>.partial.json`.

Horizontal travel happens only at the carry height, which clears the catalog's tallest block. The operator
clears the table of blocks first.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo

from .. import charuco
from ..approach_table import DIRECTIONS, ApproachTable, Hit, now_text, write_atomic
from ..blocks import block
from ..kinematics import inverse
from ..robot import RobotCamera
from .approach import bearing, carry_heights, direction_vector, lead_point, place_heights
from .grasp import MAX_ROT_ERR_DEG, MIN_GRASP_Z, hand_down, heading, ik_restarts, tilt, wrap
from .pick_place import GRIP_SLIP_M, MIN_CARRY_CLEAR_M, SLOW_ABOVE_M, TABLE_TOP_M
from .touch import DONE, TouchSettings, TouchStrategy
from .wrist import WORK_WIDTH_PX, jaw_mask, pinhole, project

logger = logging.getLogger(__name__)

#: Phases of a run.
CORNERS = "corners"
INTRINSICS = "intrinsics"
MOUNT = "mount"
SAMPLES = "samples"
VERIFY = "verify"
SAVE = "save"
PARK = "park"
#: Steps of one approach, after its moves.
STILL = "still"
LOOK = "look"
#: Kinds of approach.
SAMPLE = "sample"
MOUNT_A = "mount_a"
MOUNT_B = "mount_b"
CHECK = "verify"
PROBE = "probe"
VIEW = "view"

#: The block whose width sets the jaw opening and whose grasp sets how far a held block hangs.
BLOCK = "cube"
#: The fraction of the block's height the grasp holds it at, and each open tip's margin (grasp's defaults).
GRASP_DEPTH = 0.5
OPEN_MARGIN_M = 0.008
#: Pose sources a corner may come from.
CORNER_SOURCES = ("kamery", "ostatnie widziane")
#: The fewest corners a run needs, and captures closer than this count as one sighting [s].
MIN_CORNERS = 4
SIGHTING_S = 0.02
#: The board fit may miss its corners by this much [m, RMS] and its scale may be off by this share.
FIT_MAX_RMS_M = 0.004
FIT_MAX_SCALE = 0.03
#: The jaw's lowest point stays this far above the board at every level [m].
JAW_BOARD_CLEAR_M = 0.005
#: Looks of one approach may spread this much [m].
MAX_LOOK_SPREAD_M = 0.0005
#: A mount bias spread over this gets a warning [m].
MOUNT_SPREAD_WARN_M = 0.001
#: The hand's heading and tilt count as reached within this [deg].
HEADING_TOL_DEG = 2.0
#: Candidate headings of an approach, against the bearing of its point [deg], and how far the board may
#: lie from where the plan expects it when the plan counts what the camera will see [m].
HEADINGS_DEG = (90.0, -90.0, 0.0, 45.0, -45.0)
VIEW_SLACK_M = 0.006
#: The intrinsics' looks: TCP heights over the board [m], shifts along its long side [m], headings against
#: the bearing [deg] and leans of the hand away from the base [deg]. The arm leans the hand only in its own
#: vertical plane, and points it straight down only low over the board.
VIEW_HEIGHTS_M = (0.06, 0.08, 0.10, 0.12)
VIEW_SHIFTS_M = (0.0, 0.04, -0.04)
VIEW_HEADINGS_DEG = (90.0, -90.0)
VIEW_LEANS_DEG = (0.0, 10.0, 20.0, 30.0)
#: A calibration look needs this many board points.
MIN_VIEW_POINTS = 8
#: Jaw mask dilation [px] at `WORK_WIDTH_PX`, scaled with the frame width, and the frame margin [px] a
#: predicted marker keeps.
MASK_DILATE_PX = 6
FRAME_MARGIN_PX = 5
#: The hand rises at least this high before the joint move back to the start pose [m].
PARK_RISE_M = 0.03
#: Headings of the mount's pairs against the bearing [deg]: the wrist roll turns half a turn from each and
#: stays in its range.
MOUNT_HEADINGS_DEG = (90.0, -90.0, 45.0, -45.0, 135.0, -135.0)
#: When the camera sees too little of the board at the end of an approach, the wrist roll alone turns by one
#: of these [deg] and the camera looks again, at most `MAX_RELOOKS` times.
RELOOK_TURNS_DEG = (45.0, -45.0, 90.0, -90.0, 135.0, -135.0)
MAX_RELOOKS = 3
#: How `touch` words a target that would bring the arm onto itself.
SELF_CONTACT = "ramie uderzyloby samo w siebie"
#: A joint move is checked against the arm's own body every this much of its largest joint's turn [deg].
JOINT_CHECK_DEG = 2.0
#: A joint move steps each joint at most this share of its speed limit per tick.
JOINT_MOVE_SHARE = 0.5

Measure = Callable[["CalibrateApproachStrategy", Observation, "Planned"], "np.ndarray | str"]
OnStill = Callable[["CalibrateApproachStrategy", Observation, "Planned"], None]


@dataclass
class CalibrateApproachSettings:
    #: The board: squares along its short and long side, the measured square and marker sides [m], the
    #: OpenCV dictionary, its top above the table [m] and the detector's label prefix. As in `charuco:`.
    squares_x: int = 3
    squares_y: int = 5
    square_m: float = 0.045
    marker_m: float = 0.035
    dictionary: str = "DICT_6X6_250"
    thickness_m: float = 0.0
    label_prefix: str = "charuco"
    #: The approach table's JSON file [path]; required. Its folder is made if missing.
    table_file: str = ""
    #: The stand's name, for the file.
    stand: str = ""
    #: Add this run's board spot to the samples already in the file; false starts the file over.
    merge: bool = True
    #: The wrist camera: its key in `obs.frames` and robot.yaml, and an upside-down frame turned first.
    wrist_camera: str = "wrist"
    rotate_180: bool = False
    #: A look takes frames captured after the arm stood still and at most this old [s].
    max_frame_age_s: float = 0.3
    #: Corners to measure, comma-separated ids; empty takes every corner seen.
    corners: str = ""
    #: Final-leg directions, comma-separated: out, in, ccw, cw.
    directions: str = "out,in,ccw,cw"
    #: Surfaces the held block is released on, comma-separated [m above the table].
    levels_m: str = "0.0,0.029,0.058"
    #: Approaches from each direction at each corner and level.
    repeats: int = 2
    #: How far a held block hangs below the TCP [m]; 0 takes a cube picked off the table, as pick_place does.
    hang_m: float = 0.0
    #: The final horizontal leg [m], the carry's clearance over the tallest block [m], the gap under the held
    #: block at the release [m], the slow descent's speed [m/s] and the other moves' speed [m/s].
    lead_m: float = 0.03
    carry_clearance_m: float = 0.03
    place_gap_m: float = 0.002
    place_speed_mps: float = 0.02
    speed_mps: float = 0.10
    #: The arm stands still once no joint moves more than `still_deg` per tick for `still_s` [s, deg]; then
    #: `looks` frames are measured.
    still_s: float = 0.5
    still_deg: float = 0.1
    looks: int = 3
    #: The corners: the longest wait [s], sightings averaged per corner and the oldest sighting taken [s].
    corner_wait_s: float = 10.0
    corner_samples: int = 5
    corner_max_age_s: float = 30.0
    #: Looks for the wrist camera's intrinsics; reuse the intrinsics in `table_file` instead.
    intrinsics_views: int = 18
    reuse_intrinsics: bool = False
    #: Pairs of approaches half a turn apart per level, for the camera mount's bias.
    mount_pairs: int = 2
    #: Approach every corner once more with the table's correction and record what is left.
    verify: bool = True
    #: An approach is skipped when it does not arrive within `move_timeout_s` or sees no board within
    #: `look_timeout_s` [s]; more than `max_skip_share` skipped stops the run.
    move_timeout_s: float = 10.0
    look_timeout_s: float = 2.0
    max_skip_share: float = 0.25
    #: Ticks per second.
    hz: float = 20.0
    #: The run ends when an arm joint lags the supervisor's command by more than this [deg].
    max_tracking_deg: float = 25.0
    #: A safety net for the whole run [s]; 0 means none.
    time_limit_s: float = 3600.0


@dataclass(frozen=True)
class Planned:
    """One approach: where, how high, from which side and with which hand heading."""

    kind: str
    #: The point the TCP is commanded over [m, base frame], the release level and its TCP height [m].
    xy: np.ndarray
    level: int
    release_z: float
    carry_z: float
    direction: str
    #: The closing heading [rad] and the hand's tilt [rad] (views only).
    psi: float
    tilt: float = 0.0
    #: The board corner (-1 for none), the repeat (0 for a mount pair) and the pair it belongs to.
    corner: int = -1
    repeat: int = 0
    pair: int = -1
    #: Joints [rad, the description's order] that put the hand at the start of the final leg, or at a look
    #: over the board: the transit's goal.
    start_q: np.ndarray | None = None


@dataclass(frozen=True)
class Waypoint:
    """A straight move of the TCP to `point`, or a joint move to `joints` when given."""

    point: np.ndarray
    step_m: float
    #: The hand's closing heading [rad] and tilt [rad] on the way, or its whole rotation (3, 3); neither
    #: leaves it free.
    psi: float | None
    tilt: float = 0.0
    settle: bool = False
    joints: np.ndarray | None = None
    rotation: np.ndarray | None = None


@dataclass
class CalibrationStatus:
    """What the run has done, for the panel, logs and tests."""

    phase: str = CORNERS
    #: Approaches done in the phase and planned for it.
    index: int = 0
    total: int = 0
    #: Samples recorded and approaches skipped in the whole run.
    samples: int = 0
    skipped: int = 0
    #: The last correction measured [m, base frame x and y].
    last_delta_m: tuple[float, float] = (math.nan, math.nan)
    #: The corners as perceived [m, base frame], the board's yaw [deg], fit residual [m] and scale.
    corners: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    board_yaw_deg: float = math.nan
    fit_rms_m: float = math.nan
    fit_scale: float = math.nan
    #: The wrist camera: frame size, focal length [px], residual [px] and views.
    frame_size: tuple[int, int] = (0, 0)
    focal_px: float = math.nan
    k_rms_px: float = math.nan
    k_views: int = 0
    #: The mount bias per level [m, hand frame] and its spread over pairs [m].
    mount_bias_m: dict[int, tuple[float, float]] = field(default_factory=dict)
    mount_spread_m: dict[int, float] = field(default_factory=dict)
    #: The median repeat spread per direction [m] and the preferred direction.
    spread_m: dict[str, float] = field(default_factory=dict)
    preferred: str = ""
    #: What the self-check left, per approach [m].
    verify_m: list[float] = field(default_factory=list)
    #: Where the table went, and a partial one.
    table_path: str = ""
    partial_path: str = ""
    note: str = ""


def _ids(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"calibrate_approach: corners must be comma-separated numbers, got {text!r}")
        out.append(int(part))
    return out


def _floats(text: str, name: str) -> list[float]:
    try:
        values = [float(p) for p in str(text).split(",") if p.strip()]
    except ValueError:
        raise ValueError(f"calibrate_approach: {name} must be comma-separated numbers, "
                         f"got {text!r}") from None
    if not values:
        raise ValueError(f"calibrate_approach: {name} must not be empty")
    return values


class CalibrateApproachStrategy(TouchStrategy):
    """Measures where the hand lands on the board's corners from each side and height, and saves the table."""

    settings = CalibrateApproachSettings  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, settings: CalibrateApproachSettings | None = None, *,
                 schedule: Sequence[tuple[Any, int, str]] | None = None, on_still: OnStill | None = None,
                 measure: bool | Measure = True):
        cfg = settings if settings is not None else CalibrateApproachSettings()
        self.board = charuco.BoardSpec(cfg.squares_x, cfg.squares_y, cfg.square_m, cfg.marker_m,
                                       cfg.dictionary, cfg.thickness_m)
        self.board.check()
        if not cfg.table_file.strip() and measure is not False:
            raise ValueError("calibrate_approach: table_file must name the approach table's JSON file")
        if not " ".join(cfg.label_prefix.split()):
            raise ValueError("calibrate_approach: label_prefix must not be empty")
        self.corner_ids = _ids(cfg.corners)
        if any(k >= self.board.n_corners for k in self.corner_ids):
            raise ValueError(f"calibrate_approach: the board has corners 0 to {self.board.n_corners - 1}, "
                             f"got {cfg.corners!r}")
        self.directions = [d.strip() for d in cfg.directions.split(",") if d.strip()]
        if not self.directions or any(d not in DIRECTIONS for d in self.directions) \
                or len(set(self.directions)) != len(self.directions):
            raise ValueError(f"calibrate_approach: directions must be distinct names out of "
                             f"{', '.join(DIRECTIONS)}, got {cfg.directions!r}")
        self.levels_m = _floats(cfg.levels_m, "levels_m")
        if any(not 0 <= z < 0.2 for z in self.levels_m) or self.levels_m != sorted(set(self.levels_m)):
            raise ValueError(f"calibrate_approach: levels_m must rise within [0, 0.2) m, "
                             f"got {cfg.levels_m!r}")
        for name in ("repeats", "looks", "corner_samples", "intrinsics_views"):
            if not int(getattr(cfg, name)) >= 1:
                raise ValueError(f"calibrate_approach: {name} must be at least 1, is {getattr(cfg, name)}")
        if not cfg.mount_pairs >= 0:
            raise ValueError(f"calibrate_approach: mount_pairs must be zero or more, is {cfg.mount_pairs}")
        for name in ("lead_m", "carry_clearance_m", "place_speed_mps", "speed_mps", "still_s", "still_deg",
                     "max_frame_age_s", "corner_wait_s", "corner_max_age_s", "move_timeout_s",
                     "look_timeout_s", "hz", "max_tracking_deg"):
            value = getattr(cfg, name)
            if not (value > 0 and math.isfinite(value)):
                raise ValueError(f"calibrate_approach: {name} must be positive, is {value}")
        for name in ("hang_m", "place_gap_m", "time_limit_s"):
            value = getattr(cfg, name)
            if not value >= 0:
                raise ValueError(f"calibrate_approach: {name} must be zero or more, is {value}")
        if not 0 <= cfg.max_skip_share < 1:
            raise ValueError(f"calibrate_approach: max_skip_share must lie within [0, 1), "
                             f"is {cfg.max_skip_share}")
        if measure is not False and not cfg.wrist_camera:
            raise ValueError("calibrate_approach: wrist_camera must name the camera on the hand")
        if schedule is not None and measure is True:
            raise ValueError("calibrate_approach: an explicit schedule probes; pass measure=False or a "
                             "callable")
        super().__init__(TouchSettings(label=cfg.label_prefix, hz=cfg.hz, speed_mps=cfg.speed_mps,
                                       descend_speed_mps=cfg.place_speed_mps,
                                       max_tracking_deg=cfg.max_tracking_deg, time_limit_s=cfg.time_limit_s,
                                       close_jaw=False))
        self.calib = cfg
        self.prefix = " ".join(cfg.label_prefix.split()).lower()
        self.calibration = CalibrationStatus()
        self._schedule_in = None if schedule is None else [
            (np.asarray(xy, float)[:2].copy(), int(level), str(d)) for xy, level, d in schedule]
        self._on_still = on_still
        self._measure: bool | Measure = measure
        self._phases.update({CORNERS: self._wait_corners, INTRINSICS: self._run_phase, MOUNT: self._run_phase,
                             SAMPLES: self._run_phase, VERIFY: self._run_phase, SAVE: self._save,
                             PARK: self._park})
        self._reset_run()

    # ------------------------------------------------------------------ state
    def _reset_run(self) -> None:
        self._cv2: Any = None
        self._camera: RobotCamera | None = None
        self._cv_board: Any = None
        self._detector: Any = None
        self._K: np.ndarray | None = None
        self._dist: np.ndarray | None = None
        self._size: tuple[int, int] | None = None
        self._p_cam = np.zeros(3)
        self._start_joints: dict[str, float] = {}
        self._gripper = 0.0
        self._hang = 0.0
        self._levels: list[tuple[float, float, float, float]] = []
        self._pan_xy = np.zeros(2)
        self._sightings: dict[int, dict[int, np.ndarray]] = {}
        self._corners: dict[int, np.ndarray] = {}
        self._targets: list[int] = []
        self._fit: charuco.BoardFit | None = None
        self._existing: ApproachTable | None = None
        self._queue: list[Planned] = []
        self._phase_total = 0
        self._planned: Planned | None = None
        self._path: list[Waypoint] = []
        self._wp = 0
        self._wp_t0 = 0.0
        self._step = ""
        self._step_t0 = 0.0
        self._still_ticks = 0
        self._still_t0 = 0.0
        self._q_prev: np.ndarray | None = None
        self._looks: list[np.ndarray] = []
        self._look_views: list[charuco.BoardView] = []
        self._captures: set[float] = set()
        self._look_problem = ""
        self._views: list[charuco.BoardPoints] = []
        self._view_normals: list[np.ndarray] = []
        self._samples: list[dict[str, Any]] = []
        self._pairs: dict[int, dict[str, Any]] = {}
        self._mount_raw: list[dict[str, Any]] = []
        self._bias: dict[int, np.ndarray] = {}
        self._skipped: list[dict[str, Any]] = []
        self._verify: list[dict[str, Any]] = []
        self._attempted = 0
        self._table: ApproachTable | None = None
        self._saved = False
        self._failed = ""
        self._joint_goal: np.ndarray | None = None
        self._t_start = 0.0
        self._intrinsics: charuco.Intrinsics | None = None
        self._moving = False
        self._safe_q: np.ndarray | None = None
        self._hub_q: list[np.ndarray] | None = None
        self._park_route: list[np.ndarray] | None = None
        self._escape_q: np.ndarray | None = None
        self._relooks = 0
        self._rolls_tried: list[float] = []
        self._hit: str | None = None
        self._sample_done = 0
        self._sample_skips = 0
        self._probe_mode = self._schedule_in is not None

    @property
    def cv2(self) -> Any:
        return self._cv2

    @property
    def K(self) -> np.ndarray | None:
        return self._K

    @property
    def planned(self) -> Planned | None:
        """The approach in progress."""
        return self._planned

    @property
    def fit(self) -> charuco.BoardFit | None:
        return self._fit

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self._samples)

    @property
    def table(self) -> ApproachTable | None:
        return self._table

    def level_heights(self) -> list[tuple[float, float, float, float]]:
        """Per level: (surface, release TCP height, wanted carry height, lowest carry height) [m]."""
        return list(self._levels)

    # ------------------------------------------------------------------ start
    def start(self, run: RunInfo, obs: Observation) -> None:
        self._reset_run()
        super().start(run, obs)
        cfg, kin = self.calib, self.kin
        self.calibration = CalibrationStatus()
        self._start_joints = dict(obs.joints)
        self._t_start = float(obs.t)
        spec = kin.spec
        cube = block(BLOCK)
        width, height = 2 * cube.half_size[0], 2 * cube.half_size[2]
        self._gripper = kin.gripper_for_gap(width)
        if spec.gripper is not None:
            g = spec.joints.index(spec.gripper)
            lo, hi = self.limits
            self.q_cmd[g] = float(np.clip(kin.to_q({spec.gripper: self._gripper})[g], lo[g], hi[g]))
            self._contacts = kin.self_contacts(kin.from_q(self.q_cmd))
        if cfg.hang_m > 0:
            self._hang = float(cfg.hang_m)
        else:
            centre = kin.jaw_centre(kin.gripper_for_gap(width + 2 * OPEN_MARGIN_M))
            self._hang = max(MIN_GRASP_Z, height - GRASP_DEPTH * height + float(centre[0]))
        reach = kin.jaw_reach(self._gripper)
        self._levels = []
        for top in self.levels_m:
            release, want, low = place_heights(top, self._hang, cfg.place_gap_m, GRIP_SLIP_M, TABLE_TOP_M,
                                               cfg.carry_clearance_m, MIN_CARRY_CLEAR_M)
            lowest = release - reach
            if lowest < cfg.thickness_m + JAW_BOARD_CLEAR_M:
                raise ValueError(f"calibrate_approach: at level {top * 1000:.0f} mm the jaw would come "
                                 f"within {(lowest - cfg.thickness_m) * 1000:.1f} mm of the board")
            self._levels.append((top, release, want, low))
        self._pan_xy = kin.joint_anchor(spec.arm_joints[0], spec.home)[:2].copy()
        if self._measure is True:
            cv2 = _optional("cv2")
            if cv2 is None or not hasattr(cv2, "aruco"):
                raise RuntimeError("calibrate_approach: the wrist camera needs OpenCV with the aruco module: "
                                   "pip install opencv-contrib-python (the twin brings it)")
            camera = spec.cameras.get(cfg.wrist_camera)
            if camera is None:
                raise RuntimeError(f"calibrate_approach: camera {cfg.wrist_camera!r} is not in robot.yaml "
                                   f"`cameras` ({', '.join(spec.cameras) or 'none'})")
            self._cv2, self._camera = cv2, camera
            self._cv_board, self._detector = charuco.charuco_detector(self.board, cv2, min_markers=1)
            home = dict(spec.home, **({spec.gripper: self._gripper} if spec.gripper else {}))
            self._p_cam = (inverse(kin.camera(cfg.wrist_camera, home)) @ kin.tcp(home))[:3, 3].copy()
        if self._measure is not False:
            path = Path(cfg.table_file).expanduser()
            if cfg.merge and path.exists():
                try:
                    self._existing = ApproachTable.load(path)
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"calibrate_approach: {exc}; set merge: false to start the file "
                                       "over") from exc
                why = self._existing.merge_problem(robot=spec.name, board=self.board.as_dict(), size=None,
                                                   levels=self._level_dicts())
                if why:
                    raise RuntimeError(f"calibrate_approach: the table file does not match this run ({why}); "
                                       "set merge: false to start it over")
        if self._probe_mode:
            self._plan_probes()
            self._begin(SAMPLES)
        else:
            self._enter(CORNERS)
            self.calibration.phase = CORNERS

    def _level_dicts(self) -> list[dict[str, float]]:
        return [{"surface_m": top, "tcp_z_m": release} for top, release, _, _ in self._levels]

    # ------------------------------------------------------------------ corners
    def _wait_corners(self, obs: Observation) -> Action | None:
        cfg = self.calib
        for key, pose in obs.objects.items():
            k = charuco.corner_of(self.prefix, str(key))
            if k is None or k >= self.board.n_corners:
                continue
            if str(pose.source) not in CORNER_SOURCES or not float(pose.age_s) <= cfg.corner_max_age_s:
                continue
            position = np.asarray(pose.position, float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                continue
            # One capture is one sighting, however many ticks read it.
            seen_at = int(round((float(obs.t) - float(pose.age_s)) / SIGHTING_S))
            self._sightings.setdefault(k, {})[seen_at] = position.copy()
        if self._measure is True and self._size is None:
            frame = obs.frames.get(cfg.wrist_camera)
            if frame is not None:
                h, w = np.asarray(frame.image).shape[:2]
                self._size = (int(w), int(h))
                self.calibration.frame_size = self._size
        every = list(range(self.board.n_corners))
        enough = [k for k in every if len(self._sightings.get(k, {})) >= cfg.corner_samples]
        wanted = self.corner_ids or every
        waited = self._t - self._phase_t0
        done = all(k in enough for k in every) or waited >= cfg.corner_wait_s
        if not done or (self._measure is True and self._size is None and waited < cfg.corner_wait_s):
            return self.hold(f"czekam na rogi tablicy z kamer: {len(enough)} z {len(every)} "
                             f"(detektor {self.prefix})")
        if self._measure is True and self._size is None:
            return self._fail(f"brak klatki z kamery {cfg.wrist_camera}")
        lacking = [k for k in wanted if k not in enough]
        if len(enough) < MIN_CORNERS or (self.corner_ids and lacking):
            missing = f" (brak {', '.join(f'c{k}' for k in lacking)})" if self.corner_ids else ""
            return self._fail(f"za malo rogow tablicy: {len(enough)} z {len(every)}, widzianych "
                              f"{len(self._sightings)}, potrzeba {MIN_CORNERS}{missing} - wlacz detektor "
                              f"{self.prefix}, odsun ramie")
        # Every corner seen fits the board; the listed ones are measured.
        self._corners = {k: np.mean(list(self._sightings[k].values()), axis=0) for k in sorted(enough)}
        self._targets = [k for k in sorted(enough) if k in wanted]
        ids = sorted(self._corners)
        fit = charuco.fit_board(self.board.corner_xy(ids), np.array([self._corners[k][:2] for k in ids]))
        self._fit = fit
        st = self.calibration
        st.corners = {k: (float(p[0]), float(p[1]), float(p[2])) for k, p in self._corners.items()}
        st.board_yaw_deg, st.fit_rms_m, st.fit_scale = math.degrees(fit.yaw), fit.rms_m, fit.scale
        if abs(fit.scale - 1.0) > FIT_MAX_SCALE or fit.rms_m > FIT_MAX_RMS_M:
            seen_mm = fit.scale * cfg.square_m * 1000
            return self._fail(f"rogi tablicy nie pasuja do jej wymiarow (kwadrat {seen_mm:.1f} mm, "
                              f"ustawiono {cfg.square_m * 1000:.1f}; dopasowanie {fit.rms_m * 1000:.1f} mm)")
        if self._existing is not None and self._size is not None:
            why = self._existing.merge_problem(robot=self.kin.spec.name, board=self.board.as_dict(),
                                               size=self._size, levels=self._level_dicts())
            if why:
                return self._fail(f"plik tablicy nie pasuje do tego przebiegu ({why}) - ustaw merge: false")
        if self._measure is True:
            assert self._size is not None and self._camera is not None
            K = pinhole(self._camera.fovy_deg, (self._size[1], self._size[0]))
            self._K, self._dist = K, None
            reuse = self._reused_intrinsics() if cfg.reuse_intrinsics else None
            if reuse is not None:
                self._K, self._dist = reuse
                st.focal_px = float(self._K[0, 0])
                return self._after_intrinsics()
            self._plan_views()
            if self._queue:
                return self._begin(INTRINSICS)
            return self._fail("kalibracja kamery nadgarstka: zadne ujecie nad tablica nie jest osiagalne")
        return self._after_intrinsics()

    def _reused_intrinsics(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        wrist = (self._existing.data.get("wrist") if self._existing is not None else None) or {}
        if wrist.get("K") is None or self._size is None or list(wrist.get("size") or []) != list(self._size):
            return None
        dist = wrist.get("dist")
        return np.asarray(wrist["K"], float), None if dist is None else np.asarray(dist, float)

    # ------------------------------------------------------------------ planning
    def _reaches(self, point: np.ndarray, psi: float, seed: Mapping[str, float], tilt_rad: float = 0.0
                 ) -> dict[str, float] | None:
        sol = self.kin.ik(point, hand_down(psi, tilt_rad), seed=seed, accept=self._clear,
                          restarts=ik_restarts(tilt_rad) if tilt_rad else 2)
        if sol.ok and self._clear(sol.joints) and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG):
            return sol.joints
        return None

    def _seed(self) -> dict[str, float]:
        spec = self.kin.spec
        seed = dict(spec.home)
        if spec.gripper is not None:
            seed[spec.gripper] = self._gripper
        return seed

    def _visible(self, joints: Mapping[str, float]) -> float:
        """The board points a look at `joints` should give solvePnP: the fewest over the board shifted by
        `VIEW_SLACK_M` each way, as the plan cannot know the errors the calibration measures. A marker counts
        with its white square whole in the frame and clear of the jaws."""
        if self._cv2 is None or self._K is None or self._size is None or self._fit is None:
            return 0.0
        cv2, K, (W, H) = self._cv2, self._K, self._size
        T = self.kin.camera(self.calib.wrist_camera, joints)
        dilate = max(1, round(MASK_DILATE_PX * W / WORK_WIDTH_PX))
        mask = jaw_mask(self.kin, joints, K, T, (H, W), cv2, dilate)
        markers = np.asarray(self._cv_board.getObjPoints(), float).reshape(-1, 4, 3)
        centres = markers.mean(axis=1)
        half = self.board.square_m / 2
        squares = (centres[:, None, :] + np.array([[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0],
                                                    [-half, half, 0.0]])[None]).reshape(-1, 3)
        corners = np.asarray(self._cv_board.getChessboardCorners(), float).reshape(-1, 3)
        near = np.linalg.norm(corners[:, None, :2] - centres[None, :, :2], axis=2) < 0.8 * self.board.square_m
        fit, m = self._fit, FRAME_MARGIN_PX
        R = fit.R_base_board

        def seen(points: np.ndarray, shift: np.ndarray) -> np.ndarray:
            base = points @ R.T
            base[:, :2] += fit.origin + shift
            base[:, 2] += self.calib.thickness_m
            uv, depth = project(K, T, base)
            ok = ((depth > 0) & (uv[:, 0] >= m) & (uv[:, 0] <= W - 1 - m) & (uv[:, 1] >= m)
                  & (uv[:, 1] <= H - 1 - m))
            ui = np.clip(np.nan_to_num(uv[:, 0]).round().astype(int), 0, W - 1)
            vi = np.clip(np.nan_to_num(uv[:, 1]).round().astype(int), 0, H - 1)
            return ok & ~mask[vi, ui]

        counts = []
        for shift in ((0.0, 0.0), (VIEW_SLACK_M, 0.0), (-VIEW_SLACK_M, 0.0), (0.0, VIEW_SLACK_M),
                      (0.0, -VIEW_SLACK_M)):
            d = np.array(shift)
            whole = np.asarray(seen(squares, d).reshape(-1, 4).all(axis=1), bool)
            found = seen(corners, d) & (near & whole[None, :]).any(axis=1)
            n = int(found.sum())
            counts.append(n if n >= charuco.MARKERS_BELOW else n + 4 * int(whole.sum()))
        return float(min(counts))

    def _heading_for(self, xy: np.ndarray, release_z: float) -> tuple[float, dict[str, float]] | None:
        """The heading of approaches to `xy` at `release_z` that shows the camera the most of the board."""
        b = bearing(xy, self._pan_xy)
        best: tuple[float, int, float, dict[str, float]] | None = None
        for rank, off in enumerate(HEADINGS_DEG):
            psi = wrap(b + math.radians(off))
            joints = self._reaches(np.array([*xy, release_z]), psi, self._seed())
            if joints is None:
                continue
            seen = self._visible(joints) if self._measure is True else 0.0
            key = (seen, -rank)
            if best is None or key > best[:2]:
                best = (seen, -rank, psi, joints)
            if self._measure is not True:
                break
        return None if best is None else (best[2], best[3])

    def _approach_plan(self, kind: str, xy: np.ndarray, level: int, direction: str, psi: float,
                       at: Mapping[str, float], **extra: Any) -> Planned | str:
        """An approach to `xy` at `level` from `direction`, or why it cannot be made."""
        _, release, want, low = self._levels[level]
        d = direction_vector(direction, xy, self._pan_xy)
        lead = lead_point(xy, d, self.calib.lead_m)
        for z in carry_heights(want, low):
            above = self._reaches(np.array([*xy, z]), psi, at)
            if above is None:
                continue
            start = self._reaches(np.array([*lead, z]), psi, above)
            if start is not None:
                return Planned(kind, np.asarray(xy, float).copy(), level, release, z, direction, psi,
                               start_q=self.kin.to_q(start), **extra)
        return "punkt prowadzacy poza zasiegiem"

    def _plan_views(self) -> None:
        """Looks over the board for the intrinsics, taken in turn from each heading and lean."""
        assert self._fit is not None
        centre = self._fit.to_base(np.array(self.board.size_m) / 2)
        along = self._fit.R_base_board[:2, 1]
        groups = []
        for lean in VIEW_LEANS_DEG:
            for off in VIEW_HEADINGS_DEG:
                # A positive tilt leans the approach toward the hand's y axis, which points to the base at
                # a heading a quarter turn counter-clockwise from the bearing.
                t = math.radians(-lean if off > 0 else lean)
                groups.append([(z, centre + shift * along, off, t) for z in VIEW_HEIGHTS_M
                               for shift in VIEW_SHIFTS_M])
        views: list[Planned] = []
        while len(views) < self.calib.intrinsics_views and any(groups):
            for group in groups:
                while group and len(views) < self.calib.intrinsics_views:
                    z, xy, off, t = group.pop(0)
                    psi = wrap(bearing(xy, self._pan_xy) + math.radians(off))
                    joints = self._reaches(np.array([*xy, z]), psi, self._seed(), t)
                    if joints is not None:
                        views.append(Planned(VIEW, np.asarray(xy, float).copy(), -1, z, z, "", psi, t,
                                             start_q=self.kin.to_q(joints)))
                        break
        self._queue = views

    def _plan_samples(self) -> None:
        cfg = self.calib
        ids = self._order(self._targets)
        queue, skipped = [], []
        for k in ids:
            xy = self._corners[k][:2]
            for level, (_, release, _, _) in enumerate(self._levels):
                chosen = self._heading_for(xy, release)
                if chosen is None:
                    skipped.append({"corner": k, "level": level, "direction": "", "repeat": 0, "kind": SAMPLE,
                                    "why": "punkt poza zasiegiem z chwytakiem w dol"})
                    continue
                psi, at = chosen
                plans = {}
                for d in self.directions:
                    plans[d] = self._approach_plan(SAMPLE, xy, level, d, psi, at, corner=k)
                for r in range(1, cfg.repeats + 1):
                    for d in self.directions:
                        p = plans[d]
                        if isinstance(p, str):
                            if r == 1:
                                skipped.append({"corner": k, "level": level, "direction": d, "repeat": 0,
                                                "kind": SAMPLE, "why": p})
                            continue
                        queue.append(replace(p, repeat=r))
        self._queue = queue
        self._skipped.extend(skipped)

    def _roll_flip(self, joints: Mapping[str, float]) -> dict[str, float] | None:
        """`joints` with only the wrist roll turned half a turn, within the limits; None when it cannot."""
        spec = self.kin.spec
        if spec.roll_joint is None or spec.roll_joint not in joints:
            return None
        roll = float(joints[spec.roll_joint])
        flipped = dict(joints, **{spec.roll_joint: roll - 180.0 if roll > 0 else roll + 180.0})
        return flipped if self._clear(flipped) else None

    def _plan_mount(self) -> None:
        """Per level, pairs of looks half a turn of the wrist apart: an approach to a corner, then the wrist
        roll alone turned half a turn, the arm's other joints still."""
        cfg = self.calib
        queue = []
        pair = 0
        d = self.directions[0]
        for level, (_, release, _, _) in enumerate(self._levels):
            options = []
            # Any corner seen will do: the bias belongs to the camera, not to the corners measured.
            for k in sorted(self._corners):
                xy = self._corners[k][:2]
                b = bearing(xy, self._pan_xy)
                best: tuple[float, int, Planned] | None = None
                for off in MOUNT_HEADINGS_DEG:
                    psi = wrap(b + math.radians(off))
                    joints = self._reaches(np.array([*xy, release]), psi, self._seed())
                    flipped = None if joints is None else self._roll_flip(joints)
                    if joints is None or flipped is None:
                        continue
                    seen = (min(self._visible(joints), self._visible(flipped)) if self._measure is True
                            else 1.0)
                    if best is not None and seen <= best[0]:
                        continue
                    plan = self._approach_plan(MOUNT_A, xy, level, d, psi, joints, corner=k)
                    if not isinstance(plan, str):
                        best = (seen, k, plan)
                    if self._measure is not True:
                        break
                if best is not None:
                    options.append(best)
            # The best seen corner first, then each next one farthest from those chosen.
            good = sorted((o for o in options if o[0] >= charuco.MIN_POINTS or self._measure is not True),
                          key=lambda o: -o[0])
            chosen: list[tuple[float, int, Planned]] = good[:1]
            while len(chosen) < min(cfg.mount_pairs, len(good)):
                rest = [o for o in good if o not in chosen]
                chosen.append(max(rest, key=lambda o: min(float(np.linalg.norm(self._corners[o[1]][:2]
                                                                               - self._corners[c[1]][:2]))
                                                         for c in chosen)))
            for _, _, a in chosen:
                queue.append(replace(a, pair=pair))
                queue.append(replace(a, kind=MOUNT_B, psi=wrap(a.psi + math.pi), pair=pair, start_q=None))
                pair += 1
        self._queue = queue

    def _plan_verify(self) -> None:
        assert self._table is not None
        preferred = self._table.preferred_direction or self.directions[0]
        queue = []
        for k in self._order(self._targets):
            xy = self._corners[k][:2]
            for level, (_, release, _, _) in enumerate(self._levels):
                hit = self._table.lookup(xy, release, preferred)
                if not isinstance(hit, Hit):
                    continue
                target = xy + np.asarray(hit.delta_xy, float)
                chosen = self._heading_for(target, release)
                if chosen is None:
                    continue
                plan = self._approach_plan(CHECK, target, level, preferred, chosen[0], chosen[1], corner=k)
                if not isinstance(plan, str):
                    queue.append(plan)
        self._queue = queue

    def _plan_probes(self) -> None:
        assert self._schedule_in is not None
        queue = []
        for xy, level, d in self._schedule_in:
            if not 0 <= level < len(self._levels):
                raise ValueError(f"calibrate_approach: schedule level {level} outside "
                                 f"0..{len(self._levels) - 1}")
            chosen = self._heading_for(xy, self._levels[level][1])
            plan = (self._approach_plan(PROBE, xy, level, d, chosen[0], chosen[1]) if chosen is not None
                    else "punkt poza zasiegiem z chwytakiem w dol")
            if isinstance(plan, str):
                self._skipped.append({"corner": -1, "level": level, "direction": d, "repeat": 0,
                                      "kind": PROBE, "why": plan})
                continue
            queue.append(plan)
        self._queue = queue

    def _order(self, ids: list[int]) -> list[int]:
        """Corners in nearest-neighbour order from the one nearest the hand."""
        if not ids:
            return []
        left = list(ids)
        here = self.tcp_cmd[:2]
        out = []
        while left:
            k = min(left, key=lambda c: float(np.linalg.norm(self._corners[c][:2] - here)))
            out.append(k)
            left.remove(k)
            here = self._corners[k][:2]
        return out

    # ------------------------------------------------------------------ phases of approaches
    def _begin(self, phase: str) -> Action | None:
        self._enter(phase)
        self.calibration.phase = phase
        self.calibration.index, self.calibration.total = 0, len(self._queue)
        self._phase_total = len(self._queue)
        self._planned = None
        return None

    def _run_phase(self, obs: Observation) -> Action | None:
        """Run the queued approaches of the phase one by one, then move on."""
        if self._planned is None:
            if not self._queue:
                return self._phase_done()
            planned = self._queue.pop(0)
            self._planned = planned
            if planned.kind == MOUNT_B and not any(m["pair"] == planned.pair and m["kind"] == MOUNT_A
                                                   for m in self._mount_raw):
                return self._skip("para bez pierwszego pomiaru")
            why = self._start_approach(planned)
            if why:
                return self._skip(why)
        return self._approach_step(obs)

    def _phase_done(self) -> Action | None:
        phase = self.status.phase
        if phase == INTRINSICS:
            return self._solve_intrinsics()
        if phase == MOUNT:
            self._solve_mount()
            if not self._pairs:
                return self._fail("obrot chwytaka: zadna para nie dala pomiaru - bez niego tablica bylaby "
                                  "przesunieta o blad mocowania kamery")
            return self._after_mount()
        if phase == SAMPLES:
            if self._probe_mode:
                return self._begin_park()
            return self._after_samples()
        if phase == VERIFY:
            return self._begin_save()
        raise RuntimeError(f"calibrate_approach: no end for phase {phase}")

    def _after_intrinsics(self) -> Action | None:
        if self.calib.mount_pairs > 0 and self._measure is not False:
            self._plan_mount()
            if not self._queue:
                return self._fail("obrot chwytaka: zaden rog nie bedzie widoczny w obu obrotach")
            return self._begin(MOUNT)
        return self._after_mount()

    def _after_mount(self) -> Action | None:
        self._plan_samples()
        if not self._queue:
            return self._fail("zadne podejscie do rogow tablicy nie jest osiagalne")
        return self._begin(SAMPLES)

    def _after_samples(self) -> Action | None:
        self._table = self._build_table(complete=True)
        st = self.calibration
        st.spread_m = dict(self._table.data.get("spread_m") or {})
        st.preferred = self._table.preferred_direction
        if self.calib.verify:
            self._plan_verify()
            if self._queue:
                return self._begin(VERIFY)
        return self._begin_save()

    # ------------------------------------------------------------------ one approach
    def _joint_path_clear(self, qa: np.ndarray, qb: np.ndarray) -> bool:
        """Whether the straight joint move from `qa` to `qb` keeps the arm off itself."""
        arm = self.kin.arm
        n = max(1, math.ceil(math.degrees(float(np.abs(qb - qa)[arm].max())) / JOINT_CHECK_DEG))
        base = self.kin.self_contacts(self.kin.from_q(qa))
        return not any(self.kin.self_contacts(self.kin.from_q(qa + s * (qb - qa))) - base
                       for s in np.linspace(0.0, 1.0, n + 1)[1:])

    def _joint_route(self, qa: np.ndarray, qb: np.ndarray) -> list[np.ndarray] | None:
        """Joint goals from `qa` to `qb` that keep the arm off itself: straight, or through one of the hub
        poses; None when neither does."""
        if self._joint_path_clear(qa, qb):
            return [qb]
        for hub in self._hubs():
            if self._joint_path_clear(qa, hub) and self._joint_path_clear(hub, qb):
                return [hub, qb]
        return None

    def _hubs(self) -> list[np.ndarray]:
        """Poses a blocked joint move may go through: the hand down over the board at the top carry height,
        across the arm and along it, then the arm's rest pose."""
        if self._hub_q is None:
            if self._fit is not None:
                centre = self._fit.to_base(np.array(self.board.size_m) / 2)
            else:
                centre = np.mean([p.xy for p in self._queue], axis=0) if self._queue else np.array([0.2, 0.0])
            z = max((carry for _, _, carry, _ in self._levels), default=0.08)
            hubs = []
            for off in (90.0, 0.0, -90.0):
                joints = self._reaches(np.array([*centre, z]), wrap(bearing(centre, self._pan_xy)
                                                                    + math.radians(off)), self._seed())
                if joints is not None:
                    hubs.append(self.kin.to_q(joints))
            hubs.append(np.clip(self.kin.to_q(self._seed()), *self.limits))
            self._hub_q = hubs
        return self._hub_q

    def _start_approach(self, planned: Planned) -> str:
        """Lay out the moves of `planned`; the reason it cannot be made, or ''."""
        cfg = self.calib
        here = self.tcp_cmd.copy()
        fast = self.approach_step_m
        path: list[Waypoint] = []
        self._joint_goal = None
        # Up from where the hand is keeps its rotation, so it retraces the way down.
        up = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, :3].copy()
        q_from = self.q_cmd.copy()
        if self._escape_q is not None and planned.kind != MOUNT_B:
            # The last approach stopped short of hitting the arm: back the way it came first.
            back = self._joint_route(q_from, self._escape_q)
            path.extend(Waypoint(here, fast, None, joints=q) for q in back or [])
            q_from = self._escape_q if back else q_from
            self._escape_q = None
        if planned.kind == MOUNT_B:
            # The wrist roll alone, half a turn: the arm's other joints and their play stay where they are.
            flipped = self._roll_flip(self.kin.from_q(self.q_cmd))
            if flipped is not None:
                self._joint_goal = self.kin.to_q(flipped)
        else:
            assert planned.start_q is not None
            route = self._joint_route(q_from, planned.start_q)
            if route is None:
                return SELF_CONTACT + " w drodze do punktu"
            if planned.kind == VIEW:
                # Up first, then the joints to the look.
                path.append(Waypoint(np.array([*here[:2], max(here[2], planned.carry_z)]), fast, None,
                                     rotation=up))
                path.extend(Waypoint(here, fast, planned.psi, planned.tilt, joints=q) for q in route)
            else:
                # Up to the carry height, the joints across to the lead point, the leg, then down.
                d = direction_vector(planned.direction, planned.xy, self._pan_xy)
                lead = lead_point(planned.xy, d, cfg.lead_m)
                z = planned.carry_z
                path.append(Waypoint(np.array([*here[:2], max(here[2], z)]), fast, None, rotation=up))
                path.extend(Waypoint(np.array([*lead, z]), fast, planned.psi, joints=q) for q in route)
                path.append(Waypoint(np.array([*planned.xy, z]), fast, planned.psi, settle=True))
                above = min(planned.release_z + SLOW_ABOVE_M, z)
                if above > planned.release_z + 1e-6:
                    path.append(Waypoint(np.array([*planned.xy, above]), fast, planned.psi, settle=True))
                bottom = np.array([*planned.xy, planned.release_z])
                path.append(Waypoint(bottom, self.descend_step_m, planned.psi))
        self._path, self._wp, self._wp_t0 = path, 0, self._t
        self._step, self._step_t0 = ("roll" if planned.kind == MOUNT_B else "move"), self._t
        if planned.kind != MOUNT_B:
            # Where this approach starts: a pose the arm stands in, to back out to.
            self._safe_q = self.q_cmd.copy()
        self._settle_count = 0
        self._still_ticks, self._q_prev = 0, None
        self._looks, self._look_views, self._captures, self._look_problem = [], [], set(), ""
        self._relooks, self._rolls_tried = 0, []
        self._note_approach(planned)
        return ""

    def _transit(self, obs: Observation, wp: Waypoint) -> Action | None:
        """A joint move to `wp.joints`, then on to the next waypoint. Its path is no straight line, but it
        runs between two hand poses at one height over a cleared table, and it never leaves the arm on the
        other branch of its inverse kinematics."""
        if self._joint_goal is None:
            assert wp.joints is not None
            self._joint_goal = np.asarray(wp.joints, float).copy()
        if self._t - self._wp_t0 > self.calib.move_timeout_s:
            self._joint_goal = None
            return self._skip("ramie nie dojechalo")
        self._moving = True
        try:
            action = self._joint_move(obs)
        finally:
            self._moving = False
        if self._hit is not None:
            why, self._hit = self._hit, None
            self._joint_goal = None
            self._escape_q = self._safe_q
            return self._skip(why)
        if action is not None:
            return action
        self._wp += 1
        self._wp_t0 = self._t
        self._settle_count = 0
        if self._wp >= len(self._path):
            self._step, self._step_t0 = STILL, self._t
            self._still_ticks, self._q_prev = 0, None
            return self.hold(self._approach_note())
        return self._approach_step(obs)

    def _turned(self, wp: Waypoint) -> bool:
        if wp.psi is None:
            return True
        joints = self.kin.from_q(self.q_cmd)
        tol = math.radians(HEADING_TOL_DEG)
        return (abs(wrap(heading(self.kin, joints) - wp.psi)) <= tol
                and abs(tilt(self.kin, joints) - abs(wp.tilt)) <= tol)

    def _approach_step(self, obs: Observation) -> Action | None:
        planned = self._planned
        assert planned is not None
        cfg = self.calib
        if self._step == "roll":
            if self._joint_goal is None:
                return self._skip("obrot chwytaka poza zakresem")
            if self._t - self._step_t0 > cfg.move_timeout_s:
                self._joint_goal = None
                return self._skip("ramie nie dojechalo")
            self._moving = True
            try:
                action = self._joint_move(obs)
            finally:
                self._moving = False
            if self._hit is not None:
                why, self._hit = self._hit, None
                self._joint_goal = None
                self._escape_q = self._safe_q
                return self._skip(why)
            if action is not None:
                return action
            self._step, self._step_t0 = STILL, self._t
            self._still_ticks, self._q_prev = 0, None
            return self.hold(self._approach_note())
        if self._step == "move":
            wp = self._path[self._wp]
            if wp.joints is not None:
                return self._transit(obs, wp)
            done = self._settled(wp.point) if wp.settle else self._arrived(wp.point)
            if done and self._turned(wp):
                self._wp += 1
                self._wp_t0 = self._t
                self._settle_count = 0
                if self._wp >= len(self._path):
                    self._step, self._step_t0 = STILL, self._t
                    self._still_ticks, self._q_prev = 0, None
                    return self.hold(self._approach_note())
                wp = self._path[self._wp]
            if self._t - self._wp_t0 > cfg.move_timeout_s:
                return self._skip("ramie nie dojechalo")
            self._moving = True
            try:
                rotation = wp.rotation if wp.rotation is not None else (
                    None if wp.psi is None else hand_down(wp.psi, wp.tilt))
                action = self._drive(wp.point, wp.step_m, rotation=rotation)
            finally:
                self._moving = False
            if self._hit is not None:
                why, self._hit = self._hit, None
                self._escape_q = self._safe_q
                return self._skip(why)
            return action
        if self._step == STILL:
            if self._stood_still(obs):
                if self._on_still is not None:
                    self._on_still(self, obs, planned)
                if self._measure is False:
                    return self._finish_approach(None)
                self._step, self._step_t0 = LOOK, self._t
                return self._look(obs)
            if self._t - self._step_t0 > cfg.move_timeout_s:
                return self._skip("ramie nie dojechalo")
            return self.hold(self._approach_note())
        if self._step == LOOK:
            if not self._stood_still(obs):
                self._step, self._step_t0 = STILL, self._t
                return self.hold(self._approach_note())
            return self._look(obs)
        raise RuntimeError(f"calibrate_approach: unknown step {self._step}")

    def _relook(self) -> bool:
        """Turn the wrist roll alone to the heading whose view should hold the most of the board, when the
        camera sees too little of it. The other joints and their play stay, so the hand's error does too, and
        the TCP's move round the roll axis is in its command. False when no turn is left to try."""
        planned = self._planned
        spec = self.kin.spec
        if planned is None or planned.kind in (VIEW, MOUNT_B) or self._relooks >= MAX_RELOOKS \
                or spec.roll_joint is None or self._measure is not True:
            return False
        joints = self.kin.from_q(self.q_cmd)
        roll = float(joints[spec.roll_joint])
        if not any(abs(roll - tried) < 1.0 for tried in self._rolls_tried):
            self._rolls_tried.append(roll)
        best: tuple[float, dict[str, float]] | None = None
        for turn in RELOOK_TURNS_DEG:
            if any(abs(roll + turn - tried) < 1.0 for tried in self._rolls_tried):
                continue
            option = dict(joints, **{spec.roll_joint: roll + turn})
            if not self._clear(option):
                continue
            seen = self._visible(option)
            if best is None or seen > best[0]:
                best = (seen, option)
        if best is None:
            return False
        self._relooks += 1
        self._rolls_tried.append(float(best[1][spec.roll_joint]))
        self._joint_goal = self.kin.to_q(best[1])
        self._step, self._step_t0 = "roll", self._t
        self._looks, self._look_views, self._captures = [], [], set()
        return True

    def _stood_still(self, obs: Observation) -> bool:
        """Whether every arm joint has moved at most `still_deg` per tick for `still_s`, with the command on
        its target."""
        arm = self.kin.arm
        q = np.degrees(self.kin.to_q(dict(obs.joints))[arm])
        moved = math.inf if self._q_prev is None else float(np.abs(q - self._q_prev).max())
        self._q_prev = q
        if moved <= self.calib.still_deg and self._command_on_target:
            if self._still_ticks == 0:
                self._still_t0 = self._t
            self._still_ticks += 1
        else:
            self._still_ticks = 0
        return self._still_ticks / self.hz >= self.calib.still_s

    def _look(self, obs: Observation) -> Action | None:
        planned = self._planned
        assert planned is not None
        cfg = self.calib
        if self._t - self._step_t0 > cfg.look_timeout_s:
            if self._relook():
                return self.hold(self._approach_note())
            why = "tablica niewidoczna w kamerze nadgarstka"
            return self._skip(f"{why} ({self._look_problem})" if self._look_problem else why)
        if callable(self._measure):
            got = self._measure(self, obs, planned)
            if isinstance(got, str):
                self._look_problem = got
                return self.hold(self._approach_note())
            self._looks.append(np.asarray(got, float).copy())
        else:
            frame = obs.frames.get(cfg.wrist_camera)
            if frame is None:
                self._look_problem = "brak klatki"
                return self.hold(self._approach_note())
            captured = round(float(obs.t) - float(frame.age_s), 4)
            if float(frame.age_s) > cfg.max_frame_age_s or captured < self._still_t0 - 1e-9 \
                    or captured in self._captures:
                return self.hold(self._approach_note())
            self._captures.add(captured)
            image = np.asarray(frame.image)
            if cfg.rotate_180:
                image = image[::-1, ::-1]
            got = self._measure_frame(image, obs, planned)
            if isinstance(got, str):
                self._look_problem = got
                return self.hold(self._approach_note())
            if planned.kind == VIEW:
                return self._finish_approach(None)
            self._looks.append(got)
        if len(self._looks) < cfg.looks:
            return self.hold(self._approach_note())
        raws = np.array(self._looks)
        spread = float(np.linalg.norm(raws[:, None, :2] - raws[None, :, :2], axis=2).max())
        if spread > MAX_LOOK_SPREAD_M:
            return self._skip(f"pomiary rozjechane o {spread * 1000:.1f} mm")
        return self._finish_approach(np.median(raws, axis=0))

    def _measure_frame(self, image: np.ndarray, obs: Observation, planned: Planned) -> np.ndarray | str:
        """The vector from the TCP to the planned corner [m, base axes], or a calibration view's points."""
        cv2, K = self._cv2, self._K
        assert cv2 is not None and K is not None and self._fit is not None
        h, w = image.shape[:2]
        if self._size is not None and (w, h) != self._size:
            return f"klatka {w}x{h}, oczekiwano {self._size[0]}x{self._size[1]}"
        joints = dict(obs.joints)
        T = self.kin.camera(self.calib.wrist_camera, joints)
        dilate = max(1, round(MASK_DILATE_PX * w / WORK_WIDTH_PX))
        mask = jaw_mask(self.kin, joints, K, T, (h, w), cv2, dilate)
        if planned.kind == VIEW:
            pts = charuco.board_points(image, self._cv_board, self._detector, cv2, mask=mask,
                                       all_markers=True)
            if len(pts.obj) < MIN_VIEW_POINTS:
                return f"za malo punktow tablicy ({len(pts.obj)})"
            self._views.append(pts)
            return np.zeros(3)
        pts = charuco.board_points(image, self._cv_board, self._detector, cv2, mask=mask)
        view = charuco.board_pose(pts, K, self._dist, cv2)
        if not view.ok:
            return view.problem
        self._look_views.append(view)
        corner = self.board.corner_obj([planned.corner])[0] if planned.corner >= 0 else None
        if corner is None:
            return "brak rogu"
        return charuco.tcp_to_corner(view, corner, self._p_cam, self._fit.R_base_board)

    def _finish_approach(self, raw: np.ndarray | None) -> Action | None:
        planned = self._planned
        assert planned is not None
        self._attempted += 1
        self.calibration.index += 1
        if raw is not None:
            self._record(planned, raw)
        if planned.kind == SAMPLE:
            self._sample_done += 1
        self._planned = None
        return self.hold(self._approach_note())

    def _skip(self, why: str) -> Action | None:
        planned = self._planned
        assert planned is not None
        self._attempted += 1
        self.calibration.index += 1
        self.calibration.skipped += 1
        self._skipped.append({"corner": planned.corner, "level": planned.level,
                              "direction": planned.direction, "repeat": planned.repeat, "kind": planned.kind,
                              "why": why})
        logger.info("Pominiete podejscie (%s, rog c%d, poziom %d): %s", planned.kind, planned.corner,
                    planned.level + 1, why)
        self._planned = None
        if planned.kind == SAMPLE:
            self._sample_skips += 1
            total = self._sample_done + self._sample_skips + sum(p.kind == SAMPLE for p in self._queue)
            if self._sample_skips > self.calib.max_skip_share * max(total, 1):
                return self._fail(f"za duzo pominietych pomiarow: {self._sample_skips} z {total} "
                                  f"(ostatni: {why})")
        return self.hold(f"pominiete: {why}")

    # ------------------------------------------------------------------ the measurements
    def _delta(self, planned: Planned, raw: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
        """The correction [m, base frame] from a raw TCP-to-corner vector: what the commanded TCP lacked."""
        tcp = self.tcp_cmd
        corner_z = float(self._corners[planned.corner][2]) if planned.corner in self._corners else 0.0
        d = np.asarray(raw, float).copy()
        d[:2] += tcp[:2] - planned.xy
        d[2] += tcp[2] - corner_z
        if bias is not None:
            psi = heading(self.kin, self.kin.from_q(self.q_cmd))
            d[:2] = charuco.remove_bias(d[:2], psi, bias)
        return d

    def _record(self, planned: Planned, raw: np.ndarray) -> None:
        psi = heading(self.kin, self.kin.from_q(self.q_cmd))
        view = self._look_views[-1] if self._look_views else None
        if planned.kind in (MOUNT_A, MOUNT_B):
            self._mount_raw.append({"pair": planned.pair, "kind": planned.kind, "corner": planned.corner,
                                    "level": planned.level, "direction": planned.direction, "psi": psi,
                                    "xy": planned.xy.copy(), "delta": self._delta(planned, raw, None),
                                    "raw": np.asarray(raw, float).copy(), "tcp": self.tcp_cmd.copy(),
                                    "t_s": self._t - self._t_start,
                                    "points": 0 if view is None else view.points,
                                    "reproj_px": math.nan if view is None else view.reproj_px})
            return
        if planned.kind == CHECK:
            # What the corrected command still lacked; a look again after a turn of the wrist moved the
            # commanded TCP round the roll axis, so that move counts too.
            bias = self._bias.get(planned.level, np.zeros(2))
            missing = np.asarray(raw, float)[:2] + self.tcp_cmd[:2] - planned.xy
            left = charuco.remove_bias(missing, psi, bias)
            error = float(np.linalg.norm(left))
            self._verify.append({"corner": planned.corner, "level": planned.level,
                                 "direction": planned.direction, "xy_m": planned.xy, "error_m": error,
                                 "residual_m": left})
            self.calibration.verify_m.append(error)
            self.calibration.note = (f"sprawdzenie: rog c{planned.corner}, poziom {planned.level + 1}, "
                                     f"blad {error * 1000:.1f} mm")
            return
        delta = self._delta(planned, raw, self._bias.get(planned.level))
        self._samples.append({"spot": 0, "corner": planned.corner, "xy_m": planned.xy.copy(),
                              "level": planned.level, "direction": planned.direction,
                              "repeat": planned.repeat, "psi_deg": math.degrees(psi), "delta_m": delta,
                              "raw_m": np.asarray(raw, float),
                              "points": 0 if view is None else view.points,
                              "reproj_px": math.nan if view is None else view.reproj_px,
                              "t_s": round(self._t - self._t_start, 2)})
        st = self.calibration
        st.samples += 1
        st.last_delta_m = (float(delta[0]), float(delta[1]))

    def _solve_intrinsics(self) -> Action | None:
        cv2 = self._cv2
        assert cv2 is not None and self._camera is not None and self._size is not None
        W, H = self._size
        nominal = (H / 2) / math.tan(math.radians(self._camera.fovy_deg) / 2)
        result = charuco.calibrate(self._views, (W, H), nominal, cv2)
        st = self.calibration
        st.k_rms_px, st.k_views, st.focal_px = result.rms_px, result.views, float(result.K[0, 0])
        self._intrinsics = result
        if not result.trusted:
            return self._fail(f"kalibracja kamery nadgarstka nieudana: {result.reason}")
        self._K, self._dist = result.K, result.dist
        return self._after_intrinsics()

    def _solve_mount(self) -> None:
        by_pair: dict[int, dict[str, dict[str, Any]]] = {}
        for m in self._mount_raw:
            by_pair.setdefault(int(m["pair"]), {})[m["kind"]] = m
        per_level: dict[int, list[np.ndarray]] = {}
        pairs = []
        for pair, got in sorted(by_pair.items()):
            if MOUNT_A not in got or MOUNT_B not in got:
                continue
            a, b = got[MOUNT_A], got[MOUNT_B]
            Ra, Rb = charuco.rot2(a["psi"]), charuco.rot2(b["psi"])
            M = Ra - Rb
            if abs(float(np.linalg.det(M))) < 1e-3:
                continue
            # Only the wrist roll moved between the looks: the TCP went round the roll axis by what the
            # hand's geometry says, and the arm's own error stayed.
            moved = (b["tcp"] - a["tcp"])[:2]
            bias = np.linalg.solve(M, a["raw"][:2] - b["raw"][:2] - moved)
            mean = a["delta"] - np.array([*(Ra @ bias), 0.0])
            level = int(a["level"])
            per_level.setdefault(level, []).append(bias)
            pairs.append({"pair": pair, "corner": a["corner"], "level": level, "direction": a["direction"],
                          "psi_deg": [math.degrees(a["psi"]), math.degrees(b["psi"])], "bias_m": bias,
                          "raw_a_m": a["raw"], "raw_b_m": b["raw"], "tcp_moved_m": moved})
            self._samples.append({"spot": 0, "corner": a["corner"], "xy_m": np.asarray(a["xy"], float),
                                  "level": level, "direction": a["direction"], "repeat": 0,
                                  "psi_deg": math.degrees(a["psi"]), "delta_m": mean,
                                  "raw_m": a["raw"], "points": min(a["points"], b["points"]),
                                  "reproj_px": max(a["reproj_px"], b["reproj_px"]),
                                  "t_s": round(b["t_s"], 2)})
            self.calibration.samples += 1
        self._pairs = {p["pair"]: p for p in pairs}
        all_biases = [b for v in per_level.values() for b in v]
        fallback = np.mean(all_biases, axis=0) if all_biases else np.zeros(2)
        st = self.calibration
        for level in range(len(self._levels)):
            got = per_level.get(level, [])
            bias = np.mean(got, axis=0) if got else fallback
            self._bias[level] = np.asarray(bias, float)
            spread = (max(float(np.linalg.norm(x - y)) for x in got for y in got) if len(got) > 1 else 0.0)
            st.mount_bias_m[level] = (float(bias[0]), float(bias[1]))
            st.mount_spread_m[level] = spread
            if spread > MOUNT_SPREAD_WARN_M:
                logger.warning("Rozrzut obrotu chwytaka na poziomie %d: %.1f mm", level + 1, spread * 1000)
                st.note = f"uwaga: rozrzut obrotu chwytaka {spread * 1000:.1f} mm na poziomie {level + 1}"

    # ------------------------------------------------------------------ the table
    def _suggested(self) -> dict[str, Any]:
        """A `focal_px` and tip pixel for `grasp_wrist` from this run's camera, in the frame as received."""
        if self._K is None or self._size is None:
            return {}
        kin, cfg = self.kin, self.calib
        home = self._seed()
        T_cam, T_tcp = kin.camera(cfg.wrist_camera, home), kin.tcp(home)
        tip = (inverse(T_cam) @ T_tcp @ np.array([*kin.jaw_centre(0.0), 1.0]))[:3]
        bias = np.mean([self._bias[k] for k in self._bias], axis=0) if self._bias else np.zeros(2)
        R_tcp_cam = (inverse(T_tcp) @ T_cam)[:3, :3]
        tip = tip + R_tcp_cam.T @ np.array([0.0, bias[1], bias[0]])
        u = float(self._K[0, 0] * tip[0] / tip[2] + self._K[0, 2])
        v = float(self._K[1, 1] * tip[1] / tip[2] + self._K[1, 2])
        W, H = self._size
        if cfg.rotate_180:
            u, v = W - 1 - u, H - 1 - v
        return {"focal_px": round(float(self._K[0, 0]), 1), "tip_pixel": [round(u, 1), round(v, 1)]}

    def _build_table(self, complete: bool) -> ApproachTable:
        cfg, kin = self.calib, self.kin
        fit = self._fit
        intr = self._intrinsics
        wrist = {"camera": cfg.wrist_camera, "rotate_180": cfg.rotate_180,
                 "size": list(self._size) if self._size is not None else None,
                 "K": None if self._K is None else self._K, "dist": self._dist,
                 "rms_px": None if intr is None else intr.rms_px,
                 "views": None if intr is None else intr.views,
                 "tilt_spread_deg": None if intr is None else intr.tilt_spread_deg,
                 "tcp_in_camera_m": self._p_cam,
                 "mount_bias_m": {str(k): v for k, v in self._bias.items()},
                 "mount_spread_m": {str(k): v for k, v in self.calibration.mount_spread_m.items()},
                 "suggested": self._suggested()}
        protocol = {"directions": self.directions, "lead_m": cfg.lead_m,
                    "carry_clearance_m": cfg.carry_clearance_m,
                    "place_gap_m": cfg.place_gap_m, "place_speed_mps": cfg.place_speed_mps,
                    "speed_mps": cfg.speed_mps, "gripper": self._gripper, "hang_m": self._hang,
                    "repeats": cfg.repeats}
        levels = self._level_dicts()
        if self._existing is not None and cfg.merge:
            table = ApproachTable(dict(self._existing.data))
            spot = table.next_spot()
            table.data["wrist"] = wrist
            table.data["stand"] = cfg.stand or table.data.get("stand", "")
        else:
            table = ApproachTable.new(robot=kin.spec.name, stand=cfg.stand, board=self.board.as_dict(),
                                      protocol=protocol, wrist=wrist, levels=levels)
            spot = 1
        centre = fit.to_base(np.array(self.board.size_m) / 2) if fit is not None else np.full(2, math.nan)
        table.data["spots"] = list(table.data.get("spots", [])) + [{
            "id": spot, "created": now_text(), "board_xy_m": centre,
            "board_yaw_deg": math.degrees(fit.yaw) if fit is not None else None,
            "fit_rms_m": fit.rms_m if fit is not None else None,
            "fit_scale": fit.scale if fit is not None else None,
            "corners": {f"c{k}": p for k, p in self._corners.items()},
            "mount_bias_m": {str(k): v for k, v in self._bias.items()}, "K": self._K}]
        for key, new in (("samples", self._samples), ("mount_pairs", list(self._pairs.values())),
                         ("skipped", self._skipped), ("verify", self._verify)):
            table.data[key] = list(table.data.get(key, [])) + [dict(item, spot=spot) for item in new]
        table.data["time_s"] = float(table.data.get("time_s") or 0.0) + (self._t - self._t_start)
        table.data["complete"] = bool(complete)
        table.summarize()
        return table

    def _begin_save(self) -> Action | None:
        self._enter(SAVE)
        self.calibration.phase = SAVE
        return None

    def _save(self, obs: Observation) -> Action | None:
        cfg = self.calib
        table = self._build_table(complete=True)
        self._table = table
        path = table.save(cfg.table_file)
        self._saved = True
        st = self.calibration
        st.table_path = str(path)
        st.spread_m = dict(table.data.get("spread_m") or {})
        st.preferred = table.preferred_direction
        planned = len(self._samples) + sum(s["kind"] in (SAMPLE, MOUNT_A, MOUNT_B) for s in self._skipped)
        spread = st.spread_m.get(st.preferred, math.nan)
        check = f", sprawdzenie {np.median(st.verify_m) * 1000:.1f} mm" if st.verify_m else ""
        st.note = (f"zapisano tablice podejscia: {st.samples} z {planned} pomiarow, "
                   f"kierunek {st.preferred or '-'} (rozrzut {spread * 1000:.1f} mm){check}")
        logger.info("%s -> %s", st.note, path)
        return self._begin_park()

    def _write_partial(self) -> None:
        cfg = self.calib
        if self._saved or self._measure is False or not cfg.table_file.strip() or not (
                self._samples or self._mount_raw or self._skipped):
            return
        try:
            table = self._build_table(complete=False)
            path = Path(cfg.table_file).expanduser()
            partial = write_atomic(path.with_name(path.name + ".partial.json"), table.data)
            self.calibration.partial_path = str(partial)
        except Exception:
            logger.exception("Zapis czesciowej tablicy podejscia nie powiodl sie")

    # ------------------------------------------------------------------ the end
    def _begin_park(self) -> Action | None:
        self._enter(PARK)
        self.calibration.phase = PARK
        here = self.tcp_cmd
        top = max((carry for _, _, carry, _ in self._levels), default=float(here[2]))
        up = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, :3].copy()
        self._path = [Waypoint(np.array([*here[:2], max(here[2] + PARK_RISE_M, top)]), self.approach_step_m,
                               None, rotation=up)]
        self._wp, self._wp_t0 = 0, self._t
        self._joint_goal = None
        self._park_route: list[np.ndarray] | None = None
        self._planned = None
        return None

    def _park(self, obs: Observation) -> Action | None:
        if self._park_route is None:
            wp = self._path[0]
            if not (self._arrived(wp.point) or self._t - self._wp_t0 > self.calib.move_timeout_s):
                self._moving = True
                try:
                    action = self._drive(wp.point, wp.step_m, rotation=wp.rotation)
                finally:
                    self._moving = False
                if self._hit is None:
                    return action
                # Up would bring the arm onto itself: on to the joint move from here.
                self._hit = None
            start = np.clip(self.kin.to_q(self._start_joints), *self.limits)
            # A way back that would touch the arm's own body is not taken: the arm stays up.
            self._park_route = self._joint_route(self.q_cmd, start) or []
            if not self._park_route:
                self.calibration.note += " (bez powrotu do pozycji startowej)"
        while self._park_route or self._joint_goal is not None:
            if self._joint_goal is None:
                self._joint_goal = self._park_route.pop(0)
            self._moving = True
            try:
                action = self._joint_move(obs)
            finally:
                self._moving = False
            if self._hit is not None:
                self._hit, self._joint_goal, self._park_route = None, None, []
                self.calibration.note += " (bez powrotu do pozycji startowej)"
                break
            if action is not None:
                return action
        if self._failed:
            return self._halt(self._failed, self.kin.from_q(self.q_cmd))
        if self._probe_mode:
            return self._halt("sondowanie zakonczone", self.kin.from_q(self.q_cmd), success=True)
        return self._halt(self.calibration.note, self.kin.from_q(self.q_cmd), success=True)

    def _joint_move(self, obs: Observation) -> Action | None:
        """Step the joints toward `_joint_goal` within the speed limits; None once there."""
        goal = self._joint_goal
        assert goal is not None
        arm = self.kin.arm
        step = np.where(np.isfinite(self._max_step), self._max_step * JOINT_MOVE_SHARE, 0.05)
        delta = goal - self.q_cmd
        if float(np.abs(delta[arm]).max()) <= 1e-6 and self._command_on_target:
            self._joint_goal = None
            self.carrot = self.tcp_cmd.copy()
            return None
        # All joints in proportion: the move follows the straight line `_joint_path_clear` checked.
        q_new = self.q_cmd + delta / max(1.0, float(np.max(np.abs(delta) / step)))
        target = self.kin.from_q(q_new)
        contacts = self.kin.self_contacts(target)
        new = contacts - self._contacts
        if new:
            pairs = ", ".join(f"{a} - {b}" for a, b in sorted(new))
            return self._halt(f"ramie uderzyloby samo w siebie ({pairs})", target=self.kin.from_q(self.q_cmd))
        self.q_cmd, self._contacts = q_new, contacts
        self.tcp_cmd = self.kin.tcp(target)[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()
        return Action(target=target, note=self._note())

    def _fail(self, why: str) -> Action | None:
        """Stop measuring: save what there is as a partial table, then park."""
        self._failed = why
        self.calibration.note = why
        self._queue, self._planned = [], None
        self._write_partial()
        if self.status.phase in (CORNERS,) and not self._attempted:
            return self._halt(why, self.kin.from_q(self.q_cmd))
        return self._begin_park()

    def _halt(self, reason: str, target: Mapping[str, float], success: bool = False) -> Action:
        if self._moving and reason.startswith(SELF_CONTACT):
            # An approach's straight path would bring the arm onto itself: the target is not sent, the
            # approach is skipped and the next one starts with a joint move.
            self._hit = reason
            return self.hold()
        if not success and not self._saved:
            self._write_partial()
        self.calibration.phase = DONE
        return super()._halt(reason, target, success)

    def stop(self, reason: str) -> None:
        if self.started and not self._saved:
            self._write_partial()
        super().stop(reason)

    # ------------------------------------------------------------------ notes
    def _note_approach(self, planned: Planned) -> None:
        self.calibration.note = self._approach_note()

    def _approach_note(self) -> str:
        planned = self._planned
        st = self.calibration
        if planned is None:
            return st.note
        n, total = st.index + 1, max(self._phase_total, 1)
        if planned.kind == VIEW:
            return f"kalibracja kamery nadgarstka: ujecie {n} z {total}"
        if planned.kind in (MOUNT_A, MOUNT_B):
            return f"obrot chwytaka: para {planned.pair + 1} z {(total + 1) // 2}"
        if planned.kind == CHECK:
            last = f", blad {st.verify_m[-1] * 1000:.1f} mm" if st.verify_m else ""
            return f"sprawdzenie: rog c{planned.corner}, poziom {planned.level + 1}{last}"
        if planned.kind == PROBE:
            return f"sonda {n} z {total}: poziom {planned.level + 1}, kierunek {planned.direction}"
        return (f"rog c{planned.corner}, poziom {planned.level + 1} z {len(self._levels)}, kierunek "
                f"{planned.direction}, powtorzenie {planned.repeat} z {self.calib.repeats} "
                f"(pomiar {n} z {total})")

    def _note(self) -> str:
        phase = self.status.phase
        if phase in (INTRINSICS, MOUNT, SAMPLES, VERIFY) and self._planned is not None:
            return self._approach_note()
        if phase == PARK:
            return "powrot do pozycji startowej"
        return self.calibration.note or self.status.stopped_because or phase


def _optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


__all__ = ["CalibrateApproachSettings", "CalibrateApproachStrategy", "CalibrationStatus", "Planned"]
