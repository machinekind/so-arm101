"""Straddle a block from above, close the jaw and verify the hold through the lift.

The twin loads `grasp` through `machinekind_twin.strategies` and builds `GraspSettings`
from its configuration or panel. See the package README for settings.

`blocks.block` supplies the block's dimensions and yaw symmetry. The strategy plans at the
fingertips' midpoint and uses the model's jaw geometry to find the TCP targets. By default,
it opens at the hover, straddles below the catalog top height, closes and lifts.
With `probe: true`, a closed-jaw descent first measures the top from the stalled TCP height
and the jaw's reach. A probe that reaches its bottom with `need_contact: false` uses the nominal top.

The pose is latched for the run. The hand points down with the jaws across the block's short
side. A pose without orientation keeps the current closing heading. The reach check requires
all planned points to be reachable at one heading within the joint and collision limits.

A tip stall during the straddle triggers a rise and a retry half the short side away.
Retries alternate sides of the original aim within `max_attempts`. With `probe: true`, each retry
probes again. Without the probe, an empty close after a sideways retry tries the other side.
Any other empty close ends the run.
The close counts as held only after the jaw has left its opening and settled against the block.
The lift stops if the jaw stops squeezing or creeps closed. Success keeps the squeeze commanded.

Subclasses can refine the aim through `before_straddle`. With the probe on, `before_descent`
also refines the aim, and `after_touch` and `after_miss` judge the probe.
`after_grasp_failed` and `carrying` handle failures. `grasp_headings` supplies candidate headings.
`grasp_verdict` supplies the verdict for logs and judges."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo

from ..blocks import BlockSpec, block
from ..blocks.classic_world import BLOCKS
from ..kinematics import IKSolution, RobotKinematics
from .jaw import JawMonitor, JawReading
from .touch import (
    DESCEND,
    HOLD_WAIT_S,
    MIN_TOUCH_Z,
    SPEED_SHARE,
    WAIT,
    TouchSettings,
    TouchStrategy,
    normalised,
)

logger = logging.getLogger(__name__)

#: Probe depth below the nominal top before adding jaw reach [m].
#: It must exceed the stall threshold plus contact deflection.
PROBE_BELOW_M = 0.010
#: The lowest TCP height of the grasp point [m]: the jaw's sole stays above the table.
MIN_GRASP_Z = 0.012
#: How long the jaw may take to open, to start closing and to stand after the close [s].
JAW_WAIT_S = 3.0
#: The supervisor's gripper command is on the target within this many app units.
JAW_COMMAND_TOL = 0.5
#: The reach check's largest orientation error of a top-down IK solution [deg].
MAX_ROT_ERR_DEG = 5.0
#: Allowed errors in tilt from vertical and in the chosen closing heading [deg].
MAX_TILT_DEG = 8.0
#: After a jaw tip landed on the block, the jaw closes only once the TCP rose this much [m].
CLEAR_BEFORE_CLOSE_M = 0.010
#: Default jaw reach below the TCP [m], replaced by the model's value in `start`.
JAW_REACH_M = 0.0066
#: Retry offsets from the first aim in half short sides along the closing heading.
#: Positive offsets point toward the moving jaw.
SHIFTS = (1, -1, 2, -2)

#: Phases of a run after touch's (`TouchStatus.phase`).
RISE = "rise"
OPEN = "open"
STRADDLE = "straddle"
STRADDLED = "straddled"
CLOSE = "close"
LIFT_OFF = "lift_off"
CHECK = "check"
RETARGET = "retarget"

#: Display text for each grasp verdict.
VERDICTS = {
    "none": "brak werdyktu chwytu",
    "tip": "szczeka oparla sie o blok przy zjezdzie",
    "empty": "chwyt pusty",
    "held": "szczeka trzyma blok",
    "slipped": "blok wypadl ze szczeki",
    "crept": "blok zsuwa sie w szczekach",
    "lifted": "blok podniesiony",
}


@dataclass
class GraspSettings(TouchSettings):
    #: Key in `obs.objects`, normalized to lower case and single spaces. Empty uses `block`.
    label: str = ""
    #: Bottom TCP height of the probe [m above the table, base frame]. Ignored without `probe`.
    #: Zero uses the block dimensions and jaw reach. Explicit heights must be at least `MIN_TOUCH_Z`.
    touch_z: float = 0.0
    #: Hover offset above the probe's bottom TCP height [m].
    #: Without `probe`, the block dimensions set this reference height.
    #: A lower hover keeps more downward poses reachable.
    hover_m: float = 0.04
    #: Time limit for the run [s]. Zero disables it.
    time_limit_s: float = 120.0
    #: Block name for `blocks.block`, which supplies dimensions and yaw symmetry.
    block: str = field(default="cube", metadata={"choices": tuple(BLOCKS)})
    #: Measure the top with a closed-jaw descent before the straddle. False uses the catalog top height.
    #: Without the probe, `close_jaw`, `touch_z` and `need_contact` have no effect.
    probe: bool = False
    #: Each open tip clears the block's short side by this much when the jaw straddles it [m].
    open_margin_m: float = 0.008
    #: Squeeze below the block's short side [m]. It must exceed the jaw rule's holding margin.
    squeeze_m: float = 0.008
    #: Depth of the tip centres below the top as a fraction of block height (0, 1].
    grasp_depth: float = 0.5
    #: How far straight up the block is lifted before the final check [m].
    lift_m: float = 0.04
    #: How long the jaw must hold at the top before the run succeeds [s].
    hold_s: float = 1.0
    #: Maximum attempts from the hover, including retries after tip stalls [count].
    max_attempts: int = 3


@dataclass
class GraspStatus:
    """Grasp state alongside `TouchStatus` for logs and tests."""

    #: The block centre the run aims at [m, base frame]; None until the pose is taken.
    block_xy: tuple[float, float] | None = None
    #: The closing heading of the top-down hand [deg, base frame].
    heading_deg: float = float("nan")
    #: The block's top measured by the probe [m above the table]. NaN without `probe`.
    top_m: float = float("nan")
    #: The TCP point of the close [m, base frame].
    grasp_point: tuple[float, float, float] | None = None
    #: The gap of the commanded and of the measured jaw, between the tip centres [m].
    gap_cmd_m: float = float("nan")
    jaw_gap_m: float = float("nan")
    #: The jaw rule on the last tick.
    holding: bool = False
    #: Ticks in a row the jaw held at the lift height.
    hold_ticks: int = 0
    straddle_stalls: int = 0
    empty_closes: int = 0
    #: How far the measured TCP rose above the grasp point [m].
    lifted_m: float = float("nan")
    #: '', 'tip', 'empty', 'held', 'slipped', 'crept' or 'lifted'.
    verdict: str = ""


class GraspStrategy(TouchStrategy):
    """Grasps and lifts a block from above; the jaw rule checks the hold on the way up."""

    settings = GraspSettings

    def __init__(self, settings: GraspSettings | None = None):
        config = settings if settings is not None else GraspSettings()
        if config.block not in BLOCKS:
            raise ValueError(f"grasp: block must be one of {', '.join(BLOCKS)}, is {config.block!r}")
        # Validate here so errors use grasp's messages.
        if config.probe and not (config.touch_z == 0 or config.touch_z >= MIN_TOUCH_Z):
            raise ValueError(f"grasp: touch_z must be zero (from the block) or at least {MIN_TOUCH_Z} m, "
                             f"is {config.touch_z}")
        for name in ("open_margin_m", "squeeze_m", "lift_m", "hold_s"):
            value = getattr(config, name)
            if not value > 0:
                raise ValueError(f"grasp: {name} must be positive, is {value}")
        if not 0 < config.grasp_depth <= 1:
            raise ValueError(f"grasp: grasp_depth must lie within (0, 1], is {config.grasp_depth}")
        if not config.max_attempts >= 1:
            raise ValueError(f"grasp: max_attempts must be at least 1, is {config.max_attempts}")
        if config.probe and not config.close_jaw:
            raise ValueError("grasp: close_jaw must be on: the probe needs closed tips")
        spec = block(config.block)
        #: Block geometry in its body frame (x long, y short, z height) and yaw symmetry.
        self.block: BlockSpec = spec
        self.short_m = 2 * spec.half_size[1]
        self.height_m = 2 * spec.half_size[2]
        #: Nominal top above the table [m], used until a probe measures it.
        self.top_nominal = self.height_m
        #: The jaw's reach below the TCP [m], from the model once `start` has it.
        self._reach_m = JAW_REACH_M
        #: Original settings. `config` holds the resolved probe height and label.
        self.grasp_settings = config
        # Without the probe, block dimensions set the hover reference.
        self._auto_touch_z = config.touch_z == 0 or not config.probe
        label = normalised(config.label) or config.block
        super().__init__(replace(config, touch_z=self._probe_z(), label=label))

        self.grasp = GraspStatus()
        #: The rotation of the block's pose in the base frame, identity until a pose is taken.
        self.block_rotation = np.eye(3)
        #: TCP rotation and closing heading [rad, base frame]. None / NaN before latching a pose.
        self._rot: np.ndarray | None = None
        self._psi = float("nan")
        #: The candidate headings `_aim` tries, in order [rad].
        self._headings: list[float] = []
        #: Original block centre [m, base frame] and retry index.
        self._block_xy0: np.ndarray | None = None
        self._shift_k = 0
        self._retarget_xy: np.ndarray | None = None
        self._jaw: JawMonitor | None = None
        self._reading: JawReading | None = None
        self._g = -1
        self._jaw_goal_q = float("nan")
        self._grip_step_q = float("inf")
        self._jaw_tol_q = 0.0
        self._jaw_on_command = True
        #: Openings [m] and gripper commands [app units], resolved in `start`.
        self.gap_open_m = self.short_m + 2 * config.open_margin_m
        self.gap_closed_m = self.short_m - config.squeeze_m
        self.cmd_open = self.cmd_closed = self.cmd_block = self.cmd_probe = float("nan")
        #: The fingertips' midpoint in the TCP frame for the probe's and the open jaw [m].
        self._centre_probe = np.zeros(3)
        self._centre_open = np.zeros(3)
        self._rise_then = OPEN
        self._rise_z0 = float("nan")
        self._rise_point = np.zeros(3)
        # Straddle and lift TCP targets [m, base frame].
        self._straddle_xy = np.full(2, np.nan)
        self._grasp_z = float("nan")
        self._lift_off_point = np.zeros(3)
        #: Lift ticks spent moving the target or waiting for the supervisor's command [count].
        self._lift_ticks = 0
        # Initial closing angle [rad] and progress toward a settled jaw.
        self._q_open_entry = float("nan")
        self._jaw_moved = False
        self._still_ticks = 0
        #: When the hover was reached before the hand turned top-down [s, `obs.t`].
        self._turn_t0 = float("nan")
        # Hover hooks remain pending until they return None.
        self._before_descent_pending = False
        self._before_straddle_pending = False
        self._descent_waited = False
        self._phases.update({RISE: self._rise, OPEN: self._open, STRADDLE: self._straddle,
                             STRADDLED: self._straddled, CLOSE: self._close, LIFT_OFF: self._lift_off,
                             CHECK: self._check})

    def _probe_z(self) -> float:
        """Return the probe's bottom TCP height [m above the table, base frame].

        With the probe on, an explicit `touch_z` overrides the nominal top minus `PROBE_BELOW_M`
        plus the jaw's reach."""
        if not self._auto_touch_z:
            return float(self.grasp_settings.touch_z)
        return max(MIN_TOUCH_Z, self.top_nominal - PROBE_BELOW_M + self._reach_m)

    # ------------------------------------------------------------------- start
    def start(self, run: RunInfo, obs: Observation) -> None:
        """Initialize the probe and reject openings that the jaw cannot reach or distinguish as a hold."""
        super().start(run, obs)
        kin, spec, cfg = self.kin, self.kin.spec, self.grasp_settings
        if spec.gripper is None or spec.gripper_range_q is None:
            raise RuntimeError("grasp: the arm has no gripper")
        g = self._g = spec.joints.index(spec.gripper)
        closed, opened = spec.gripper_range_q
        try:
            self.cmd_probe = kin.from_q(self._lo)[spec.gripper]
            self._reach_m = kin.jaw_reach(self.cmd_probe)
            self.cmd_open = kin.gripper_for_gap(self.gap_open_m)
            self.cmd_closed = kin.gripper_for_gap(self.gap_closed_m)
            self.cmd_block = kin.gripper_for_gap(self.short_m)
        except ValueError as exc:
            raise RuntimeError(f"grasp: {exc}") from exc
        if self._auto_touch_z and abs(self._reach_m - JAW_REACH_M) > 1e-4:
            self.config = replace(self.config, touch_z=self._probe_z())
        self._centre_probe = kin.jaw_centre(self.cmd_probe)
        self._centre_open = kin.jaw_centre(self.cmd_open)

        jaw = spec.jaw
        margin_units = jaw.margin * 100.0 / (opened - closed)
        if self._q(self.cmd_block) - self._q(self.cmd_closed) <= jaw.margin + jaw.still:
            margin_gap = kin.jaw_gap(self.cmd_block) - kin.jaw_gap(max(0.0, self.cmd_block - margin_units))
            raise RuntimeError(f"grasp: squeeze_m {cfg.squeeze_m} m is within the jaw margin "
                               f"({margin_gap:.4f} m): a held block would not count as held")
        if self._q(self.cmd_open) - self._q(self.cmd_block) <= jaw.margin + jaw.still:
            raise RuntimeError(f"grasp: open_margin_m {cfg.open_margin_m} m is within the jaw margin: a jaw "
                               "closing on the block could not be told from one that never moved")

        self._jaw = JawMonitor(jaw, closed_q=float(self._lo[g]))
        self._reading = None
        # Leave speed margin so the supervisor can follow each target.
        vel = float(run.max_vel.get(spec.gripper, spec.max_vel.get(spec.gripper, np.inf)))
        self._grip_step_q = (vel * SPEED_SHARE / self.hz * (opened - closed) / 100.0 if vel > 0
                             else float("inf"))
        self._jaw_tol_q = JAW_COMMAND_TOL * (opened - closed) / 100.0
        self._jaw_goal_q = float(self.q_cmd[g])
        self._jaw_on_command = True

        self.grasp = GraspStatus()
        self.block_rotation = np.eye(3)
        self._rot, self._psi, self._headings = None, float("nan"), []
        self._block_xy0, self._shift_k, self._retarget_xy = None, 0, None
        self._straddle_xy, self._grasp_z = np.full(2, np.nan), float("nan")
        self._turn_t0 = float("nan")
        self._before_descent_pending = self._before_straddle_pending = self._descent_waited = False

    def _q(self, gripper: float) -> float:
        """A gripper command [app units] as the gripper joint's angle [rad]."""
        spec = self.kin.spec
        assert spec.gripper is not None
        return float(self.kin.to_q({spec.gripper: gripper})[self._g])

    # --------------------------------------------------------------------- tick
    def act(self, obs: Observation) -> Action:
        """Update the jaw reading and candidate headings before running the touch tick."""
        if self._kin is None or self._jaw is None:
            raise RuntimeError("grasp: act() przed start()")
        kin, g = self.kin, self._g
        q_measured = kin.to_q(obs.joints)
        # Missing command joints fall back to the last target.
        q_command = kin.to_q({**kin.from_q(self.q_cmd), **obs.command})
        self._reading = self._jaw.update(float(q_measured[g]), float(q_command[g]))
        self._jaw_on_command = abs(float(q_command[g] - self.q_cmd[g])) <= self._jaw_tol_q
        self.grasp.holding = self._reading.holding
        fixed, moving = kin.fingertips(obs.joints)
        self.grasp.jaw_gap_m = float(np.linalg.norm(moving - fixed))
        if self.status.phase == WAIT:
            self._take_heading(obs)
        return super().act(obs)

    @property
    def reading(self) -> JawReading:
        """Return this tick's jaw reading."""
        if self._reading is None:
            raise RuntimeError("grasp: no jaw reading before the first tick")
        return self._reading

    def _take_heading(self, obs: Observation) -> None:
        """Take candidate headings from the pose that `_wait` will latch this tick."""
        pose = {normalised(str(key)): value for key, value in obs.objects.items()}.get(self.label)
        if pose is None or self._pose_problem(pose):
            return
        rotation = np.asarray(pose.rotation, float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            rotation = np.eye(3)
        self._heading_from(rotation)

    def _heading_from(self, rotation: np.ndarray) -> None:
        """Store the block rotation and list candidate headings nearest the current heading first."""
        self.block_rotation = np.asarray(rotation, float).copy()
        self._headings = grasp_headings(self.block_rotation, self.block.yaw_period, self._heading_now())

    def _heading_now(self) -> float:
        """The closing heading of the last target [rad, base frame]."""
        return heading(self.kin, self.kin.from_q(self.q_cmd))

    # ------------------------------------------------------------------- aiming
    def _aim(self, xy: np.ndarray) -> None:
        """Choose the first heading that reaches every planned point above block centre `xy`.

        Raise without changing the aim if no heading works."""
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise RuntimeError("punkt poza zasiegiem ramienia: wspolrzedne nieskonczone")
        headings = self._headings or grasp_headings(self.block_rotation, self.block.yaw_period,
                                                    self._heading_now())
        top = self.grasp.top_m if np.isfinite(self.grasp.top_m) else self.top_nominal
        grasp_z = self._grasp_height(top)
        first_failure: tuple[np.ndarray, dict[str, float], IKSolution] | None = None
        for psi in headings:
            plan = self._plan(xy, psi, grasp_z)
            failure = self._reach(plan, psi)
            if failure is None:
                self._take_plan(xy, psi, plan, grasp_z)
                return
            first_failure = first_failure or failure
        assert first_failure is not None
        raise RuntimeError(self._unreachable(*first_failure))

    def _plan(self, xy: np.ndarray, psi: float, grasp_z: float) -> dict[str, np.ndarray]:
        """Return TCP targets that place the jaw midpoint over block centre `xy` at heading `psi`."""
        u = np.array([math.cos(psi), math.sin(psi)])
        hover_z = self.config.touch_z + self.config.hover_m
        # Jaw midpoint = TCP + R @ centre. With the hand down, centre[2] lies along the heading.
        probe_xy = xy - self._centre_probe[2] * u
        straddle_xy = xy - self._centre_open[2] * u
        return {"probe_xy": probe_xy, "straddle_xy": straddle_xy,
                "probe_hover": np.array([*probe_xy, hover_z]),
                "probe_bottom": np.array([*probe_xy, self.config.touch_z]),
                "straddle_hover": np.array([*straddle_xy, hover_z]),
                "grasp": np.array([*straddle_xy, grasp_z]),
                "lift": np.array([*straddle_xy, grasp_z + self.grasp_settings.lift_m])}

    def _reach(self, plan: dict[str, np.ndarray], psi: float
               ) -> tuple[np.ndarray, dict[str, float], IKSolution] | None:
        """Check that full IK reaches every planned point with the hand down.

        Return None on success, or the first failed point, its seed and its solution."""
        kin, R = self.kin, top_down(psi)
        spec = kin.spec
        assert spec.gripper is not None
        seed = kin.from_q(self.q_cmd)
        names = ("probe_hover", "probe_bottom") if self.grasp_settings.probe else ("probe_hover",)
        for name in (*names, "straddle_hover", "grasp", "lift"):
            if name == "straddle_hover":
                seed = dict(seed, **{spec.gripper: self.cmd_open})
            point = plan[name]
            sol = kin.ik(point, R, seed=seed, accept=self._clear)
            if not (sol.ok and self._clear(sol.joints) and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG)):
                return point, seed, sol
            seed = sol.joints
        return None

    def _unreachable(self, point: np.ndarray, seed: dict[str, float], sol: IKSolution) -> str:
        """Describe why the arm cannot reach `point` with the hand down."""
        where = f"punkt ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m"
        if sol.ok and self._clear(sol.joints):
            return f"{where}: chwytak nie ustawi sie pionowo (blad {math.degrees(sol.rot_err):.0f} st.)"
        reachable = self.kin.ik(point, seed=seed, accept=self._within_limits)
        if not (reachable.ok and self._within_limits(reachable.joints)):
            return f"{where} poza zasiegiem ramienia"
        if not self.kin.self_contacts(reachable.joints):
            return f"{where}: chwytak nie siega tam pionowo"
        return f"{where}: ramie uderzyloby samo w siebie"

    def _take_plan(self, xy: np.ndarray, psi: float, plan: dict[str, np.ndarray], grasp_z: float) -> None:
        self._psi, self._rot = float(psi), top_down(psi)
        self._headings = [float(psi)]
        self.grasp.heading_deg = math.degrees(psi)
        self.touch_xy = plan["probe_xy"].copy()
        x, y = (float(v) for v in self.touch_xy)
        self.status.touch_point = (x, y, float(self.config.touch_z))
        self.grasp.block_xy = (float(xy[0]), float(xy[1]))
        if self._block_xy0 is None:
            self._block_xy0 = xy.copy()
        self._straddle_xy = plan["straddle_xy"].copy()
        self._grasp_z = float(grasp_z)
        self.grasp.grasp_point = (float(self._straddle_xy[0]), float(self._straddle_xy[1]), self._grasp_z)

    def _grasp_height(self, top_m: float) -> float:
        """Return the closing TCP height [m above the table, base frame].

        Place the tip centres `grasp_depth` of the block height below its top, bounded by `MIN_GRASP_Z`."""
        depth = self.grasp_settings.grasp_depth * self.height_m
        return max(MIN_GRASP_Z, top_m - depth + float(self._centre_open[0]))

    @property
    def straddle_hover(self) -> np.ndarray:
        """Where the open jaw waits above the block before the straddle [m, base frame]."""
        return np.array([*self._straddle_xy, self.config.touch_z + self.config.hover_m])

    @property
    def grasp_target(self) -> np.ndarray:
        """Where the TCP closes the jaw [m, base frame]."""
        return np.array([*self._straddle_xy, self._grasp_z])

    def _drive(self, goal: np.ndarray, step_m: float, *, rotation: np.ndarray | None = None) -> Action:
        """Run touch's step with the chosen downward orientation after latching the pose."""
        return super()._drive(goal, step_m, rotation=self._rot if rotation is None else rotation)

    def _vertical(self) -> bool:
        """Whether the last target's hand is top-down on the chosen heading within `MAX_TILT_DEG`."""
        joints = self.kin.from_q(self.q_cmd)
        limit = math.radians(MAX_TILT_DEG)
        return tilt(self.kin, joints) <= limit and abs(wrap(heading(self.kin, joints) - self._psi)) <= limit

    def _enter(self, phase: str) -> None:
        super()._enter(phase)
        if phase == DESCEND:
            self._before_descent_pending, self._descent_waited = True, False
        if phase == OPEN:
            self._before_straddle_pending = True

    # ------------------------------------------------------------------ the hover and the probe
    def _approach(self, obs: Observation) -> Action | None:
        """Wait at the hover for the hand to point down, then descend for the probe or open the jaw."""
        if self._rot is None or self._vertical():
            self._turn_t0 = float("nan")
            if self.grasp_settings.probe:
                return super()._approach(obs)
            if self._settled(self.hover_point):
                self.status.attempt += 1
                # OPEN times its travel to the straddle hover from here.
                self._rise_point = self.hover_point
                self._enter(OPEN)
                return None
            return self._drive(self.hover_point, self.approach_step_m)
        # Finish turning at the hover within `JAW_WAIT_S`.
        if self._arrived(self.hover_point):
            if not np.isfinite(self._turn_t0):
                self._turn_t0 = self._t
            elif self._t - self._turn_t0 >= JAW_WAIT_S:
                return self._halt(self._not_vertical(), self.kin.from_q(self.q_cmd))
        self._settle_count = 0
        return self._drive(self.hover_point, self.approach_step_m)

    def _not_vertical(self) -> str:
        return (f"chwytak nie ustawil sie pionowo nad blokiem "
                f"({math.degrees(tilt(self.kin, self.kin.from_q(self.q_cmd))):.0f} st.)")

    def _descend(self, obs: Observation) -> Action | None:
        if self._before_descent_pending:
            action = self.before_descent(obs)
            if action is not None:
                self._descent_waited = True
                return action
            self._before_descent_pending = False
            if self._descent_waited:
                # Refresh the descent reference after the hook held or moved the arm.
                self._bias_z, self._hover_z = self._offset_z, float(self._tcp_measured[2])
                self._settle_count = 0
        return super()._descend(obs)

    def before_descent(self, obs: Observation) -> Action | None:
        """Run before each probe descent tick until this hook returns None. Never runs without `probe`.

        Overrides may hold or move at hover height and re-aim through `_aim`. Return None only
        when settled at the current hover point. The default returns None immediately."""
        return None

    def after_touch(self, obs: Observation) -> Action:
        """Use a touch to measure the block's top and start the grasp.

        A probe that reaches its bottom uses the nominal top. Other verdicts go to `after_miss`."""
        kind, reason = self.descent_verdict()
        if kind not in ("touched", "reached"):
            return self.after_miss(kind, reason, obs)
        top = self.status.stop_z_m - self._reach_m if kind == "touched" else self.top_nominal
        self.grasp.top_m = float(top)
        self._grasp_z = self._grasp_height(top)
        self.grasp.grasp_point = (float(self._straddle_xy[0]), float(self._straddle_xy[1]), self._grasp_z)
        self._start_rise(OPEN)
        return self.hold("w gore po dotyku")

    def after_miss(self, kind: str, reason: str, obs: Observation) -> Action:
        """Handle a probe verdict of "missed", "blocked" or "stuck".

        A miss after a sideways shift tries the other side while attempts remain.
        Other failures end the run with touch's reason."""
        attempt = self.status.attempt
        if kind == "missed" and self._shift_k >= 1:
            if attempt < self.grasp_settings.max_attempts and self._next_shift() is not None:
                self._start_rise(RETARGET)
                return self.hold("obok pusto - probuje z drugiej strony")
            return self._halt(f"wyczerpane proby chwytu ({attempt}): {reason}", self.kin.from_q(self.q_cmd))
        return self._halt(reason, self.kin.from_q(self.q_cmd))

    # ------------------------------------------------------------------ the grasp
    def _start_rise(self, then: str) -> None:
        """Start a vertical rise to hover height, followed by `then` (OPEN or RETARGET)."""
        self._rise_then = then
        self._rise_z0 = float(self._tcp_measured[2])
        hover_z = self.config.touch_z + self.config.hover_m
        self._rise_point = np.array([self.tcp_cmd[0], self.tcp_cmd[1], max(hover_z, float(self.tcp_cmd[2]))])
        self._enter(RISE)

    def _rise(self, obs: Observation) -> Action | None:
        # Let the supervisor reach the held pose before rising.
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        arrived = self._arrived(self._rise_point)
        # Clear the block before closing so the moving tip cannot sweep across its top.
        if arrived or self._tcp_measured[2] >= self._rise_z0 + CLEAR_BEFORE_CLOSE_M:
            self._jaw_goal_q = float(self._lo[self._g])
            self._ramp_jaw()
        if arrived and self._jaw_ready() and self._jaw_goal_q == float(self._lo[self._g]):
            if self._rise_then == OPEN:
                self._enter(OPEN)
                return None
            assert self._retarget_xy is not None
            try:
                self.retarget(float(self._retarget_xy[0]), float(self._retarget_xy[1]))
            except RuntimeError as exc:
                return self._halt(str(exc), self.kin.from_q(self.q_cmd))
            return self.hold("ponowna proba obok")
        return self._drive(self._rise_point, self.approach_step_m)

    def _open(self, obs: Observation) -> Action | None:
        self._set_jaw_goal(self.cmd_open)
        self._ramp_jaw()
        reading = self.reading
        settled = self._settled(self.straddle_hover)
        jaw_open = abs(reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin
        vertical = self._vertical()
        ready = settled and self._jaw_ready() and jaw_open and vertical
        if self._before_straddle_pending and (ready or self._looking):
            action = self.before_straddle(obs, settled=ready)
            if action is not None:
                return action
            self._before_straddle_pending = False
        if ready:
            # Refresh the stall reference after moving the TCP and opening the jaw.
            self._bias_z, self._hover_z = self._offset_z, float(self._tcp_measured[2])
            self._enter(STRADDLE)
            return None
        travel = float(np.linalg.norm(self.straddle_hover - self._rise_point)) / self.config.speed_mps
        if not self._looking and self._t - self._phase_t0 >= JAW_WAIT_S + travel:
            if not jaw_open:
                gaps = f"{self.grasp.jaw_gap_m * 1000:.0f} mm zamiast {self.gap_open_m * 1000:.0f} mm"
                return self._halt(f"szczeka nie otworzyla sie w {JAW_WAIT_S:.0f} s ({gaps})",
                                  self.kin.from_q(self.q_cmd))
            if not vertical:
                return self._halt(self._not_vertical(), self.kin.from_q(self.q_cmd))
            distance = float(np.linalg.norm(self.straddle_hover - self._tcp_measured)) * 1000
            return self._halt(f"ramie nie stanelo nad blokiem ({distance:.0f} mm od punktu)",
                              self.kin.from_q(self.q_cmd))
        return self._drive(self.straddle_hover, self.approach_step_m)

    def before_straddle(self, obs: Observation, settled: bool) -> Action | None:
        """Run at the open-jaw hover once settled, or on every tick while `_looking` is true.

        Return None to finish this hook for the straddle. Descent waits for the arm and jaw
        to settle at the current hover point. The default returns None immediately."""
        return None

    @property
    def _looking(self) -> bool:
        """Whether a subclass is inspecting the block at the straddle hover."""
        return False

    def _straddle(self, obs: Observation) -> Action | None:
        st = self.status
        st.rise_m = self._offset_z - self._bias_z
        if st.rise_m > self.config.contact_gap_m:
            return self._straddle_stall()
        if self._settled(self.grasp_target):
            if st.rise_m > self.config.contact_gap_m / 2:
                return self._straddle_stall()
            self._enter(CLOSE)
            self._q_open_entry = self.reading.q
            self._jaw_moved, self._still_ticks = False, 0
            return None
        return self._drive(self.grasp_target, self.descend_step_m)

    def _straddle_stall(self) -> Action:
        """Hold the measured pose when a jaw tip stalls on the block."""
        self.q_cmd = np.clip(self.kin.to_q(self._hold_pose(self._measured)), self._lo, self._hi)
        self._contacts = self.kin.self_contacts(self.kin.from_q(self.q_cmd))
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()
        self.grasp.straddle_stalls += 1
        self.grasp.verdict = "tip"
        self._enter(STRADDLED)
        return self.hold("szczeka na bloku")

    def _straddled(self, obs: Observation) -> Action:
        """Wait for the supervisor's command to reach the held pose before handling the failure."""
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        return self.after_grasp_failed("tip", obs)

    def _close(self, obs: Observation) -> Action | None:
        self._set_jaw_goal(self.cmd_closed)
        self._ramp_jaw()
        reading, jaw = self.reading, self.kin.spec.jaw
        # A jaw that has not started closing can satisfy the holding rule.
        self._jaw_moved |= reading.q <= self._q_open_entry - jaw.margin
        self._still_ticks = self._still_ticks + 1 if self._jaw_moved and reading.still else 0
        if self._jaw_ready() and self._still_ticks >= self.settle_ticks:
            if reading.holding:
                assert self._jaw is not None
                self._jaw.mark()
                self.grasp.verdict = "held"
                self._lift_off_point = self.grasp_target + np.array([0.0, 0.0, self.grasp_settings.lift_m])
                self._lift_ticks = 0
                self._enter(LIFT_OFF)
                return None
            self.grasp.empty_closes += 1
            self.grasp.verdict = "empty"
            return self.after_grasp_failed("empty", obs)
        # Allow for float rounding of `obs.t`, so a whole number of ticks ends the wait on time.
        if self._t - self._phase_t0 >= JAW_WAIT_S - 1e-9:
            if not self._jaw_moved:
                return self._halt(f"szczeka nie ruszyla po rozkazie zamkniecia "
                                  f"({self.grasp.jaw_gap_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd))
            return self._halt(f"szczeka nie staje {JAW_WAIT_S:.0f} s po zamknieciu - serwo albo odczyt",
                              self.kin.from_q(self.q_cmd))
        return self.hold()

    def _lift_off(self, obs: Observation) -> Action | None:
        self.grasp.lifted_m = float(self._tcp_measured[2]) - self._grasp_z
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        if self._arrived(self._lift_off_point):
            self.grasp.hold_ticks = 0
            self._enter(CHECK)
            return None
        # Extend travel time while targets advance or the supervisor follows under a speed cap.
        on_point = float(np.linalg.norm(self._lift_off_point - self.tcp_cmd)) <= self.ik_tol
        if (not on_point and not self.status.out_of_reach) or not self._command_on_target:
            self._lift_ticks += 1
        # Allow `JAW_WAIT_S` after travel for the measured arm to arrive.
        lift_m = self.grasp_settings.lift_m
        travel = max(lift_m / self.config.speed_mps, self._lift_ticks / self.hz)
        if self._t - self._phase_t0 >= travel + JAW_WAIT_S:
            return self._halt(f"ramie nie podnioslo bloku ({self.grasp.lifted_m * 1000:.0f} mm z "
                              f"{lift_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd))
        return self._drive(self._lift_off_point, self.approach_step_m)

    def _check(self, obs: Observation) -> Action:
        self.grasp.lifted_m = float(self._tcp_measured[2]) - self._grasp_z
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        grasp = self.grasp
        grasp.hold_ticks = grasp.hold_ticks + 1 if self.reading.holding else 0
        if grasp.hold_ticks >= math.ceil(self.grasp_settings.hold_s * self.hz):
            grasp.verdict = "lifted"
            return self._halt(f"blok w szczekach, podniesiony o {grasp.lifted_m * 1000:.0f} mm "
                              f"(szczeka {grasp.jaw_gap_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd),
                              success=True)
        if self._t - self._phase_t0 >= self.grasp_settings.hold_s + JAW_WAIT_S:
            return self._halt("szczeka nie stoi po podniesieniu - blok sie rusza w szczekach",
                              self.kin.from_q(self.q_cmd))
        return self.hold()

    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        """Check the jaw at the start of each LIFT_OFF and CHECK tick.

        Return an action to interrupt the phase or None to continue. The default stops on loss
        of squeeze or creep since the close. Stillness is required only for the final hold check."""
        if not reading.squeezing:
            self.grasp.verdict = "slipped"
            return self._halt("blok wypadl ze szczeki podczas podnoszenia", self.kin.from_q(self.q_cmd))
        if reading.crept:
            self.grasp.verdict = "crept"
            return self._halt("blok zsuwa sie w szczekach", self.kin.from_q(self.q_cmd))
        return None

    def after_grasp_failed(self, why: str, obs: Observation) -> Action:
        """Handle "tip" or "empty" while holding the arm's pose.

        A tip stall retries half the short side away while attempts remain. Without the probe,
        an empty close after a sideways retry tries the other side within the same attempt budget.
        Any other empty close ends the run. Overrides that await a judge must enter their own phase."""
        attempt = self.status.attempt
        if why == "empty":
            if not self.grasp_settings.probe and self._shift_k >= 1:
                if attempt < self.grasp_settings.max_attempts and self._next_shift() is not None:
                    self._start_rise(RETARGET)
                    return self.hold("chwyt pusty obok - probuje z drugiej strony")
                return self._halt(f"wyczerpane proby chwytu ({attempt}): chwyt pusty",
                                  self.kin.from_q(self.q_cmd))
            return self._halt("chwyt pusty - blok przesuniety albo obok szczek", self.kin.from_q(self.q_cmd))
        if attempt >= self.grasp_settings.max_attempts or self._next_shift() is None:
            return self._halt(f"wyczerpane proby chwytu ({attempt}): szczeka oparla sie o blok przy zjezdzie",
                              self.kin.from_q(self.q_cmd))
        self._start_rise(RETARGET)
        return self.hold("szczeka na bloku - w gore i obok")

    def _next_shift(self) -> np.ndarray | None:
        """Store and return the next block centre from `SHIFTS`, or None when shifts are exhausted."""
        self._shift_k += 1
        if self._shift_k > len(SHIFTS) or self._block_xy0 is None:
            return None
        shift = SHIFTS[self._shift_k - 1] * self.short_m / 2
        self._retarget_xy = self._block_xy0 + shift * np.array([math.cos(self._psi), math.sin(self._psi)])
        return self._retarget_xy

    def grasp_verdict(self) -> tuple[str, str]:
        """Return the grasp verdict and its text for logs and judges."""
        verdict = self.grasp.verdict or "none"
        return verdict, VERDICTS.get(verdict, verdict)

    # ------------------------------------------------------------------ the jaw
    def _set_jaw_goal(self, gripper: float) -> None:
        self._jaw_goal_q = float(np.clip(self._q(gripper), self._lo[self._g], self._hi[self._g]))

    def _ramp_jaw(self) -> None:
        """Advance the gripper target toward its goal at `SPEED_SHARE` of the supervisor's speed limit."""
        g = self._g
        delta = float(np.clip(self._jaw_goal_q - self.q_cmd[g], -self._grip_step_q, self._grip_step_q))
        before = float(self.q_cmd[g])
        self.q_cmd[g] = float(np.clip(before + delta, self._lo[g], self._hi[g]))
        if self.q_cmd[g] != before:
            # The supervisor has not yet received the new target.
            self._jaw_on_command = False
        spec = self.kin.spec
        assert spec.gripper is not None
        self.grasp.gap_cmd_m = self.kin.jaw_gap(self.kin.from_q(self.q_cmd)[spec.gripper])

    def _jaw_ready(self) -> bool:
        """Whether the gripper target reached its goal and the supervisor's command reached that target."""
        return abs(float(self.q_cmd[self._g]) - self._jaw_goal_q) <= 1e-9 and self._jaw_on_command

    # ------------------------------------------------------------------ the panel
    def _note(self) -> str:
        st, grasp = self.status, self.grasp
        mm = st.distance * 1000
        if st.phase == RISE:
            text = ("w gore po dotyku" if self._rise_then == OPEN else "w gore znad bloku") + f": {mm:.0f} mm"
        elif st.phase == OPEN:
            text = f"nad blok, otwieram szczeke: {grasp.gap_cmd_m * 1000:.0f} mm"
        elif st.phase == STRADDLE:
            text = f"w dol na blok: {mm:.0f} mm"
        elif st.phase == STRADDLED:
            text = "szczeka na bloku - czekam na rozkaz"
        elif st.phase == CLOSE:
            text = "sprawdzam chwyt" if self._jaw_ready() else "zamykam szczeke na bloku"
        elif st.phase == LIFT_OFF:
            text = f"podnosze blok: {mm:.0f} mm"
        elif st.phase == CHECK:
            text = f"trzymam blok: {grasp.hold_ticks / self.hz:.1f} s"
        else:
            return super()._note()
        return text + (" - cel poza zasiegiem ramienia" if st.out_of_reach else "")


# ------------------------------------------------------------------ geometry
def top_down(psi: float) -> np.ndarray:
    """Return the TCP rotation with approach down and closing along heading `psi` [rad, base frame].

    The columns are approach, y and closing, corresponding to TCP x, y and z."""
    c, s = math.cos(psi), math.sin(psi)
    return np.column_stack([(0.0, 0.0, -1.0), (-s, c, 0.0), (c, s, 0.0)])


def wrap(angle: float) -> float:
    """`angle` folded into (-pi, pi] [rad]."""
    folded = (angle + math.pi) % (2 * math.pi) - math.pi
    return math.pi if folded == -math.pi else folded


def tilt(kin: RobotKinematics, joints: Mapping[str, float]) -> float:
    """How far the approach of `joints` leans from straight down [rad]."""
    approach, _ = kin.tool_axes(joints)
    return math.acos(float(np.clip(-approach[2], -1.0, 1.0)))


def heading(kin: RobotKinematics, joints: Mapping[str, float]) -> float:
    """Return the horizontal heading of the closing axis [rad, base frame]."""
    _, closing = kin.tool_axes(joints)
    return math.atan2(float(closing[1]), float(closing[0]))


def grasp_headings(rotation: np.ndarray, yaw_period: float, current: float) -> list[float]:
    """Return candidate closing headings [rad], nearest `current` first.

    An identity rotation keeps `current` and its half-turn mirror. Otherwise the jaws close
    across the block's short side. The body's x axis defines its long side. `yaw_period`
    sets the equivalent headings, each with a half-turn mirror."""
    R = np.asarray(rotation, float)
    if np.allclose(R, np.eye(3), atol=1e-6):
        return [wrap(current), wrap(current + math.pi)]
    base = math.atan2(float(R[1, 0]), float(R[0, 0])) + math.pi / 2
    out: list[float] = []
    for k in range(max(1, round(2 * math.pi / yaw_period))):
        for mirror in (0.0, math.pi):
            psi = wrap(base + k * yaw_period + mirror)
            if all(abs(wrap(psi - other)) > 1e-6 for other in out):
                out.append(psi)
    return sorted(out, key=lambda psi: abs(wrap(psi - current)))

